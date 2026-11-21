"""Engine: o loop que orquestra tudo.

Ciclo, uma vez por candle de 1 minuto FECHADO:

    novo candle de 1min?
      -> recalcula features do 1min + anexa contexto do 5min (ja fechado)
      -> reconcilia a posicao contra o broker
      -> tem posicao?  SIM -> checa stop / alvo / fim de pregao
                       NAO -> decisor opina -> risco aprova/dimensiona -> broker executa
      -> registra tudo no journal

Dois pontos de desenho que importam:

ORIENTADO A EVENTO, NAO A TIMER. O passo roda quando NASCE uma barra nova,
nao a cada 60 segundos de relogio. Timer cego processa a mesma barra duas
vezes (duas ordens para um sinal) ou perde uma barra quando o feed atrasa.

RECONCILIACAO ANTES DE DECIDIR. A posicao real vem do broker, nao da
memoria do processo. Se divergirem, o robo PARA: operar sobre uma crenca
errada de posicao e como o dinheiro desaparece de verdade.
"""
from __future__ import annotations

import logging
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from .config import Config
from .data.base import MarketFeed, MarketSnapshot, detect_staleness
from .decision.base import Decisor, Signal
from .execution.base import Broker
from .features import FeatureSnapshot, build_feature_frame, latest_snapshot
from .journal import Journal
from .risk import Position, RiskManager

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    acted: bool = False
    reason: str = ""
    signal: Signal | None = None
    bar_time: pd.Timestamp | None = None
    features: FeatureSnapshot | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class TradingEngine:
    def __init__(self, cfg: Config, feed: MarketFeed, broker: Broker,
                 decisor: Decisor, journal: Journal) -> None:
        self.cfg = cfg
        self.feed = feed
        self.broker = broker
        self.decisor = decisor
        self.journal = journal
        self.risk = RiskManager(cfg)
        self.position: Position | None = None
        self._last_bar: pd.Timestamp | None = None
        self._bar_index = 0
        self._last_m5_time: Any = None

        journal.write("engine_start", {
            "decisor": decisor.name,
            "broker": broker.name,
            "sends_real_orders": broker.sends_real_orders,
            "symbol_root": cfg.symbol.root,
            "allow_long": cfg.direction.allow_long,
            "allow_short": cfg.direction.allow_short,
            "max_contracts": cfg.risk.max_contracts,
            "max_daily_loss_brl": cfg.risk.max_daily_loss_brl,
        })

    # --- um ciclo -----------------------------------------------------------

    def step(self, now: datetime | None = None, force: bool = False) -> StepResult:
        try:
            snap = self.feed.snapshot()
        except Exception as exc:  # noqa: BLE001
            self.journal.error("feed.snapshot", exc)
            return StepResult(reason=f"falha no feed: {exc}")

        now = now or datetime.now()
        bar_time = snap.m1.last_time

        # Sem barra nova: normalmente nao ha o que fazer, e sair aqui e o que
        # impede ordem duplicada para o mesmo sinal.
        #
        # MAS com posicao aberta nao podemos sair: stop, alvo e fim de pregao
        # dependem do RELOGIO, nao da chegada de barra. Se o feed parar de
        # produzir barras as 17:30, sair aqui significaria nunca zerar e
        # carregar posicao overnight - exatamente o que o robo deve evitar.
        bar_is_new = self._last_bar is None or bar_time > self._last_bar
        if not force and not bar_is_new and self.position is None:
            return StepResult(reason="nenhuma barra nova", bar_time=bar_time)

        if self.risk.roll_day_if_needed(now):
            self.journal.write("day_roll", {"day": str(now.date())})

        # Dado velho: feed travado, conexao caida ou pregao fechado.
        #
        # SEM posicao: nao decide nada. Entrar com dado velho e operar um
        # preco que nao existe mais.
        #
        # COM posicao: NAO da para simplesmente nao fazer nada - isso deixa
        # risco aberto sem ninguem olhando o stop nem o fim do pregao. O
        # fail-safe e zerar a mercado e parar o dia: ficar exposto as cegas
        # e pior do que sair num preco ruim.
        age = detect_staleness(snap.m1, 1, now=pd.Timestamp(now))
        if age > 3.0 and not force:
            stale_info = {"idade_em_barras": round(age, 1), "ultima_barra": str(bar_time)}
            if self.position is not None:
                logger.critical(
                    "DADO VELHO (%.1f candles) COM POSICAO ABERTA - zerando a mercado "
                    "e parando o dia", age,
                )
                self.journal.kill_switch("dado velho com posicao aberta", stale_info)
                res = self.broker.close_all(snap.price, comment="stale_feed")
                self.journal.order_result(res.ok, {"action": "close_on_stale",
                                                   "resultado": str(res)})
                if res.ok:
                    pnl = self.risk.register_exit(
                        self.position, res.filled_price or snap.price, self._bar_index)
                    self.journal.position_closed({
                        "motivo": "stale_feed", "pnl_brl": round(pnl, 2),
                        "side": self.position.side, "contracts": self.position.contracts,
                    })
                    self.position = None
                else:
                    logger.critical("NAO consegui zerar com feed velho - verifique o terminal!")
                self.risk.halt("feed travado com posicao aberta")
                return StepResult(reason=f"feed velho ({age:.1f} candles): zerou e parou",
                                  bar_time=bar_time)
            self.journal.risk_block("dado velho", stale_info)
            return StepResult(reason=f"ultima barra tem {age:.1f} candles de idade", bar_time=bar_time)

        if bar_is_new:
            self._last_bar = bar_time
            self._bar_index += 1

        # --- features: 1min + contexto do 5min ------------------------------
        try:
            frame = build_feature_frame(snap.m1.df, snap.m5.df,
                                        **self._rule_feature_kwargs())
            feats = latest_snapshot(frame)
        except Exception as exc:  # noqa: BLE001
            self.journal.error("build_feature_frame", exc)
            return StepResult(reason=f"falha nas features: {exc}", bar_time=bar_time)

        # registra quando o contexto de 5min virou - e a "analise de 5min"
        if feats.raw.get("m5_time") != self._last_m5_time:
            self._last_m5_time = feats.raw.get("m5_time")
            self.journal.write("m5_context_update", {
                "m5_bar": self._last_m5_time, "regime": feats.regime,
                "m5_slope_atr": feats.get("m5_slope_atr"),
                "m5_atr_rank": feats.get("m5_atr_rank"),
            })

        # --- reconciliacao --------------------------------------------------
        if (problema := self._reconcile()) is not None:
            self.risk.halt(problema)
            self.journal.kill_switch("divergencia de posicao", {"detalhe": problema})
            return StepResult(reason=problema, bar_time=bar_time, features=feats)

        # --- com posicao: cuidar dela ---------------------------------------
        if self.position is not None:
            return self._manage_open_position(snap, feats, now, bar_time)

        # --- sem posicao: avaliar entrada -----------------------------------
        return self._evaluate_entry(snap, feats, now, bar_time)

    def _rule_feature_kwargs(self) -> dict[str, Any]:
        rp = self.cfg.decisor_params("rules")
        out = {}
        for src, dst in (("ema_fast", "ema_fast"), ("ema_slow", "ema_slow"),
                         ("rsi_period", "rsi_period"), ("breakout_lookback", "breakout_lookback")):
            if src in rp:
                out[dst] = int(rp[src])
        return out

    # --- posicao aberta -----------------------------------------------------

    def _manage_open_position(self, snap: MarketSnapshot, feats: FeatureSnapshot,
                              now: datetime, bar_time: pd.Timestamp) -> StepResult:
        pos = self.position
        assert pos is not None
        bar = snap.m1.df.iloc[-1]
        low, high, price = float(bar["low"]), float(bar["high"]), snap.price

        # kill switch considerando o que esta em aberto
        unreal = pos.unrealized_brl(price, self.cfg.symbol.point_value)
        self.risk.check_kill_switch(open_position_pnl=unreal)

        decision = self.risk.check_exit(pos, low, high, price, now)
        if decision is None:
            self.journal.write("holding", {
                "bar": str(bar_time), "side": pos.side, "contracts": pos.contracts,
                "entry": pos.entry_price, "price": price,
                "unrealized_brl": round(unreal, 2),
                "stop": pos.stop_price, "target": pos.target_price,
            })
            return StepResult(reason="mantendo posicao", bar_time=bar_time, features=feats,
                              extra={"unrealized_brl": round(unreal, 2)})

        reason, exit_hint = decision
        res = self.broker.close_all(exit_hint, comment=f"exit:{reason}")
        self.journal.order_result(res.ok, {"action": "close", "motivo": reason,
                                           "resultado": str(res), "raw": res.raw})
        if not res.ok:
            # Nao limpamos a posicao interna: o broker ainda a tem.
            logger.critical("falha ao zerar (%s): %s", reason, res.comment)
            return StepResult(reason=f"FALHA ao zerar: {res.comment}", bar_time=bar_time,
                              features=feats)

        fill = res.filled_price if res.filled_price else exit_hint
        pnl = self.risk.register_exit(pos, fill, self._bar_index)
        self.journal.position_closed({
            "motivo": reason, "side": pos.side, "contracts": pos.contracts,
            "entry": pos.entry_price, "exit": fill,
            "points": round(pos.unrealized_points(fill), 1),
            "pnl_brl": round(pnl, 2),
            "pnl_dia_brl": round(self.risk.state.realized_pnl_brl, 2),
            "decisor_entrada": pos.decisor, "motivo_entrada": pos.entry_reason,
        })
        logger.info("SAIDA (%s) %s %dx | %+.0f pts | R$ %+.2f | dia R$ %+.2f",
                    reason, pos.side, pos.contracts, pos.unrealized_points(fill),
                    pnl, self.risk.state.realized_pnl_brl)
        self.position = None
        return StepResult(True, f"saiu por {reason}", bar_time=bar_time, features=feats,
                          extra={"pnl_brl": round(pnl, 2), "exit_reason": reason})

    # --- avaliar entrada ----------------------------------------------------

    def _evaluate_entry(self, snap: MarketSnapshot, feats: FeatureSnapshot,
                        now: datetime, bar_time: pd.Timestamp) -> StepResult:
        if self.risk.must_force_flat(now):
            return StepResult(reason="apos horario de zeragem: nao abre posicao",
                              bar_time=bar_time, features=feats)

        if self.risk.state.halted:
            return StepResult(reason=f"robo parado: {self.risk.state.halt_reason}",
                              bar_time=bar_time, features=feats)

        if len(snap.m1) < self.decisor.warmup_bars():
            return StepResult(reason=f"warmup: {len(snap.m1)} de "
                                     f"{self.decisor.warmup_bars()} candles",
                              bar_time=bar_time, features=feats)

        signal = self.decisor.safe_decide(feats)
        self.journal.decision(signal, {
            "bar": str(bar_time), "price": feats.price, "atr": feats.atr,
            "regime": feats.regime, "m5_bar": feats.raw.get("m5_time"),
            "resumo": feats.summary(),
        })

        if not signal.is_entry:
            return StepResult(reason=f"HOLD: {signal.reason}", signal=signal,
                              bar_time=bar_time, features=feats)

        plan_decision = self.risk.plan_entry(signal.action, feats.price, feats.atr,
                                             now, self._bar_index)
        if not plan_decision.approved or plan_decision.plan is None:
            self.journal.risk_block(plan_decision.reason, {
                "signal": signal.action, "confidence": signal.confidence,
                "bar": str(bar_time),
            })
            return StepResult(reason=f"risco bloqueou: {plan_decision.reason}",
                              signal=signal, bar_time=bar_time, features=feats)

        plan = plan_decision.plan
        self.journal.order_attempt({
            "side": plan.side, "contracts": plan.contracts, "price_hint": feats.price,
            "stop": plan.stop_price, "target": plan.target_price,
            "risk_brl": round(plan.risk_brl, 2), "rr": round(plan.rr_ratio, 2),
            "decisor": signal.decisor, "motivo": signal.reason,
        })
        res = self.broker.market_order(plan.side, plan.contracts, feats.price,
                                       comment=f"{signal.decisor}:{signal.action}")
        self.journal.order_result(res.ok, {"action": "open", "resultado": str(res),
                                           "raw": res.raw})
        if not res.ok:
            return StepResult(reason=f"ordem falhou: {res.comment}", signal=signal,
                              bar_time=bar_time, features=feats)

        fill = res.filled_price if res.filled_price else feats.price
        # Stop e alvo recalculados a partir do preco REAL de execucao, nao do
        # preco visto: senao o risco em reais nao e o que voce dimensionou.
        if plan.side == "BUY":
            stop_px, tgt_px = fill - plan.stop_points, fill + plan.target_points
        else:
            stop_px, tgt_px = fill + plan.stop_points, fill - plan.target_points

        self.position = Position(
            side=plan.side, entry_price=fill, contracts=res.contracts or plan.contracts,
            stop_price=stop_px, target_price=tgt_px, opened_at=now,
            decisor=signal.decisor, entry_reason=signal.reason,
            meta={"confidence": signal.confidence, **signal.meta},
        )
        self.journal.write("position_opened", {
            "side": plan.side, "contracts": self.position.contracts,
            "entry": fill, "price_hint": feats.price,
            "slippage_points": round(abs(fill - feats.price), 1),
            "stop": stop_px, "target": tgt_px,
            "decisor": signal.decisor, "confidence": signal.confidence,
            "motivo": signal.reason, "regime": feats.regime,
        })
        logger.info("ENTRADA %s %dx @ %.0f | stop %.0f | alvo %.0f | %s",
                    plan.side, self.position.contracts, fill, stop_px, tgt_px, signal.reason)
        return StepResult(True, f"abriu {plan.side}", signal=signal, bar_time=bar_time,
                          features=feats, extra={"entry": fill})

    # --- reconciliacao ------------------------------------------------------

    def _reconcile(self) -> str | None:
        """Compara a crenca interna com o broker. Devolve o problema ou None."""
        try:
            broker_pos = self.broker.current_position()
        except Exception as exc:  # noqa: BLE001
            self.journal.error("broker.current_position", exc)
            return f"nao consegui ler a posicao no broker: {exc}"

        if self.position is None and broker_pos is None:
            return None
        if self.position is None and broker_pos is not None:
            return (f"broker tem posicao {broker_pos.side} {broker_pos.contracts}x que o robo "
                    f"nao conhece (ordem manual? stop da corretora?)")
        if self.position is not None and broker_pos is None:
            return (f"robo acredita estar {self.position.side} {self.position.contracts}x, "
                    f"mas o broker nao tem posicao (foi zerada por fora?)")
        assert self.position is not None and broker_pos is not None
        if self.position.side != broker_pos.side:
            return (f"lado divergente: robo {self.position.side}, broker {broker_pos.side}")
        if self.position.contracts != broker_pos.contracts:
            return (f"quantidade divergente: robo {self.position.contracts}x, "
                    f"broker {broker_pos.contracts}x")
        return None

    # --- loop ---------------------------------------------------------------

    def run(self, poll_seconds: float = 5.0, max_steps: int | None = None) -> None:
        """Loop ao vivo. Interrompa com Ctrl+C - ele zera a posicao antes de sair."""
        logger.warning(
            "=== robo no ar | modo %s | decisor %s | ordens reais: %s ===",
            self.cfg.mode, self.decisor.name,
            "SIM" if self.broker.sends_real_orders else "NAO (paper)",
        )
        steps = 0
        try:
            while max_steps is None or steps < max_steps:
                now = datetime.now()
                result = self.step(now)
                if result.acted or result.signal is not None:
                    logger.debug("step: %s", result.reason)
                steps += 1

                if self.risk.state.halted and self.position is None:
                    logger.warning("dia encerrado: %s", self.risk.state.summary())
                    break
                time_mod.sleep(poll_seconds)
        except KeyboardInterrupt:
            logger.warning("interrompido pelo usuario")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Encerramento seguro: zera posicao aberta antes de sair."""
        if self.position is not None:
            logger.warning("encerrando com posicao aberta - zerando a mercado")
            try:
                snap = self.feed.snapshot()
                res = self.broker.close_all(snap.price, comment="shutdown")
                self.journal.order_result(res.ok, {"action": "close_on_shutdown",
                                                   "resultado": str(res)})
                if res.ok:
                    pnl = self.risk.register_exit(
                        self.position, res.filled_price or snap.price, self._bar_index)
                    logger.warning("zerado no shutdown | PnL R$ %+.2f", pnl)
                    self.position = None
            except Exception as exc:  # noqa: BLE001
                self.journal.error("shutdown.close_all", exc)
                logger.critical("NAO consegui zerar no shutdown - verifique o terminal!")
        self.journal.write("engine_stop", {"resumo_do_dia": self.risk.state.summary()})
        logger.warning("=== %s ===", self.risk.state.summary())
