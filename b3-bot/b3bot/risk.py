"""Gestao de risco. O modulo que impede o robo de te arruinar.

Divisao de responsabilidade: o decisor diz a DIRECAO, este modulo decide
se opera, com QUANTOS contratos, com qual stop e quando sair. O decisor
nunca manda ordem direto - tudo passa por aqui.

Protecoes implementadas, em ordem de importancia:
  1. KILL SWITCH de perda diaria: atingido o limite, o dia acaba. Nao ha
     "so mais uma para recuperar" - e assim que uma perda controlada vira
     descontrolada.
  2. Stop OBRIGATORIO em toda entrada. Sem stop calculavel, nao opera.
  3. Zeragem forcada antes do fim do pregao. Day trade nao carrega
     posicao overnight: o gap de abertura passa por cima do stop.
  4. Alvo precisa pagar o custo com folga - rejeita setup cujo alvo nao
     cobre varias vezes o custo de ida e volta.
  5. Teto de contratos, limite de operacoes por dia, pausa por perdas
     consecutivas e cooldown depois de sair.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Literal

from .config import Config
from .costs import CostModel

logger = logging.getLogger(__name__)

Side = Literal["BUY", "SELL"]
ExitReason = Literal["stop", "target", "session_end", "kill_switch", "signal_flip", "manual"]


@dataclass
class Position:
    side: Side
    entry_price: float
    contracts: int
    stop_price: float
    target_price: float
    opened_at: datetime
    decisor: str = ""
    entry_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.side == "BUY"

    def unrealized_points(self, price: float) -> float:
        return (price - self.entry_price) * (1.0 if self.is_long else -1.0)

    def unrealized_brl(self, price: float, point_value: float) -> float:
        return self.unrealized_points(price) * self.contracts * point_value

    def hit_stop(self, low: float, high: float) -> bool:
        """Stop tocado dentro da barra (usa extremos, nao o fechamento).

        Checar apenas o fechamento subestima perdas no backtest: o preco
        passa pelo stop no meio da barra e voce foi executado la.
        """
        return low <= self.stop_price if self.is_long else high >= self.stop_price

    def hit_target(self, low: float, high: float) -> bool:
        return high >= self.target_price if self.is_long else low <= self.target_price


@dataclass
class DayState:
    """Estado do dia. Zera na virada."""

    day: date
    realized_pnl_brl: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ""
    last_exit_bar: int = -10_000

    def register_close(self, pnl_brl: float, bar_index: int) -> None:
        self.realized_pnl_brl += pnl_brl
        self.trades += 1
        self.last_exit_bar = bar_index
        if pnl_brl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1

    def summary(self) -> str:
        wr = (100.0 * self.wins / self.trades) if self.trades else 0.0
        status = f"PARADO ({self.halt_reason})" if self.halted else "ativo"
        return (f"{self.day} | PnL R$ {self.realized_pnl_brl:+.2f} | "
                f"{self.trades} trades ({self.wins}W/{self.losses}L, {wr:.0f}%) | {status}")


@dataclass(frozen=True)
class EntryPlan:
    side: Side
    contracts: int
    stop_price: float
    target_price: float
    stop_points: float
    target_points: float
    risk_brl: float
    reward_brl: float
    cost_brl: float

    @property
    def rr_ratio(self) -> float:
        return self.reward_brl / self.risk_brl if self.risk_brl > 0 else 0.0


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str
    plan: EntryPlan | None = None

    def __str__(self) -> str:
        if self.approved and self.plan:
            p = self.plan
            return (f"APROVADO {p.side} {p.contracts}x | stop {p.stop_points:.0f}pts "
                    f"(R$ {p.risk_brl:.2f}) | alvo {p.target_points:.0f}pts "
                    f"(R$ {p.reward_brl:.2f}) | R:R {p.rr_ratio:.2f}")
        return f"REJEITADO: {self.reason}"


class RiskManager:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.r = cfg.risk
        self.costs = CostModel(cfg.costs, cfg.symbol)
        self.point_value = cfg.symbol.point_value
        self.tick = cfg.symbol.tick_size
        self.state = DayState(day=date.today())

    # --- ciclo do dia -------------------------------------------------------

    def roll_day_if_needed(self, now: datetime) -> bool:
        if now.date() != self.state.day:
            logger.info("virada de dia: %s", self.state.summary())
            self.state = DayState(day=now.date())
            return True
        return False

    def halt(self, reason: str) -> None:
        if not self.state.halted:
            self.state.halted = True
            self.state.halt_reason = reason
            logger.critical("ROBO PARADO PARA O DIA: %s", reason)

    def check_kill_switch(self, open_position_pnl: float = 0.0) -> str | None:
        """Checa o kill switch. Inclui o PnL em aberto - de proposito.

        Olhar so o realizado deixa a perda crescer numa posicao aberta sem
        acionar nada.
        """
        total = self.state.realized_pnl_brl + open_position_pnl
        if total <= -abs(self.r.max_daily_loss_brl):
            reason = (f"perda diaria R$ {total:.2f} atingiu o limite de "
                      f"R$ {-abs(self.r.max_daily_loss_brl):.2f}")
            self.halt(reason)
            return reason
        if self.state.consecutive_losses >= self.r.max_consecutive_losses:
            reason = f"{self.state.consecutive_losses} perdas consecutivas"
            self.halt(reason)
            return reason
        return None

    # --- janela de pregao ---------------------------------------------------

    def in_entry_window(self, now: datetime) -> bool:
        t = now.time()
        return self.cfg.session.entry_start <= t < self.cfg.session.entry_end

    def must_force_flat(self, now: datetime) -> bool:
        return now.time() >= self.cfg.session.force_flat_at

    # --- entrada ------------------------------------------------------------

    def plan_entry(self, side: Side, price: float, atr: float, now: datetime,
                   bar_index: int = 0) -> RiskDecision:
        """Valida e dimensiona uma entrada. Toda rejeicao tem motivo legivel."""
        if self.state.halted:
            return RiskDecision(False, f"robo parado: {self.state.halt_reason}")

        if (kill := self.check_kill_switch()) is not None:
            return RiskDecision(False, kill)

        if self.state.trades >= self.r.max_daily_trades:
            return RiskDecision(False, f"limite de {self.r.max_daily_trades} operacoes no dia atingido")

        if not self.in_entry_window(now):
            return RiskDecision(
                False,
                f"fora da janela de entrada "
                f"({self.cfg.session.entry_start:%H:%M}-{self.cfg.session.entry_end:%H:%M}), agora {now:%H:%M}",
            )

        bars_since_exit = bar_index - self.state.last_exit_bar
        if bars_since_exit < self.r.cooldown_bars_after_exit:
            return RiskDecision(
                False,
                f"cooldown: {bars_since_exit} de {self.r.cooldown_bars_after_exit} candles desde a ultima saida",
            )

        if not math.isfinite(atr) or atr <= 0:
            return RiskDecision(False, "ATR indisponivel: impossivel calcular stop")

        # --- stop -----------------------------------------------------------
        raw_stop = self.r.stop_atr_mult * atr
        if raw_stop > self.r.max_stop_points:
            return RiskDecision(
                False,
                f"stop exigido {raw_stop:.0f} pts > maximo {self.r.max_stop_points} "
                f"(ATR {atr:.0f}): volatilidade alta demais para o seu risco",
            )
        stop_points = self._round_tick(max(raw_stop, self.r.min_stop_points))

        # O alvo acompanha o stop EFETIVO, nao o ATR cru.
        # Por que: quando o ATR e baixo, min_stop_points eleva o stop mas o
        # alvo colado no ATR nao sobe, e o payoff desaba (com ATR 40,
        # stop 1.5x -> piso de 100 pts e alvo 2.5x -> 100 pts: R:R 1.0, que
        # apos custo fica NEGATIVO). Preservar a razao pretendida
        # (target_atr_mult / stop_atr_mult) mantem o payoff em qualquer
        # nivel de volatilidade.
        rr_target = self.r.target_atr_mult / self.r.stop_atr_mult
        target_points = self._round_tick(stop_points * rr_target)
        if target_points <= 0:
            return RiskDecision(False, "alvo calculado nao positivo")

        # --- sizing ---------------------------------------------------------
        risk_per_contract = stop_points * self.point_value
        if risk_per_contract <= 0:
            return RiskDecision(False, "risco por contrato nao positivo")

        contracts = int(self.r.risk_per_trade_brl // risk_per_contract)
        contracts = min(contracts, self.r.max_contracts)
        if contracts < self.cfg.symbol.min_volume:
            return RiskDecision(
                False,
                f"risco por contrato R$ {risk_per_contract:.2f} excede o risco por "
                f"operacao R$ {self.r.risk_per_trade_brl:.2f} (stop {stop_points:.0f} pts). "
                f"Aumente risk_per_trade_brl ou reduza stop_atr_mult.",
            )

        # --- o alvo paga o custo? ------------------------------------------
        cost = self.costs.round_trip(contracts).total
        reward = target_points * contracts * self.point_value
        risk = stop_points * contracts * self.point_value
        breakeven_pts = self.costs.breakeven_points(contracts)

        if target_points < breakeven_pts * 3:
            return RiskDecision(
                False,
                f"alvo {target_points:.0f} pts nao cobre 3x o custo de ida e volta "
                f"({breakeven_pts:.1f} pts): o custo come o trade",
            )

        # Trava final: o payoff LIQUIDO precisa valer a pena. Esta checagem
        # roda depois do clamp do stop, do arredondamento no tick e dos
        # custos - e o unico ponto onde o R:R real aparece.
        net_rr = (reward - cost) / risk if risk > 0 else 0.0
        if net_rr < self.r.min_rr_after_costs:
            return RiskDecision(
                False,
                f"R:R liquido {net_rr:.2f} < minimo {self.r.min_rr_after_costs:.2f} "
                f"(stop {stop_points:.0f} pts / alvo {target_points:.0f} pts, custo R$ {cost:.2f})",
            )

        if side == "BUY":
            stop_price = self._round_tick_price(price - stop_points)
            target_price = self._round_tick_price(price + target_points)
        else:
            stop_price = self._round_tick_price(price + stop_points)
            target_price = self._round_tick_price(price - target_points)

        plan = EntryPlan(
            side=side, contracts=contracts,
            stop_price=stop_price, target_price=target_price,
            stop_points=stop_points, target_points=target_points,
            risk_brl=risk, reward_brl=reward - cost, cost_brl=cost,
        )
        return RiskDecision(True, "aprovado", plan)

    # --- saida --------------------------------------------------------------

    def check_exit(self, pos: Position, low: float, high: float, price: float,
                   now: datetime) -> tuple[ExitReason, float] | None:
        """Precisa sair? Devolve (motivo, preco de saida) ou None.

        Ordem de precedencia: stop antes de alvo. Quando a barra toca os
        dois, assumir o alvo e a forma classica de inflar backtest - sem o
        dado tick a tick nao da para saber qual veio primeiro, e assumir o
        pior e o unico jeito honesto.
        """
        if self.must_force_flat(now):
            return ("session_end", price)
        if pos.hit_stop(low, high):
            return ("stop", pos.stop_price)
        if pos.hit_target(low, high):
            return ("target", pos.target_price)
        if self.state.halted:
            return ("kill_switch", price)
        return None

    def register_exit(self, pos: Position, exit_price: float, bar_index: int) -> float:
        """Fecha a posicao na contabilidade e devolve o PnL liquido."""
        pnl = self.costs.net_pnl_brl(pos.entry_price, exit_price, pos.contracts, pos.side)
        self.state.register_close(pnl, bar_index)
        self.check_kill_switch()
        return pnl

    # --- utilitarios --------------------------------------------------------

    def _round_tick(self, points: float) -> float:
        """Arredonda distancia para multiplo do tick (WIN: 5 pontos)."""
        return max(self.tick, round(points / self.tick) * self.tick)

    def _round_tick_price(self, price: float) -> float:
        return round(price / self.tick) * self.tick
