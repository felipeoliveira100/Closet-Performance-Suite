"""Motor de backtest.

Reutiliza RiskManager e CostModel - os MESMOS que rodam ao vivo. Isso e
deliberado: um backtest com contabilidade propria divergiria do robo real
e voce descobriria a diferenca com dinheiro na mesa.

Premissas conservadoras, todas explicitas:
  - entrada e saida pagam slippage (default 1 tick = 5 pts em WIN);
  - quando a barra toca stop E alvo, assume STOP (sem tick a tick nao da
    para saber a ordem, e assumir o alvo e como se inflam backtests);
  - decisao usa somente a barra fechada;
  - custos cobrados nas duas pontas pelo CostModel;
  - zeragem forcada no fim do pregao, sem posicao overnight.

O que este backtest NAO captura, e voce precisa saber: fila em ordem
limitada, falta de liquidez no seu tamanho, leilao, gap intradia, e o
comportamento do book. Em WIN com 1 contrato isso e pequeno; com tamanho
maior, nao e.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .costs import CostModel
from .decision.base import Decisor
from .features import snapshot_at
from .risk import Position, RiskManager

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    side: str
    contracts: int
    entry_price: float
    exit_price: float
    points: float
    gross_brl: float
    cost_brl: float
    net_brl: float
    exit_reason: str
    decisor: str
    entry_reason: str
    confidence: float
    regime: str


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series | None = None
    daily_pnl: pd.Series | None = None
    bars: int = 0
    decisor: str = ""
    halted_days: int = 0
    blocked_by_risk: int = 0
    signals_generated: int = 0

    # --- metricas -----------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def net(self) -> float:
        return sum(t.net_brl for t in self.trades)

    @property
    def gross(self) -> float:
        return sum(t.gross_brl for t in self.trades)

    @property
    def costs(self) -> float:
        return sum(t.cost_brl for t in self.trades)

    @property
    def wins(self) -> list[Trade]:
        return [t for t in self.trades if t.net_brl > 0]

    @property
    def losses(self) -> list[Trade]:
        return [t for t in self.trades if t.net_brl <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.wins) / self.n if self.n else 0.0

    @property
    def expectancy(self) -> float:
        """Esperanca por trade, em reais. O numero que decide se vale operar."""
        return self.net / self.n if self.n else 0.0

    @property
    def profit_factor(self) -> float:
        g = sum(t.net_brl for t in self.wins)
        l = abs(sum(t.net_brl for t in self.losses))
        return g / l if l > 0 else (float("inf") if g > 0 else 0.0)

    @property
    def max_drawdown(self) -> float:
        if self.equity is None or self.equity.empty:
            return 0.0
        peak = self.equity.cummax()
        return float((self.equity - peak).min())

    @property
    def sharpe_daily(self) -> float:
        """Sharpe anualizado a partir do PnL diario (252 dias)."""
        if self.daily_pnl is None or len(self.daily_pnl) < 2:
            return 0.0
        sd = float(self.daily_pnl.std(ddof=1))
        if sd == 0:
            return 0.0
        return float(self.daily_pnl.mean() / sd * np.sqrt(252))

    @property
    def cost_share_of_gross(self) -> float:
        return self.costs / abs(self.gross) if self.gross else 0.0

    def summary(self) -> str:
        if not self.n:
            return (f"[{self.decisor}] NENHUM TRADE em {self.bars} candles. "
                    f"Sinais gerados: {self.signals_generated}, "
                    f"bloqueados pelo risco: {self.blocked_by_risk}.")
        by_reason: dict[str, int] = {}
        for t in self.trades:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
        lines = [
            f"[{self.decisor}] {self.n} trades em {self.bars} candles de 1min",
            f"  PnL LIQUIDO:      R$ {self.net:+10.2f}",
            f"  PnL bruto:        R$ {self.gross:+10.2f}",
            f"  custos:           R$ {self.costs:10.2f}  ({100*self.cost_share_of_gross:.1f}% do bruto)",
            f"  esperanca/trade:  R$ {self.expectancy:+10.2f}",
            f"  acerto:           {100*self.win_rate:9.1f}%  ({len(self.wins)}W / {len(self.losses)}L)",
            f"  profit factor:    {self.profit_factor:10.2f}",
            f"  max drawdown:     R$ {self.max_drawdown:10.2f}",
            f"  sharpe (anual.):  {self.sharpe_daily:10.2f}",
            f"  saidas:           {by_reason}",
            f"  sinais/bloqueios: {self.signals_generated} sinais, {self.blocked_by_risk} bloqueados pelo risco",
            f"  dias parados pelo kill switch: {self.halted_days}",
        ]
        return "\n".join(lines)


def run_backtest(cfg: Config, frame: pd.DataFrame, decisor: Decisor,
                 slippage_ticks: float = 1.0, warmup: int | None = None,
                 verbose: bool = False) -> BacktestResult:
    """Roda o decisor sobre o frame de features, barra a barra."""
    costs = CostModel(cfg.costs, cfg.symbol)
    risk = RiskManager(cfg)
    tick = cfg.symbol.tick_size
    pv = cfg.symbol.point_value
    slip = slippage_ticks * tick

    warmup = warmup if warmup is not None else decisor.warmup_bars()
    result = BacktestResult(decisor=decisor.name, bars=max(0, len(frame) - warmup))

    position: Position | None = None
    equity_points: list[tuple[pd.Timestamp, float]] = []
    running = 0.0
    halted_days: set[Any] = set()

    for i in range(warmup, len(frame)):
        row = frame.iloc[i]
        t: pd.Timestamp = frame.index[i]
        now = t.to_pydatetime()
        low, high, close = float(row["low"]), float(row["high"]), float(row["close"])

        risk.roll_day_if_needed(now)

        # --- posicao aberta: checa saida COM OS EXTREMOS DA BARRA -----------
        if position is not None:
            unreal = position.unrealized_brl(close, pv)
            risk.check_kill_switch(open_position_pnl=unreal)
            decision = risk.check_exit(position, low, high, close, now)
            if decision is not None:
                reason, exit_ref = decision
                # saida paga slippage contra o robo
                if reason in ("session_end", "kill_switch"):
                    exit_px = exit_ref - slip if position.is_long else exit_ref + slip
                elif reason == "stop":
                    # stop estourado: assume execucao 1 tick PIOR que o nivel
                    exit_px = position.stop_price - slip if position.is_long else position.stop_price + slip
                else:  # target: ordem limitada, sem slippage a favor
                    exit_px = position.target_price
                exit_px = round(exit_px / tick) * tick

                gross = costs.gross_pnl_brl(position.entry_price, exit_px, position.contracts, position.side)
                cost = costs.round_trip(position.contracts).total
                net = gross - cost
                risk.state.register_close(net, i)
                risk.check_kill_switch()
                running += net
                equity_points.append((t, running))
                result.trades.append(Trade(
                    entry_time=position.opened_at, exit_time=now, side=position.side,
                    contracts=position.contracts, entry_price=position.entry_price,
                    exit_price=exit_px, points=position.unrealized_points(exit_px),
                    gross_brl=gross, cost_brl=cost, net_brl=net, exit_reason=reason,
                    decisor=position.decisor, entry_reason=position.entry_reason,
                    confidence=float(position.meta.get("confidence", 0.0)),
                    regime=str(position.meta.get("regime", "")),
                ))
                if verbose:
                    logger.info("%s %s %dx %.0f->%.0f %+.0fpts R$%+.2f (%s)",
                                t, position.side, position.contracts,
                                position.entry_price, exit_px,
                                position.unrealized_points(exit_px), net, reason)
                position = None
            else:
                continue  # mantendo posicao: nao avalia entrada nova

        if risk.state.halted:
            halted_days.add(risk.state.day)
            continue

        if risk.must_force_flat(now):
            continue

        # --- avaliar entrada ------------------------------------------------
        snap = snapshot_at(frame, i)
        signal = decisor.safe_decide(snap)
        if not signal.is_entry:
            continue
        result.signals_generated += 1

        plan_dec = risk.plan_entry(signal.action, close, snap.atr, now, i)
        if not plan_dec.approved or plan_dec.plan is None:
            result.blocked_by_risk += 1
            continue

        plan = plan_dec.plan
        # entrada a mercado paga slippage
        entry_px = close + slip if plan.side == "BUY" else close - slip
        entry_px = round(entry_px / tick) * tick
        if plan.side == "BUY":
            stop_px, tgt_px = entry_px - plan.stop_points, entry_px + plan.target_points
        else:
            stop_px, tgt_px = entry_px + plan.stop_points, entry_px - plan.target_points

        position = Position(
            side=plan.side, entry_price=entry_px, contracts=plan.contracts,
            stop_price=stop_px, target_price=tgt_px, opened_at=now,
            decisor=signal.decisor, entry_reason=signal.reason,
            meta={"confidence": signal.confidence, "regime": snap.regime},
        )

    # posicao aberta ao fim dos dados: fecha no ultimo preco
    if position is not None:
        last_t = frame.index[-1]
        last_px = float(frame["close"].iloc[-1])
        gross = costs.gross_pnl_brl(position.entry_price, last_px, position.contracts, position.side)
        cost = costs.round_trip(position.contracts).total
        net = gross - cost
        running += net
        equity_points.append((last_t, running))
        result.trades.append(Trade(
            entry_time=position.opened_at, exit_time=last_t.to_pydatetime(),
            side=position.side, contracts=position.contracts,
            entry_price=position.entry_price, exit_price=last_px,
            points=position.unrealized_points(last_px), gross_brl=gross,
            cost_brl=cost, net_brl=net, exit_reason="end_of_data",
            decisor=position.decisor, entry_reason=position.entry_reason,
            confidence=float(position.meta.get("confidence", 0.0)),
            regime=str(position.meta.get("regime", "")),
        ))

    if equity_points:
        idx, vals = zip(*equity_points)
        result.equity = pd.Series(vals, index=pd.DatetimeIndex(idx))
        result.daily_pnl = (
            pd.Series([t.net_brl for t in result.trades],
                      index=pd.DatetimeIndex([t.exit_time for t in result.trades]))
            .resample("1D").sum()
        )
        result.daily_pnl = result.daily_pnl[result.daily_pnl != 0]
    result.halted_days = len(halted_days)
    return result
