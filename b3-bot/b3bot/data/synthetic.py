"""Feed sintetico para desenvolver e testar fora do Windows.

Gera serie de WIN com regimes alternados (tendencia e lateral), respeitando
tick de 5 pontos e consistencia de OHLC. Serve para:
  - rodar o engine inteiro em Linux, sem MT5;
  - testar que features nao tem lookahead;
  - sanity check do backtest.

NAO serve para validar estrategia. Dado sintetico nao tem microestrutura,
nao tem noticia e nao tem o comportamento de quem esta do outro lado.
Resultado bom aqui nao significa nada sobre o mercado real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Bars, MarketFeed, MarketSnapshot

TICK = 5.0


def _round_tick(x: np.ndarray | float) -> np.ndarray | float:
    return np.round(np.asarray(x) / TICK) * TICK


def generate_m1(
    n_bars: int = 2000,
    start: str = "2026-11-03 09:15",
    base_price: float = 140_000.0,
    seed: int = 7,
    regime_len: int = 120,
) -> pd.DataFrame:
    """Candles de 1 minuto sinteticos com regimes alternados."""
    rng = np.random.default_rng(seed)
    n_regimes = int(np.ceil(n_bars / regime_len))

    drifts, vols = [], []
    for i in range(n_regimes):
        kind = i % 3
        if kind == 0:      # tendencia de alta
            drifts.append(rng.uniform(1.5, 4.0)); vols.append(rng.uniform(25, 45))
        elif kind == 1:    # lateral
            drifts.append(rng.uniform(-0.4, 0.4)); vols.append(rng.uniform(15, 28))
        else:              # tendencia de baixa
            drifts.append(rng.uniform(-4.0, -1.5)); vols.append(rng.uniform(25, 45))

    drift = np.repeat(drifts, regime_len)[:n_bars]
    vol = np.repeat(vols, regime_len)[:n_bars]

    returns = drift + rng.normal(0.0, 1.0, n_bars) * vol
    close = base_price + np.cumsum(returns)
    open_ = np.concatenate([[base_price], close[:-1]])

    # extremos coerentes: high >= max(o,c), low <= min(o,c)
    wick_up = np.abs(rng.normal(0, 1, n_bars)) * vol * 0.6
    wick_dn = np.abs(rng.normal(0, 1, n_bars)) * vol * 0.6
    high = np.maximum(open_, close) + wick_up
    low = np.minimum(open_, close) - wick_dn

    open_, high, low, close = (_round_tick(a) for a in (open_, high, low, close))
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])

    volume = (rng.lognormal(6.2, 0.6, n_bars) * (1 + vol / 40)).astype(int)

    # indice so em minutos de pregao (09:15-17:55), dias uteis
    idx = _session_index(pd.Timestamp(start), n_bars)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "tick_volume": volume},
        index=idx,
    )


def _session_index(start: pd.Timestamp, n_bars: int) -> pd.DatetimeIndex:
    """Minutos de pregao consecutivos, pulando noites e fins de semana."""
    stamps: list[pd.Timestamp] = []
    cur = start
    while len(stamps) < n_bars:
        if cur.weekday() < 5 and (cur.hour, cur.minute) >= (9, 15) and (cur.hour, cur.minute) <= (17, 55):
            stamps.append(cur)
            cur += pd.Timedelta(minutes=1)
        else:
            nxt = (cur + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=9, minutes=15)
            cur = nxt if (cur.hour, cur.minute) > (17, 55) else cur.replace(hour=9, minute=15)
            while cur.weekday() >= 5:
                cur = (cur + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=9, minutes=15)
    return pd.DatetimeIndex(stamps, name="time")


def resample_to_m5(m1: pd.DataFrame, drop_partial: bool = True) -> pd.DataFrame:
    """Agrega 1min -> 5min. label/closed='left': a barra e indexada pela
    sua ABERTURA, igual ao MT5.

    `drop_partial` descarta a ultima barra de 5min quando ela ainda pode
    receber mais minutos - isto e, quando o proximo minuto apos o fim dos
    dados cairia dentro dela. Isso NAO e detalhe de teste: uma barra de
    5min com 2 minutos dentro tem close, high e low errados, e usar ela
    para decidir equivale a decidir com a barra em formacao.

    Atencao: nao basta checar len(m1) % 5 == 0. O indice de pregao salta
    noites e fins de semana, entao os blocos de 5min do calendario nao se
    alinham com multiplos de 5 barras de 1 minuto.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
    out = m1.resample("5min", label="left", closed="left").agg(agg).dropna(subset=["open"])
    if drop_partial and len(out) and len(m1):
        next_minute = m1.index[-1] + pd.Timedelta(minutes=1)
        if next_minute < out.index[-1] + pd.Timedelta(minutes=5):
            out = out.iloc[:-1]
    return out


class SyntheticFeed(MarketFeed):
    """Reproduz um pregao barra a barra, como se fosse ao vivo."""

    def __init__(self, symbol: str = "WINZ26", n_bars: int = 2000, seed: int = 7,
                 warmup: int = 120) -> None:
        self.symbol = symbol
        self._all_m1 = generate_m1(n_bars=n_bars, seed=seed)
        self._cursor = max(warmup, 60)

    def connect(self) -> None:  # nada a fazer
        return None

    def shutdown(self) -> None:
        return None

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._all_m1)

    def advance(self, bars: int = 1) -> bool:
        self._cursor += bars
        return not self.exhausted

    @property
    def now(self) -> pd.Timestamp:
        i = min(self._cursor, len(self._all_m1) - 1)
        return self._all_m1.index[i]

    def snapshot(self, m1_bars: int = 300, m5_bars: int = 200) -> MarketSnapshot:
        # tudo ATE o cursor (exclusivo) == somente barras fechadas
        m1 = self._all_m1.iloc[max(0, self._cursor - m1_bars):self._cursor]
        m5_full = resample_to_m5(self._all_m1.iloc[:self._cursor])
        # descarta a barra de 5min incompleta
        if len(m5_full) and len(self._all_m1.iloc[:self._cursor]) % 5 != 0:
            m5_full = m5_full.iloc[:-1]
        m5 = m5_full.iloc[-m5_bars:]
        last = float(m1["close"].iloc[-1]) if len(m1) else 140_000.0
        return MarketSnapshot(
            symbol=self.symbol,
            m1=Bars("M1", m1),
            m5=Bars("M5", m5),
            bid=last - TICK / 2,
            ask=last + TICK / 2,
        )
