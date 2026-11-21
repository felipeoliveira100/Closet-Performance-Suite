"""Interface do feed de market data.

Contrato que toda implementacao precisa respeitar:

  *** SO DEVOLVE CANDLE FECHADO. ***

O ultimo candle que o MT5 entrega e o que ainda esta em formacao. Usar
ele para decidir e a forma numero 1 de produzir um backtest fantastico e
um robo que perde dinheiro: voce decide com o fechamento de um candle
que ainda nao fechou, ou seja, com informacao do futuro. Toda
implementacao de MarketFeed descarta essa barra.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

OHLC_COLUMNS = ["open", "high", "low", "close", "tick_volume"]


@dataclass(frozen=True)
class Bars:
    """Candles fechados de um timeframe."""

    timeframe: str  # "M1" | "M5"
    df: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in OHLC_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Bars[{self.timeframe}]: colunas ausentes {missing}")
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError(f"Bars[{self.timeframe}]: indice precisa ser DatetimeIndex")
        if not self.df.index.is_monotonic_increasing:
            raise ValueError(f"Bars[{self.timeframe}]: indice fora de ordem")

    @property
    def last_time(self) -> pd.Timestamp:
        return self.df.index[-1]

    @property
    def last_close(self) -> float:
        return float(self.df["close"].iloc[-1])

    def __len__(self) -> int:
        return len(self.df)


@dataclass(frozen=True)
class MarketSnapshot:
    """Estado do mercado num instante: 1min + 5min, ambos ja fechados."""

    symbol: str
    m1: Bars
    m5: Bars
    bid: float | None = None
    ask: float | None = None

    @property
    def price(self) -> float:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.m1.last_close

    @property
    def spread_points(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


class MarketFeed(ABC):
    """Fonte de candles. Implementacoes: MT5Feed (real) e SyntheticFeed (teste)."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def snapshot(self, m1_bars: int = 300, m5_bars: int = 200) -> MarketSnapshot:
        """Estado atual. SEMPRE sem o candle em formacao."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


def drop_forming_bar(df: pd.DataFrame, timeframe_minutes: int, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove a ultima barra se ela ainda nao fechou.

    Uma barra de N minutos abrindo em T fecha em T + N. Se agora < T + N,
    ela esta em formacao e nao pode ser usada para decidir.
    """
    if df.empty:
        return df
    now = now or pd.Timestamp.now(tz=df.index.tz)
    last_open = df.index[-1]
    closes_at = last_open + pd.Timedelta(minutes=timeframe_minutes)
    if now < closes_at:
        return df.iloc[:-1]
    return df


def detect_staleness(bars: Bars, timeframe_minutes: int, now: pd.Timestamp | None = None,
                     tolerance_bars: float = 3.0) -> float:
    """Idade da ultima barra, em multiplos do timeframe.

    > tolerance_bars significa feed travado, pregao fechado ou conexao
    caida. O engine usa isso para NAO decidir com dado velho.
    """
    now = now or pd.Timestamp.now(tz=bars.df.index.tz)
    age_minutes = (now - bars.last_time).total_seconds() / 60.0
    return age_minutes / timeframe_minutes
