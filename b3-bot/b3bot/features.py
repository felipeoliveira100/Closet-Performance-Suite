"""Analise grafica: indicadores, estrutura de mercado e regime.

REGRA ABSOLUTA deste modulo: nenhuma funcao pode olhar para o futuro.
Toda media, todo extremo e todo rotulo no instante t usa exclusivamente
dados de t e anteriores. Nao existe shift(-n) aqui. O teste
tests/test_no_lookahead.py prova isso empiricamente: ele calcula as
features sobre a serie truncada e sobre a serie completa, e exige que os
valores do passado sejam identicos.

Sincronizacao 1min/5min - o ponto mais sutil do arquivo:
  uma barra de 5min indexada em 11:10 so esta COMPLETA as 11:15.
  Logo, ao decidir no candle de 1min que fechou as 11:13, o contexto de
  5min disponivel e o da barra de 11:05 (que fechou as 11:10), NAO o da
  barra de 11:10. `join_m5_context` faz essa conta pelo horario de
  fechamento, e nao pelo de abertura. Errar isso vaza futuro e produz
  backtest glorioso.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# indicadores basicos
# --------------------------------------------------------------------------


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR de Wilder (media exponencial do True Range com alpha=1/period)."""
    return true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # perda media zero = so alta = RSI 100
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain != 0.0, out.fillna(50.0))


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP intradiario, resetando a cada dia de pregao.

    Nao resetar por dia transforma o VWAP em media de longo prazo inutil
    para day trade - erro comum.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["tick_volume"].astype(float).replace(0.0, 1.0)
    day = df.index.normalize()
    pv_cum = (typical * vol).groupby(day).cumsum()
    v_cum = vol.groupby(day).cumsum()
    return pv_cum / v_cum


def bollinger(series: pd.Series, period: int = 20, mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period, min_periods=period).mean()
    sd = series.rolling(period, min_periods=period).std(ddof=0)
    return mid - mult * sd, mid, mid + mult * sd


# --------------------------------------------------------------------------
# estrutura de mercado e padroes de candle
# --------------------------------------------------------------------------


def rolling_range(df: pd.DataFrame, lookback: int = 20) -> tuple[pd.Series, pd.Series]:
    """Maxima e minima das `lookback` barras ANTERIORES (exclui a atual).

    shift(1) e o que impede o rompimento de se comparar consigo mesmo -
    sem ele, toda barra "rompe" seu proprio topo e a estrategia parece
    acertar sempre.
    """
    hi = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    lo = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    return hi, lo


def candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    bull = (df["close"] > df["open"]).astype(bool)

    prev_body_hi = df[["open", "close"]].max(axis=1).shift(1)
    prev_body_lo = df[["open", "close"]].min(axis=1).shift(1)
    prev_bull = bull.shift(1).fillna(False).astype(bool)

    out = pd.DataFrame(index=df.index)
    out["body_ratio"] = (body / rng).fillna(0.0)
    out["upper_wick_ratio"] = (upper / rng).fillna(0.0)
    out["lower_wick_ratio"] = (lower / rng).fillna(0.0)
    out["is_bull"] = bull.astype(int)
    out["doji"] = (out["body_ratio"] < 0.12).astype(int)
    out["hammer"] = ((out["lower_wick_ratio"] > 0.55) & (out["body_ratio"] < 0.35)).astype(int)
    out["shooting_star"] = ((out["upper_wick_ratio"] > 0.55) & (out["body_ratio"] < 0.35)).astype(int)
    out["bull_engulf"] = (
        bull & (~prev_bull) & (df["close"] >= prev_body_hi) & (df["open"] <= prev_body_lo)
    ).astype(int)
    out["bear_engulf"] = (
        (~bull) & prev_bull & (df["close"] <= prev_body_lo) & (df["open"] >= prev_body_hi)
    ).astype(int)
    return out


# --------------------------------------------------------------------------
# regime (calculado no 5min)
# --------------------------------------------------------------------------

REGIMES = ("trend_up", "trend_down", "range", "high_vol")


def classify_regime(m5: pd.DataFrame, slope_bars: int = 6, vol_window: int = 100,
                    vol_percentile: float = 0.85, slope_threshold: float = 0.35) -> pd.Series:
    """Classifica cada barra de 5min em um regime.

    Logica: inclinacao da EMA21 medida em ATRs (escala-invariante, funciona
    com WIN a 100k ou 150k pontos) + alinhamento das medias. Volatilidade
    no topo da distribuicao recente sobrepoe tudo e marca high_vol, porque
    la o stop normal nao aguenta e o payoff muda.
    """
    close = m5["close"]
    e9, e21, e50 = ema(close, 9), ema(close, 21), ema(close, 50)
    a = atr(m5, 14)

    slope_atr = (e21 - e21.shift(slope_bars)) / a.replace(0.0, np.nan)
    atr_rank = a.rolling(vol_window, min_periods=20).rank(pct=True)

    up = (slope_atr > slope_threshold) & (e9 > e21) & (close > e50)
    down = (slope_atr < -slope_threshold) & (e9 < e21) & (close < e50)

    regime = pd.Series("range", index=m5.index, dtype=object)
    regime[up] = "trend_up"
    regime[down] = "trend_down"
    regime[atr_rank > vol_percentile] = "high_vol"
    regime[a.isna() | slope_atr.isna()] = "unknown"
    return regime


# --------------------------------------------------------------------------
# montagem do frame de features
# --------------------------------------------------------------------------


def add_m1_features(m1: pd.DataFrame, ema_fast: int = 9, ema_slow: int = 21,
                    rsi_period: int = 14, breakout_lookback: int = 20) -> pd.DataFrame:
    df = m1.copy()
    df["ema_fast"] = ema(df["close"], ema_fast)
    df["ema_slow"] = ema(df["close"], ema_slow)
    df["atr"] = atr(df, 14)
    df["rsi"] = rsi(df["close"], rsi_period)
    df["vwap"] = session_vwap(df)
    bb_lo, bb_mid, bb_hi = bollinger(df["close"], 20, 2.0)
    df["bb_lo"], df["bb_mid"], df["bb_hi"] = bb_lo, bb_mid, bb_hi

    hi, lo = rolling_range(df, breakout_lookback)
    df["range_hi"], df["range_lo"] = hi, lo

    a = df["atr"].replace(0.0, np.nan)
    # features normalizadas por ATR: comparaveis entre niveis de preco e de vol
    df["dist_ema_fast_atr"] = (df["close"] - df["ema_fast"]) / a
    df["dist_ema_slow_atr"] = (df["close"] - df["ema_slow"]) / a
    df["dist_vwap_atr"] = (df["close"] - df["vwap"]) / a
    df["ema_spread_atr"] = (df["ema_fast"] - df["ema_slow"]) / a
    df["breakout_up_atr"] = (df["close"] - hi) / a
    df["breakout_dn_atr"] = (lo - df["close"]) / a
    df["bb_pos"] = (df["close"] - bb_lo) / (bb_hi - bb_lo).replace(0.0, np.nan)
    df["ret_1_atr"] = df["close"].diff(1) / a
    df["ret_5_atr"] = df["close"].diff(5) / a
    df["ret_15_atr"] = df["close"].diff(15) / a
    df["atr_pct"] = df["atr"] / df["close"]
    df["vol_ratio"] = df["tick_volume"] / df["tick_volume"].rolling(20, min_periods=20).mean()

    df = pd.concat([df, candle_patterns(m1)], axis=1)

    # hora do dia importa muito em day trade de indice
    minutes = df.index.hour * 60 + df.index.minute
    df["minutes_since_open"] = minutes - (9 * 60 + 0)
    df["session_phase"] = pd.cut(
        minutes,
        bins=[0, 10 * 60, 12 * 60, 15 * 60, 17 * 60, 24 * 60],
        labels=[0, 1, 2, 3, 4],
    ).astype("float")
    return df


def add_m5_features(m5: pd.DataFrame) -> pd.DataFrame:
    df = m5.copy()
    df["m5_ema9"] = ema(df["close"], 9)
    df["m5_ema21"] = ema(df["close"], 21)
    df["m5_ema50"] = ema(df["close"], 50)
    df["m5_atr"] = atr(df, 14)
    df["m5_rsi"] = rsi(df["close"], 14)
    a = df["m5_atr"].replace(0.0, np.nan)
    df["m5_slope_atr"] = (df["m5_ema21"] - df["m5_ema21"].shift(6)) / a
    df["m5_dist_ema21_atr"] = (df["close"] - df["m5_ema21"]) / a
    df["m5_ema_spread_atr"] = (df["m5_ema9"] - df["m5_ema21"]) / a
    df["m5_atr_rank"] = df["m5_atr"].rolling(100, min_periods=20).rank(pct=True)
    hi, lo = rolling_range(df, 12)
    df["m5_range_hi"], df["m5_range_lo"] = hi, lo
    df["m5_pos_in_range"] = (df["close"] - lo) / (hi - lo).replace(0.0, np.nan)
    df["regime"] = classify_regime(df)
    return df


M5_CONTEXT_COLS = [
    "m5_slope_atr", "m5_dist_ema21_atr", "m5_ema_spread_atr",
    "m5_atr_rank", "m5_rsi", "m5_pos_in_range", "regime",
]


def join_m5_context(m1_feat: pd.DataFrame, m5_feat: pd.DataFrame,
                    m5_minutes: int = 5) -> pd.DataFrame:
    """Anexa o contexto de 5min a cada barra de 1min, SEM vazar futuro.

    A barra de 5min aberta em T so esta disponivel em T + 5min. Fazemos o
    merge pelo horario de FECHAMENTO dela. Fazer o merge pelo horario de
    abertura (o erro natural, porque e o indice) entrega ao candle de 1min
    de 11:11 o fechamento do bloco 11:10-11:15 - informacao do futuro.
    """
    ctx = m5_feat[[c for c in M5_CONTEXT_COLS if c in m5_feat.columns]].copy()
    ctx["available_at"] = ctx.index + pd.Timedelta(minutes=m5_minutes)
    ctx = ctx.sort_values("available_at")

    left = m1_feat.sort_index()
    # o candle de 1min indexado em t fecha em t+1min: e esse o instante da decisao
    left_key = pd.DataFrame({"decision_at": left.index + pd.Timedelta(minutes=1)}, index=left.index)

    merged = pd.merge_asof(
        left_key.reset_index().rename(columns={left.index.name or "index": "m1_time"}),
        ctx.reset_index().rename(columns={ctx.index.name or "index": "m5_time"}),
        left_on="decision_at",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("m1_time")

    out = left.copy()
    for col in ctx.columns:
        if col == "available_at":
            continue
        out[col] = merged[col].to_numpy()
    out["m5_time"] = merged["m5_time"].to_numpy()
    # Nas primeiras barras ainda nao existe barra de 5min fechada. O regime
    # fica explicitamente "unknown" em vez de NaN: o decisor checa regime, e
    # NaN silencioso viraria comparacao sempre-falsa em vez de bloqueio claro.
    if "regime" in out.columns:
        out["regime"] = out["regime"].astype(object).where(out["regime"].notna(), "unknown").astype(str)
    return out


def build_feature_frame(m1: pd.DataFrame, m5: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Frame completo de features, alinhado e sem lookahead."""
    m1_feat = add_m1_features(m1, **kwargs)
    m5_feat = add_m5_features(m5)
    return join_m5_context(m1_feat, m5_feat)


# features numericas que vao para o modelo de ML
ML_FEATURE_COLS = [
    "dist_ema_fast_atr", "dist_ema_slow_atr", "dist_vwap_atr", "ema_spread_atr",
    "breakout_up_atr", "breakout_dn_atr", "bb_pos", "rsi",
    "ret_1_atr", "ret_5_atr", "ret_15_atr", "atr_pct", "vol_ratio",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "is_bull",
    "doji", "hammer", "shooting_star", "bull_engulf", "bear_engulf",
    "session_phase", "minutes_since_open",
    "m5_slope_atr", "m5_dist_ema21_atr", "m5_ema_spread_atr",
    "m5_atr_rank", "m5_rsi", "m5_pos_in_range",
]


@dataclass(frozen=True)
class FeatureSnapshot:
    """Estado do mercado no ultimo candle fechado - o que o decisor recebe."""

    time: pd.Timestamp
    price: float
    atr: float
    regime: str
    values: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: float = float("nan")) -> float:
        v = self.values.get(key, self.raw.get(key, default))
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def summary(self) -> str:
        return (
            f"{self.time:%d/%m %H:%M} | {self.price:.0f} | ATR {self.atr:.0f} | {self.regime} | "
            f"RSI {self.get('rsi'):.0f} | EMAspread {self.get('ema_spread_atr'):+.2f}atr | "
            f"VWAP {self.get('dist_vwap_atr'):+.2f}atr"
        )


def latest_snapshot(frame: pd.DataFrame) -> FeatureSnapshot:
    """Extrai o ultimo candle FECHADO do frame como snapshot de decisao."""
    if frame.empty:
        raise ValueError("frame de features vazio")
    row = frame.iloc[-1]
    values = {}
    for c in ML_FEATURE_COLS:
        if c in frame.columns:
            v = row[c]
            values[c] = float(v) if pd.notna(v) else float("nan")
    raw = {
        "open": float(row["open"]), "high": float(row["high"]),
        "low": float(row["low"]), "close": float(row["close"]),
        "ema_fast": float(row.get("ema_fast", float("nan"))),
        "ema_slow": float(row.get("ema_slow", float("nan"))),
        "vwap": float(row.get("vwap", float("nan"))),
        "range_hi": float(row.get("range_hi", float("nan"))),
        "range_lo": float(row.get("range_lo", float("nan"))),
        "m5_time": str(row.get("m5_time", "")),
    }
    return FeatureSnapshot(
        time=frame.index[-1],
        price=float(row["close"]),
        atr=float(row["atr"]) if pd.notna(row.get("atr")) else float("nan"),
        regime=str(row.get("regime", "unknown")),
        values=values,
        raw=raw,
    )

def snapshot_at(frame: pd.DataFrame, i: int) -> FeatureSnapshot:
    """Snapshot da linha i de um frame JA calculado.

    Existe por performance no backtest: recalcular o frame inteiro a cada
    barra seria O(n^2). E seguro porque as features sao causais - a linha i
    contem exatamente o que era conhecido no fechamento da barra i, e o
    teste de lookahead prova isso. Se algum dia uma feature passar a olhar
    o futuro, este atalho propaga o erro silenciosamente: e por isso que o
    teste de lookahead e obrigatorio.
    """
    if not (0 <= i < len(frame)):
        raise IndexError(f"indice {i} fora do frame de {len(frame)} barras")
    return latest_snapshot(frame.iloc[: i + 1])
