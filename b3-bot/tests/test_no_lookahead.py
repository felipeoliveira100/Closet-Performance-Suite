"""Provas empiricas de que as features nao olham para o futuro.

Se qualquer teste aqui falhar, TODO backtest do projeto esta invalidado.
Nao "ajuste" o teste para passar: conserte a feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b3bot.features import ML_FEATURE_COLS, build_feature_frame  # noqa: E402
from b3bot.data.synthetic import generate_m1, resample_to_m5  # noqa: E402


@pytest.fixture(scope="module")
def series():
    m1 = generate_m1(n_bars=1200, seed=42)
    return m1, resample_to_m5(m1)


def test_truncar_serie_nao_muda_o_passado(series):
    """O teste decisivo.

    Calculamos as features com a serie COMPLETA e com a serie cortada no
    meio. Para as barras que existem nas duas, os valores precisam ser
    identicos. Se mudarem, a feature usou dados posteriores aquela barra -
    ou seja, no backtest ela sabia o futuro.
    """
    m1, m5 = series
    cut = 800

    full = build_feature_frame(m1, m5)
    m1_trunc = m1.iloc[:cut]
    m5_trunc = resample_to_m5(m1_trunc)
    trunc = build_feature_frame(m1_trunc, m5_trunc)

    common = trunc.index.intersection(full.index)
    assert len(common) > 500, "amostra comum pequena demais para o teste valer"

    divergentes = []
    for col in ML_FEATURE_COLS:
        if col not in full.columns:
            continue
        a = full.loc[common, col].astype(float).to_numpy()
        b = trunc.loc[common, col].astype(float).to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        if not np.allclose(a[~both_nan], b[~both_nan], rtol=1e-9, atol=1e-9, equal_nan=True):
            n = int((~np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)).sum())
            divergentes.append(f"{col} ({n} barras)")

    assert not divergentes, (
        "LOOKAHEAD DETECTADO nas features: " + ", ".join(divergentes)
    )


def test_regime_do_passado_nao_muda(series):
    """Mesma prova para o rotulo de regime, que vem do 5min."""
    m1, m5 = series
    cut = 900
    full = build_feature_frame(m1, m5)
    m1_t = m1.iloc[:cut]
    trunc = build_feature_frame(m1_t, resample_to_m5(m1_t))
    common = trunc.index.intersection(full.index)
    # ignora a cauda onde a barra de 5min do truncado ainda nao fechou
    common = common[:-10]
    # fillna antes de comparar: em pandas 3.0 o dtype str mantem NaN como
    # missing, e NaN != NaN seria contado como divergencia falsa.
    a = full.loc[common, "regime"].astype(object).fillna("__NA__").astype(str)
    b = trunc.loc[common, "regime"].astype(object).fillna("__NA__").astype(str)
    dif = (a != b).sum()
    assert dif == 0, f"regime de {dif} barras mudou ao truncar a serie - lookahead"


def test_contexto_m5_sempre_de_barra_ja_fechada(series):
    """A barra de 5min usada precisa ter FECHADO antes da decisao.

    Decisao no candle de 1min indexado em t acontece em t+1min.
    A barra de 5min indexada em T fecha em T+5min.
    Exigencia: T + 5min <= t + 1min.
    """
    m1, m5 = series
    f = build_feature_frame(m1, m5)
    valid = f.dropna(subset=["m5_time"]).copy()
    valid = valid[valid["m5_time"].astype(str) != "NaT"]
    m5_time = pd.to_datetime(valid["m5_time"])
    m5_closes_at = m5_time + pd.Timedelta(minutes=5)
    decision_at = valid.index + pd.Timedelta(minutes=1)

    violations = (m5_closes_at > decision_at).sum()
    assert violations == 0, (
        f"{violations} barras usaram contexto de 5min que ainda nao havia fechado"
    )


def test_rompimento_nao_usa_a_propria_barra(series):
    """range_hi precisa ser o topo das barras ANTERIORES.

    Sem o shift(1), toda barra rompe seu proprio topo e a estrategia de
    rompimento parece acertar quase sempre no backtest.
    """
    m1, m5 = series
    f = build_feature_frame(m1, m5).dropna(subset=["range_hi", "range_lo"])
    lookback = 20
    for i in range(len(f) - 50, len(f)):
        t = f.index[i]
        pos = m1.index.get_loc(t)
        if pos < lookback + 1:
            continue
        esperado_hi = m1["high"].iloc[pos - lookback:pos].max()
        esperado_lo = m1["low"].iloc[pos - lookback:pos].min()
        assert f["range_hi"].iloc[i] == pytest.approx(esperado_hi), f"range_hi em {t} inclui a barra atual"
        assert f["range_lo"].iloc[i] == pytest.approx(esperado_lo), f"range_lo em {t} inclui a barra atual"


def test_vwap_reseta_por_dia(series):
    """VWAP de day trade reseta na abertura. Sem reset ele vira media longa."""
    m1, m5 = series
    f = build_feature_frame(m1, m5)
    dias = f.index.normalize().unique()
    assert len(dias) >= 2, "fixture precisa de mais de um dia"
    for dia in dias:
        bloco = f[f.index.normalize() == dia]
        primeira = bloco.iloc[0]
        tipico = (primeira["high"] + primeira["low"] + primeira["close"]) / 3.0
        assert primeira["vwap"] == pytest.approx(tipico, rel=1e-6), (
            f"VWAP nao resetou em {dia}"
        )


def test_nenhum_shift_negativo_no_codigo():
    """Varredura por AST: shift(-n) / diff(-n) sao lookahead explicito.

    Usa AST em vez de busca textual para nao acusar as proprias
    docstrings que explicam o problema.
    """
    import ast

    src_path = Path(__file__).resolve().parents[1] / "b3bot" / "features.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    suspeitos = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("shift", "diff"):
            continue
        args = list(node.args) + [kw.value for kw in node.keywords if kw.arg in ("periods", "n")]
        for arg in args:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                suspeitos.append(f"linha {node.lineno}: .{node.func.attr}(-...)")
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value < 0:
                suspeitos.append(f"linha {node.lineno}: .{node.func.attr}({arg.value})")

    assert not suspeitos, "lookahead explicito em features.py: " + "; ".join(suspeitos)
