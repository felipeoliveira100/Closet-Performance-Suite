"""Invariantes do backtest.

Nao testam se a estrategia da lucro - testam se o backtest esta MENTINDO.
Um backtest que viola qualquer coisa aqui produz numeros que nao poderiam
ter acontecido no mercado.
"""
from __future__ import annotations

import pandas as pd
import pytest

from b3bot.backtest import run_backtest
from b3bot.data.synthetic import generate_m1, resample_to_m5
from b3bot.decision import build_decisor
from b3bot.features import build_feature_frame


@pytest.fixture(scope="module")
def frame():
    m1 = generate_m1(n_bars=6000, seed=21)
    return build_feature_frame(m1, resample_to_m5(m1))


@pytest.fixture(scope="module")
def result(frame, request):
    from b3bot.config import load_config
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    return run_backtest(cfg, frame, build_decisor(cfg, "rules"), slippage_ticks=1.0), cfg


def test_produz_trades(result):
    res, _ = result
    assert res.n > 0, "fixture sem trades: os testes abaixo nao provariam nada"


def test_pnl_total_bate_com_a_soma_dos_trades(result):
    res, _ = result
    assert res.net == pytest.approx(sum(t.net_brl for t in res.trades))
    assert res.net == pytest.approx(res.gross - res.costs, abs=1e-6)


def test_equity_final_bate_com_pnl(result):
    res, _ = result
    assert res.equity is not None
    assert float(res.equity.iloc[-1]) == pytest.approx(res.net, abs=1e-6)


def test_nunca_excede_o_teto_de_contratos(result):
    res, cfg = result
    for t in res.trades:
        assert t.contracts <= cfg.risk.max_contracts
        assert t.contracts >= 1


def test_nenhuma_posicao_overnight(result):
    """Day trade: entrada e saida no MESMO dia, sempre."""
    res, _ = result
    for t in res.trades:
        assert t.entry_time.date() == t.exit_time.date(), (
            f"trade atravessou a noite: {t.entry_time} -> {t.exit_time}"
        )


def test_saida_nunca_antes_da_entrada(result):
    res, _ = result
    for t in res.trades:
        assert t.exit_time >= t.entry_time


def test_todo_trade_paga_custo(result):
    res, _ = result
    for t in res.trades:
        assert t.cost_brl > 0, "trade sem custo: backtest otimista"
        assert t.net_brl == pytest.approx(t.gross_brl - t.cost_brl)


def test_precos_respeitam_o_tick(result):
    res, cfg = result
    tick = cfg.symbol.tick_size
    for t in res.trades:
        assert t.entry_price % tick == 0, f"entrada {t.entry_price} fora do tick de {tick}"
        assert t.exit_price % tick == 0, f"saida {t.exit_price} fora do tick de {tick}"


def test_pnl_coerente_com_pontos_e_valor_do_ponto(result):
    res, cfg = result
    for t in res.trades:
        esperado = t.points * t.contracts * cfg.symbol.point_value
        assert t.gross_brl == pytest.approx(esperado, abs=1e-6)


def test_perda_por_trade_limitada_pelo_stop(result):
    """Nenhum trade pode perder muito mais do que o stop dimensionado.

    Tolerancia: 1 stop + slippage + custo. Se estourar, o backtest esta
    deixando o stop passar.
    """
    res, cfg = result
    for t in res.trades:
        if t.net_brl >= 0:
            continue
        limite = cfg.risk.max_stop_points * t.contracts * cfg.symbol.point_value \
            + t.cost_brl + 4 * cfg.symbol.tick_size * t.contracts * cfg.symbol.point_value
        assert abs(t.net_brl) <= limite, (
            f"perda R$ {t.net_brl:.2f} acima do teto de risco R$ {limite:.2f}"
        )


def test_kill_switch_limita_a_perda_diaria(result):
    """Nenhum dia deve perder muito mais que max_daily_loss_brl.

    Tolerancia de um trade: o kill switch dispara APOS o trade que cruza o
    limite, nao antes dele.
    """
    res, cfg = result
    if res.daily_pnl is None or res.daily_pnl.empty:
        pytest.skip("sem PnL diario")
    pior = float(res.daily_pnl.min())
    pior_trade = min((t.net_brl for t in res.trades), default=0.0)
    teto = -(cfg.risk.max_daily_loss_brl + abs(pior_trade))
    assert pior >= teto, f"dia perdeu R$ {pior:.2f}, abaixo do teto R$ {teto:.2f}"


def test_resultado_e_reproduzivel(frame):
    """Mesmo frame + mesmo decisor = mesmo resultado. Se nao, ha estado
    escondido e nenhuma conclusao do backtest se sustenta."""
    from b3bot.config import load_config
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    a = run_backtest(cfg, frame, build_decisor(cfg, "rules"))
    b = run_backtest(cfg, frame, build_decisor(cfg, "rules"))
    assert a.n == b.n
    assert a.net == pytest.approx(b.net)


def test_slippage_maior_piora_o_resultado(frame):
    """Sanity check do modelo de execucao: custo de execucao maior nao pode
    melhorar o resultado."""
    from b3bot.config import load_config
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    baixo = run_backtest(cfg, frame, build_decisor(cfg, "rules"), slippage_ticks=0.0)
    alto = run_backtest(cfg, frame, build_decisor(cfg, "rules"), slippage_ticks=4.0)
    assert alto.net <= baixo.net, "mais slippage melhorou o resultado: modelo furado"
