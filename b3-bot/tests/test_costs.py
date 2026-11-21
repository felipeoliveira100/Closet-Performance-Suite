"""Testes do modelo de custos e da tributacao de day trade."""
from __future__ import annotations

import pytest

from b3bot.costs import CostModel


def test_custo_cobrado_nas_duas_pontas(cfg):
    cm = CostModel(cfg.costs, cfg.symbol)
    rt = cm.round_trip(1)
    esperado = (cfg.costs.brokerage_per_order_brl * 2
                + cfg.costs.exchange_fee_per_contract_brl * 1 * 2
                + cfg.costs.registration_fee_per_order_brl * 2)
    assert rt.total == pytest.approx(esperado)


def test_custo_escala_com_contratos(cfg):
    cm = CostModel(cfg.costs, cfg.symbol)
    assert cm.round_trip(5).total > cm.round_trip(1).total


def test_breakeven_em_pontos_diminui_com_tamanho(cfg):
    """Taxas por ordem se diluem; por contrato nao. Logo o breakeven cai."""
    cm = CostModel(cfg.costs, cfg.symbol)
    assert cm.breakeven_points(5) < cm.breakeven_points(1)


def test_pnl_long_e_short_sao_simetricos(cfg):
    cm = CostModel(cfg.costs, cfg.symbol)
    assert cm.gross_pnl_brl(140000, 140100, 1, "BUY") == pytest.approx(100 * cfg.symbol.point_value)
    assert cm.gross_pnl_brl(140000, 139900, 1, "SELL") == pytest.approx(100 * cfg.symbol.point_value)
    assert cm.gross_pnl_brl(140000, 139900, 1, "BUY") == pytest.approx(-100 * cfg.symbol.point_value)


def test_liquido_e_sempre_menor_que_bruto(cfg):
    cm = CostModel(cfg.costs, cfg.symbol)
    for exit_px in (139500, 140000, 140500):
        g = cm.gross_pnl_brl(140000, exit_px, 2, "BUY")
        n = cm.net_pnl_brl(140000, exit_px, 2, "BUY")
        assert n < g


def test_day_trade_nao_tem_isencao_de_20k(cfg):
    """A isencao de R$ 20 mil vale so para swing em acoes a vista."""
    cm = CostModel(cfg.costs, cfg.symbol)
    t = cm.daytrade_tax(1000.0)
    assert t["ir_devido"] == pytest.approx(1000.0 * cfg.costs.daytrade_ir_rate)
    assert t["darf_a_pagar"] > 0


def test_irrf_abate_do_devido(cfg):
    cm = CostModel(cfg.costs, cfg.symbol)
    t = cm.daytrade_tax(1000.0, irrf_already_withheld_brl=50.0)
    assert t["darf_a_pagar"] == pytest.approx(200.0 - 50.0)


def test_prejuizo_nao_gera_imposto_e_fica_para_compensar(cfg):
    cm = CostModel(cfg.costs, cfg.symbol)
    t = cm.daytrade_tax(-500.0)
    assert t["ir_devido"] == 0.0
    assert t["darf_a_pagar"] == 0.0
    assert t["prejuizo_a_compensar"] == pytest.approx(500.0)


def test_irrf_maior_que_devido_nao_gera_darf_negativo(cfg):
    cm = CostModel(cfg.costs, cfg.symbol)
    t = cm.daytrade_tax(100.0, irrf_already_withheld_brl=500.0)
    assert t["darf_a_pagar"] == 0.0
