"""Testes da gestao de risco. Se algum falhar, NAO opere."""
from __future__ import annotations

from datetime import datetime

import pytest

from b3bot.risk import Position, RiskManager

NOW = datetime(2026, 11, 20, 10, 30)


def fresh(cfg):
    rm = RiskManager(cfg)
    rm.state.last_exit_bar = -1000  # sem cooldown pendente
    return rm


def test_kill_switch_bloqueia_novas_entradas(cfg):
    rm = fresh(cfg)
    rm.state.realized_pnl_brl = -(cfg.risk.max_daily_loss_brl + 1)
    d = rm.plan_entry("BUY", 140000, 85, NOW, 100)
    assert not d.approved
    assert rm.state.halted
    # e continua bloqueado em qualquer barra seguinte
    assert not rm.plan_entry("BUY", 140000, 85, NOW, 200).approved
    assert not rm.plan_entry("BUY", 140000, 85, NOW, 999).approved


def test_kill_switch_considera_prejuizo_em_aberto(cfg):
    """Olhar so o realizado deixaria a perda crescer numa posicao aberta."""
    rm = fresh(cfg)
    metade = -cfg.risk.max_daily_loss_brl / 2
    rm.state.realized_pnl_brl = metade
    assert rm.check_kill_switch(open_position_pnl=0.0) is None
    assert rm.check_kill_switch(open_position_pnl=metade - 1) is not None
    assert rm.state.halted


def test_perdas_consecutivas_param_o_dia(cfg):
    rm = fresh(cfg)
    for i in range(cfg.risk.max_consecutive_losses):
        rm.state.register_close(-10.0, bar_index=i * 100)
    assert rm.check_kill_switch() is not None
    assert rm.state.halted


def test_vitoria_zera_contador_de_perdas(cfg):
    rm = fresh(cfg)
    rm.state.register_close(-10.0, 1)
    rm.state.register_close(-10.0, 2)
    assert rm.state.consecutive_losses == 2
    rm.state.register_close(+10.0, 3)
    assert rm.state.consecutive_losses == 0


def test_payoff_liquido_nunca_negativo(cfg):
    """A trava que eu mesmo descobri: com ATR baixo o piso do stop
    distorcia o alvo e o R:R liquido caia abaixo de 1."""
    rm = fresh(cfg)
    for atr in (10, 20, 25, 30, 40, 60, 85, 120, 150, 200, 280):
        rm.state.last_exit_bar = -1000
        rm.state.halted = False
        d = rm.plan_entry("BUY", 140000, atr, NOW, 100)
        if d.approved:
            assert d.plan is not None
            assert d.plan.rr_ratio >= cfg.risk.min_rr_after_costs, (
                f"ATR {atr}: R:R liquido {d.plan.rr_ratio:.2f} abaixo do minimo"
            )
            assert d.plan.reward_brl > 0, f"ATR {atr}: retorno liquido nao positivo"


def test_stop_sempre_existe_e_do_lado_certo(cfg):
    rm = fresh(cfg)
    for side, cmp_ in (("BUY", lambda p: p.stop_price < 140000 < p.target_price),
                       ("SELL", lambda p: p.target_price < 140000 < p.stop_price)):
        rm.state.last_exit_bar = -1000
        d = rm.plan_entry(side, 140000, 85, NOW, 100)
        assert d.approved and d.plan is not None
        assert cmp_(d.plan), f"{side}: stop/alvo do lado errado"


def test_volatilidade_extrema_e_rejeitada(cfg):
    rm = fresh(cfg)
    atr_absurdo = (cfg.risk.max_stop_points / cfg.risk.stop_atr_mult) + 100
    d = rm.plan_entry("BUY", 140000, atr_absurdo, NOW, 100)
    assert not d.approved and "maximo" in d.reason


def test_janela_de_entrada_respeitada(cfg):
    rm = fresh(cfg)
    antes = datetime(2026, 11, 20, cfg.session.entry_start.hour, cfg.session.entry_start.minute) \
        .replace(minute=max(0, cfg.session.entry_start.minute - 10))
    depois = datetime(2026, 11, 20, cfg.session.entry_end.hour, cfg.session.entry_end.minute + 1)
    assert not rm.plan_entry("BUY", 140000, 85, antes, 100).approved
    assert not rm.plan_entry("BUY", 140000, 85, depois, 100).approved


def test_zeragem_forcada_no_fim_do_pregao(cfg):
    rm = fresh(cfg)
    pos = Position("BUY", 140000, 1, 139800, 140300, NOW)
    tarde = datetime(2026, 11, 20, cfg.session.force_flat_at.hour,
                     cfg.session.force_flat_at.minute)
    out = rm.check_exit(pos, 139990, 140010, 140000, tarde)
    assert out is not None and out[0] == "session_end"


def test_stop_tem_precedencia_sobre_alvo(cfg):
    """Barra que toca os dois: assumir o alvo infla o backtest."""
    rm = fresh(cfg)
    pos = Position("BUY", 140000, 1, 139800, 140300, NOW)
    out = rm.check_exit(pos, low=139700, high=140400, price=140000, now=NOW)
    assert out is not None and out[0] == "stop"


def test_limite_de_operacoes_por_dia(cfg):
    rm = fresh(cfg)
    rm.state.trades = cfg.risk.max_daily_trades
    d = rm.plan_entry("BUY", 140000, 85, NOW, 100)
    assert not d.approved and "operacoes" in d.reason


def test_cooldown_apos_saida(cfg):
    rm = fresh(cfg)
    rm.state.last_exit_bar = 100
    assert not rm.plan_entry("BUY", 140000, 85, NOW, 101).approved
    assert rm.plan_entry("BUY", 140000, 85, NOW,
                         100 + cfg.risk.cooldown_bars_after_exit).approved


def test_teto_de_contratos_respeitado(cfg):
    """Mesmo com risco por operacao absurdo, max_contracts manda."""
    from dataclasses import replace
    cfg2 = replace(cfg, risk=replace(cfg.risk, risk_per_trade_brl=100_000.0))
    rm = fresh(cfg2)
    d = rm.plan_entry("BUY", 140000, 85, NOW, 100)
    assert d.approved and d.plan is not None
    assert d.plan.contracts <= cfg2.risk.max_contracts


def test_atr_invalido_nao_opera(cfg):
    rm = fresh(cfg)
    for atr in (0, -5, float("nan"), float("inf")):
        assert not rm.plan_entry("BUY", 140000, atr, NOW, 100).approved
