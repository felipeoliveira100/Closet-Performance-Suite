"""Testes da resolucao de contrato - evita operar serie vencida."""
from __future__ import annotations

from datetime import date

import pytest

from b3bot.contracts import active_contract, days_to_expiry, expiry_date, parse_contract


def test_win_vence_so_em_meses_pares():
    for m in (2, 4, 6, 8, 10, 12):
        d = expiry_date("WIN", 2026, m)
        assert d.weekday() == 2, f"{m}/2026 cai em {d}, nao e quarta"


def test_quarta_mais_proxima_do_dia_15():
    # 15/02/2026 e domingo -> quarta 18 (3 dias) e mais perto que 11 (4 dias)
    assert expiry_date("WIN", 2026, 2) == date(2026, 2, 18)
    # 15/08/2026 e sabado -> quarta 12 (3 dias) e mais perto que 19 (4 dias)
    assert expiry_date("WIN", 2026, 8) == date(2026, 8, 12)
    # 15/04/2026 e quarta -> ela mesma
    assert expiry_date("WIN", 2026, 4) == date(2026, 4, 15)


def test_wdo_vence_no_primeiro_dia_util():
    d = expiry_date("WDO", 2026, 12)
    assert d.day <= 3 and d.weekday() < 5


def test_rolagem_antes_do_vencimento():
    venc = expiry_date("WIN", 2026, 12)  # 16/12/2026
    assert active_contract("WIN", date(2026, 12, 10)) == "WINZ26"
    # dois dias antes do vencimento ja rolou para o proximo (fev/27)
    from datetime import timedelta
    assert active_contract("WIN", venc - timedelta(days=1)) == "WING27"
    assert active_contract("WIN", venc + timedelta(days=1)) == "WING27"


def test_nunca_devolve_contrato_vencido():
    for dia in (1, 60, 120, 200, 300, 400):
        from datetime import timedelta
        d = date(2026, 1, 1) + timedelta(days=dia)
        code = active_contract("WIN", d)
        assert days_to_expiry(code, d) > 0, f"{d} -> {code} ja vencido"


def test_parse_round_trip():
    assert parse_contract("WINZ26") == ("WIN", 2026, 12)
    assert parse_contract("WDOF27") == ("WDO", 2027, 1)
    with pytest.raises(ValueError):
        parse_contract("XX")
    with pytest.raises(ValueError):
        parse_contract("WINA26")  # 'A' nao e codigo de mes
