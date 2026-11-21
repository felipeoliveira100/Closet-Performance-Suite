"""Resolucao do codigo do contrato futuro vigente na B3.

Por que isso tem modulo proprio: hardcodar "WIN" ou "WINZ25" e um dos
bugs mais comuns e mais caros em robo de futuros. Chega o vencimento, o
simbolo deixa de existir ou perde liquidez, e o robo passa a operar um
contrato morto (ou nao opera nada e voce descobre tarde).

Regras implementadas:
  WIN/IND (mini-indice e indice cheio): vence nos meses PARES
      (fev, abr, jun, ago, out, dez), na quarta-feira mais proxima do dia 15.
  WDO/DOL (mini-dolar e dolar cheio): vence TODO mes,
      no primeiro dia util do mes de vencimento.

A resolucao calculada e sempre conferida contra os simbolos que o MT5
realmente oferece (`validate_against_mt5`). Se divergir, o MT5 ganha -
porque e ele que vai receber a ordem.
"""
from __future__ import annotations

import calendar
import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# Codigos de mes do mercado futuro (padrao internacional)
MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}
CODE_TO_MONTH = {v: k for k, v in MONTH_CODES.items()}

# Meses de vencimento por raiz
EVEN_MONTHS = (2, 4, 6, 8, 10, 12)
ALL_MONTHS = tuple(range(1, 13))

EXPIRY_MONTHS: dict[str, tuple[int, ...]] = {
    "WIN": EVEN_MONTHS,
    "IND": EVEN_MONTHS,
    "WDO": ALL_MONTHS,
    "DOL": ALL_MONTHS,
}


def month_code(month: int) -> str:
    if month not in MONTH_CODES:
        raise ValueError(f"mes invalido: {month}")
    return MONTH_CODES[month]


def _wednesday_nearest_15(year: int, month: int) -> date:
    """Quarta-feira mais proxima do dia 15.

    Se o dia 15 e quarta, e ele mesmo. Caso contrario, compara a quarta
    anterior e a seguinte e devolve a mais proxima (empate -> a posterior,
    que e o caso de 15 cair na sabado/domingo equidistante).
    """
    d15 = date(year, month, 15)
    # weekday(): segunda=0 ... quarta=2
    offset_back = (d15.weekday() - 2) % 7
    prev_wed = d15 - timedelta(days=offset_back)
    next_wed = prev_wed + timedelta(days=7)
    if offset_back == 0:
        return d15
    dist_prev = (d15 - prev_wed).days
    dist_next = (next_wed - d15).days
    return prev_wed if dist_prev < dist_next else next_wed


def _first_business_day(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() >= 5:  # sabado/domingo
        d += timedelta(days=1)
    return d


def expiry_date(root: str, year: int, month: int) -> date:
    """Data de vencimento do contrato.

    Nao considera feriados nacionais da B3 (o vencimento antecipa para o
    dia util anterior quando cai em feriado). Como a rolagem usa uma
    folga de dias (roll_days_before), isso nao muda a decisao na pratica -
    mas se voce quiser precisao, plugue um calendario de feriados aqui.
    """
    root = root.upper()
    if root in ("WDO", "DOL"):
        return _first_business_day(year, month)
    return _wednesday_nearest_15(year, month)


def _candidate_months(root: str, today: date) -> list[tuple[int, int]]:
    months = EXPIRY_MONTHS.get(root.upper(), EVEN_MONTHS)
    out: list[tuple[int, int]] = []
    for year in (today.year, today.year + 1):
        for m in months:
            out.append((year, m))
    return sorted(out)


def active_contract(root: str = "WIN", today: date | None = None, roll_days_before: int = 2) -> str:
    """Codigo do contrato a operar hoje, ex: 'WINZ25'.

    `roll_days_before` e a folga de rolagem: N dias uteis antes do
    vencimento o robo ja passa para o contrato seguinte, porque a
    liquidez migra antes da data formal. 2 e um default sensato para WIN.
    """
    root = root.upper()
    today = today or date.today()
    for year, month in _candidate_months(root, today):
        exp = expiry_date(root, year, month)
        roll = exp - timedelta(days=roll_days_before)
        if today < roll:
            return f"{root}{month_code(month)}{year % 100:02d}"
    # fallback defensivo: nunca deveria chegar aqui
    year, month = _candidate_months(root, today)[-1]
    logger.warning("active_contract: fallback para %s %s", year, month)
    return f"{root}{month_code(month)}{year % 100:02d}"


def parse_contract(code: str) -> tuple[str, int, int]:
    """'WINZ25' -> ('WIN', 2025, 12)."""
    code = code.strip().upper()
    if len(code) < 5:
        raise ValueError(f"codigo de contrato invalido: {code!r}")
    yy = code[-2:]
    mcode = code[-3]
    root = code[:-3]
    if not yy.isdigit() or mcode not in CODE_TO_MONTH:
        raise ValueError(f"codigo de contrato invalido: {code!r}")
    return root, 2000 + int(yy), CODE_TO_MONTH[mcode]


def days_to_expiry(code: str, today: date | None = None) -> int:
    root, year, month = parse_contract(code)
    return (expiry_date(root, year, month) - (today or date.today())).days


def validate_against_mt5(code: str, root: str, mt5_module) -> str:
    """Confere o codigo calculado contra os simbolos reais do terminal.

    Se o codigo existir no MT5, usa ele. Se nao existir, escolhe entre os
    simbolos disponiveis com a mesma raiz o de vencimento mais proximo
    ainda no futuro. O MT5 e a fonte da verdade: e ele que recebe a ordem.
    """
    if mt5_module is None:
        return code
    try:
        if mt5_module.symbol_info(code) is not None:
            return code
        candidates = mt5_module.symbols_get(f"{root}*") or []
        valid: list[tuple[int, str]] = []
        for sym in candidates:
            name = getattr(sym, "name", str(sym))
            try:
                dte = days_to_expiry(name)
            except ValueError:
                continue
            if dte >= 0:
                valid.append((dte, name))
        if valid:
            valid.sort()
            chosen = valid[0][1]
            logger.warning(
                "contrato calculado %s nao existe no MT5; usando %s (vence em %d dias)",
                code, chosen, valid[0][0],
            )
            return chosen
        logger.error("nenhum contrato %s* valido encontrado no MT5", root)
    except Exception as exc:  # noqa: BLE001
        logger.warning("validate_against_mt5 falhou (%s); mantendo %s", exc, code)
    return code
