"""Custos da B3 e tributacao de day trade.

Por que isso existe como modulo proprio: um robo de 1 minuto em WIN faz
muitas operacoes, e custo + imposto consomem a maior parte de um edge
pequeno. O custo precisa entrar no backtest e no sizing, nao ser
descontado depois numa planilha.

CONFIRME os valores no config.yaml com a sua corretora e com a tabela
vigente da B3. Emolumentos de day trade variam conforme o seu volume
medio diario dos meses anteriores.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import CostsConfig, SymbolConfig


@dataclass(frozen=True)
class TradeCost:
    """Custo de um round-trip (abrir + fechar)."""

    brokerage: float
    exchange_fees: float
    registration: float
    iss: float

    @property
    def total(self) -> float:
        return self.brokerage + self.exchange_fees + self.registration + self.iss

    def as_points(self, contracts: int, point_value: float) -> float:
        """Custo convertido em pontos de WIN - util para comparar com o stop."""
        if contracts <= 0 or point_value <= 0:
            return 0.0
        return self.total / (contracts * point_value)


class CostModel:
    def __init__(self, costs: CostsConfig, symbol: SymbolConfig) -> None:
        self.c = costs
        self.s = symbol

    def round_trip(self, contracts: int) -> TradeCost:
        """Custo total de abrir e fechar `contracts` contratos.

        Duas pontas: 2 ordens, 2x emolumentos sobre o total de contratos.
        """
        if contracts <= 0:
            return TradeCost(0.0, 0.0, 0.0, 0.0)
        legs = 2
        brokerage = self.c.brokerage_per_order_brl * legs
        exchange = self.c.exchange_fee_per_contract_brl * contracts * legs
        registration = self.c.registration_fee_per_order_brl * legs
        iss = brokerage * self.c.iss_rate
        return TradeCost(brokerage, exchange, registration, iss)

    def gross_pnl_brl(self, entry_price: float, exit_price: float, contracts: int, side: str) -> float:
        """PnL bruto em reais. side: BUY (long) ou SELL (short)."""
        direction = 1.0 if side.upper() == "BUY" else -1.0
        points = (exit_price - entry_price) * direction
        return points * contracts * self.s.point_value

    def net_pnl_brl(self, entry_price: float, exit_price: float, contracts: int, side: str) -> float:
        """PnL liquido de custos (ainda antes do IR)."""
        gross = self.gross_pnl_brl(entry_price, exit_price, contracts, side)
        return gross - self.round_trip(contracts).total

    def breakeven_points(self, contracts: int) -> float:
        """Quantos pontos o trade precisa andar so para pagar o custo.

        Se o seu alvo em pontos nao for varias vezes isso, a estrategia
        nao sobrevive ao custo. E a primeira coisa a checar.
        """
        return self.round_trip(contracts).as_points(contracts, self.s.point_value)

    def daytrade_tax(self, monthly_net_result_brl: float, irrf_already_withheld_brl: float = 0.0) -> dict[str, float]:
        """IR de day trade do mes.

        Day trade: 20%, sem isenção de R$ 20 mil (essa vale só para swing
        em ações à vista). O IRRF de 1% retido na fonte é antecipação e
        abate do devido. Prejuízo de day trade compensa lucro de day trade.
        DARF código 6015.
        """
        if monthly_net_result_brl <= 0:
            return {
                "base": 0.0,
                "ir_devido": 0.0,
                "irrf_retido": irrf_already_withheld_brl,
                "darf_a_pagar": 0.0,
                "prejuizo_a_compensar": abs(monthly_net_result_brl),
            }
        ir = monthly_net_result_brl * self.c.daytrade_ir_rate
        darf = max(0.0, ir - irrf_already_withheld_brl)
        return {
            "base": monthly_net_result_brl,
            "ir_devido": ir,
            "irrf_retido": irrf_already_withheld_brl,
            "darf_a_pagar": darf,
            "prejuizo_a_compensar": 0.0,
        }
