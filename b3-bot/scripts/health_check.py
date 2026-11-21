#!/usr/bin/env python3
"""Checklist ANTES de deixar o robo mandar ordem. Rode no Windows.

Verifica, em ordem: terminal, conta, contrato vigente, dados, custos e
dimensionamento. Qualquer FALHA aqui significa nao operar.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b3bot.config import load_config
from b3bot.costs import CostModel
from b3bot.journal import setup_logging
from b3bot.risk import RiskManager

OK, FAIL, WARN = "  [OK]  ", "  [FALHA]", "  [AVISO]"


def main() -> int:
    setup_logging("WARNING")
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    problemas, avisos = 0, 0

    print("=" * 72)
    print("HEALTH CHECK - robo WIN")
    print("=" * 72)
    print(f"\nmodo configurado: {cfg.mode.upper()} | "
          f"envia ordens reais: {'SIM' if cfg.sends_real_orders else 'NAO'}")
    print(f"decisor ativo: {cfg.active_decisor}")
    print(f"direcao: long={cfg.direction.allow_long} short={cfg.direction.allow_short}")

    # --- 1. terminal e conta ------------------------------------------------
    print("\n1) TERMINAL E CONTA")
    try:
        from b3bot.data.mt5_feed import MT5Feed
        feed = MT5Feed(cfg.symbol, cfg.mt5)
        feed.connect()
        h = feed.health()
        for k, v in h.items():
            print(f"      {k:24}: {v}")

        if not h["terminal_connected"]:
            print(FAIL, "terminal nao conectado a corretora"); problemas += 1
        else:
            print(OK, "terminal conectado")

        if not h["algo_trading_enabled"]:
            print(FAIL, "AutoTrading DESLIGADO - toda ordem sera recusada. "
                        "Ligue 'Algo Trading' no MT5."); problemas += 1
        else:
            print(OK, "AutoTrading ligado")

        if cfg.mode == "live" and h["account_is_demo"]:
            print(WARN, "mode live mas a conta e DEMO - nada de real vai acontecer")
            avisos += 1
        elif cfg.mode == "live":
            print(WARN, "*** CONTA REAL: ordens valem dinheiro ***"); avisos += 1
        elif h["account_is_demo"]:
            print(OK, "conta DEMO")

        # --- 2. contrato ----------------------------------------------------
        print("\n2) CONTRATO")
        dte = h["days_to_expiry"]
        print(f"      simbolo: {h['symbol']} | vence em {dte} dias")
        if dte < 0:
            print(FAIL, "contrato VENCIDO"); problemas += 1
        elif dte <= 2:
            print(WARN, f"vence em {dte} dias - a liquidez ja esta migrando. "
                        "Confira se deve rolar."); avisos += 1
        else:
            print(OK, "contrato vigente")

        # --- 3. dados -------------------------------------------------------
        print("\n3) DADOS")
        snap = feed.snapshot(m1_bars=300, m5_bars=100)
        print(f"      candles de 1min: {len(snap.m1)} | ultimo: {snap.m1.last_time}")
        print(f"      candles de 5min: {len(snap.m5)} | ultimo: {snap.m5.last_time}")
        print(f"      bid/ask: {snap.bid} / {snap.ask} | spread: {snap.spread_points} pts")
        if len(snap.m1) < 150:
            print(FAIL, f"so {len(snap.m1)} candles de 1min - insuficiente para o warmup")
            problemas += 1
        else:
            print(OK, "historico suficiente")
        if snap.spread_points and snap.spread_points > cfg.symbol.tick_size * 3:
            print(WARN, f"spread largo ({snap.spread_points} pts): mercado fino ou fora do pregao")
            avisos += 1

        from b3bot.features import build_feature_frame, latest_snapshot
        frame = build_feature_frame(snap.m1.df, snap.m5.df)
        fs = latest_snapshot(frame)
        print(f"      analise atual: {fs.summary()}")
        if fs.regime == "unknown":
            print(WARN, "regime indefinido (warmup do 5min)"); avisos += 1
        else:
            print(OK, f"regime: {fs.regime}")
        feed.shutdown()
        atr_real = fs.atr
    except Exception as exc:  # noqa: BLE001
        print(FAIL, f"{type(exc).__name__}: {exc}")
        print("\n      Em Linux/macOS isso e esperado: MT5 so existe no Windows.")
        problemas += 1
        atr_real = 85.0

    # --- 4. custos ----------------------------------------------------------
    print("\n4) CUSTOS (CONFIRME com a sua corretora)")
    cm = CostModel(cfg.costs, cfg.symbol)
    for n in (1, cfg.risk.max_contracts):
        rt = cm.round_trip(n)
        print(f"      {n} contrato(s): R$ {rt.total:.2f} ida e volta = "
              f"{cm.breakeven_points(n):.1f} pts para empatar")
    print(f"      emolumentos configurados: R$ {cfg.costs.exchange_fee_per_contract_brl}/contrato/ponta")
    print(WARN, "emolumentos de day trade variam com o seu volume medio diario. "
                "Confira a tabela vigente."); avisos += 1

    # --- 5. dimensionamento -------------------------------------------------
    print("\n5) RISCO E DIMENSIONAMENTO")
    rm = RiskManager(cfg)
    rm.state.last_exit_bar = -1000
    agora = datetime.now().replace(hour=10, minute=30)
    d = rm.plan_entry("BUY", 140000, atr_real, agora, 100)
    if d.approved and d.plan:
        p = d.plan
        print(f"      com ATR {atr_real:.0f}: {p.contracts} contrato(s), "
              f"stop {p.stop_points:.0f} pts (R$ {p.risk_brl:.2f}), "
              f"alvo {p.target_points:.0f} pts, R:R {p.rr_ratio:.2f}")
        print(OK, "dimensionamento valido")
    else:
        print(WARN, f"nao operaria agora: {d.reason}"); avisos += 1
    print(f"      KILL SWITCH: para o dia em R$ -{cfg.risk.max_daily_loss_brl:.2f}")
    print(f"      pior caso do dia: ~R$ -{cfg.risk.max_daily_loss_brl:.2f} "
          f"({cfg.risk.max_daily_trades} trades max, {cfg.risk.max_consecutive_losses} perdas seguidas param)")

    # --- 6. modelo de ML ----------------------------------------------------
    if cfg.active_decisor in ("ml", "hybrid"):
        print("\n6) MODELO DE ML")
        try:
            from b3bot.decision.ml_local import load_bundle
            b = load_bundle(cfg.decisor_params("ml")["model_path"])
            print(f"      treinado em: {b['trained_at']} | {b['n_samples']} amostras")
            print(OK, "modelo carregado")
        except Exception as exc:  # noqa: BLE001
            print(FAIL, f"modelo indisponivel: {exc}"); problemas += 1

    print("\n" + "=" * 72)
    if problemas:
        print(f"RESULTADO: {problemas} FALHA(S), {avisos} aviso(s). NAO OPERE.")
        return 1
    print(f"RESULTADO: nenhuma falha, {avisos} aviso(s). Leia os avisos antes de subir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
