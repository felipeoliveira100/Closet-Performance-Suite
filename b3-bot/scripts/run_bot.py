#!/usr/bin/env python3
"""Sobe o robo. O MODO vem do config.yaml, nao da linha de comando.

    mode: paper -> analisa e registra, NAO envia nada
    mode: demo  -> envia para a conta MT5 (que precisa ser DEMO)
    mode: live  -> envia para conta real; exige live_confirmation no config

Uso:
    python scripts/run_bot.py                      # usa o config
    python scripts/run_bot.py --decisor hybrid      # sobrescreve o decisor
    python scripts/run_bot.py --synthetic           # paper com dados sinteticos (Linux)
    python scripts/run_bot.py --steps 200           # para depois de N ciclos

Ordem recomendada, e ela existe por um motivo:
    1. paper com dados reais por alguns dias   -> audite o journal
    2. demo                                     -> confira que a ordem casa com a decisao
    3. live no menor tamanho possivel
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b3bot.config import load_config
from b3bot.decision import build_decisor
from b3bot.engine import TradingEngine
from b3bot.execution.paper import PaperBroker
from b3bot.journal import Journal, setup_logging


def build_feed_and_broker(cfg, use_synthetic: bool):
    if use_synthetic:
        from b3bot.data.synthetic import SyntheticFeed
        if cfg.sends_real_orders:
            raise SystemExit(
                "ERRO: --synthetic com mode demo/live nao faz sentido - voce mandaria "
                "ordem real baseada em dado inventado. Use mode: paper."
            )
        print("Feed SINTETICO: dados inventados, para testar o encanamento.\n")
        return SyntheticFeed(n_bars=3000, seed=7), PaperBroker(cfg)

    from b3bot.data.mt5_feed import MT5Feed
    feed = MT5Feed(cfg.symbol, cfg.mt5)
    feed.connect()
    print("\n--- HEALTH CHECK DO TERMINAL ---")
    for k, v in feed.health().items():
        print(f"  {k:24}: {v}")
    print("--------------------------------\n")

    if cfg.sends_real_orders:
        from b3bot.execution.mt5_broker import MT5Broker
        broker = MT5Broker(cfg, feed.symbol)
        broker.connect()
    else:
        broker = PaperBroker(cfg)
        broker.connect()
    return feed, broker


def confirm_live(cfg) -> None:
    if cfg.mode != "live":
        return
    print("=" * 70)
    print("  ATENCAO: MODO LIVE - ORDENS COM DINHEIRO REAL")
    print("=" * 70)
    print(f"  contratos maximos:  {cfg.risk.max_contracts}")
    print(f"  risco por operacao: R$ {cfg.risk.risk_per_trade_brl:.2f}")
    print(f"  PERDA MAXIMA DIARIA (kill switch): R$ {cfg.risk.max_daily_loss_brl:.2f}")
    print(f"  operacoes maximas no dia: {cfg.risk.max_daily_trades}")
    print(f"  pior caso aproximado do dia: R$ -{cfg.risk.max_daily_loss_brl:.2f}")
    print("=" * 70)
    print("  Ctrl+C nos proximos 10 segundos para abortar.")
    print("=" * 70)
    try:
        for i in range(10, 0, -1):
            print(f"  subindo em {i}s...", end="\r", flush=True)
            time.sleep(1)
        print(" " * 40, end="\r")
    except KeyboardInterrupt:
        raise SystemExit("\nabortado pelo usuario - nada foi enviado.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--decisor", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(args.log_level or cfg.logging.level)

    print(f"modo: {cfg.mode.upper()} | decisor: {args.decisor or cfg.active_decisor} | "
          f"envia ordens reais: {'SIM' if cfg.sends_real_orders else 'NAO'}")
    confirm_live(cfg)

    decisor = build_decisor(cfg, args.decisor)
    if decisor.name == "ml" and not getattr(decisor, "ready", True):
        print("AVISO: decisor ML sem modelo treinado - vai devolver HOLD sempre.")
        print("       Treine com: python scripts/train_ml.py")

    feed, broker = build_feed_and_broker(cfg, args.synthetic)
    journal = Journal(cfg.logging.journal_path, mode=cfg.mode)
    engine = TradingEngine(cfg, feed, broker, decisor, journal)

    try:
        if args.synthetic:
            # avanca o feed sintetico a cada ciclo, sem esperar o relogio
            steps = args.steps or 1200
            for _ in range(steps):
                if feed.exhausted:
                    break
                engine.step(now=feed.now.to_pydatetime(), force=False)
                feed.advance(1)
            engine.shutdown()
        else:
            engine.run(poll_seconds=args.poll, max_steps=args.steps)
    finally:
        feed.shutdown()
        broker.shutdown()

    print(f"\njournal: {cfg.logging.journal_path}")
    print("audite com:")
    print("  python -c \"import pandas as pd; d=pd.read_json('"
          f"{cfg.logging.journal_path}', lines=True); print(d.event.value_counts())\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
