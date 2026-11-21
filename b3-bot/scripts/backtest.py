#!/usr/bin/env python3
"""Backtest de um decisor, ou comparacao dos tres no mesmo historico.

Uso:
    python scripts/backtest.py --compare                 # os 3 lado a lado
    python scripts/backtest.py --decisor hybrid
    python scripts/backtest.py --compare --source mt5    # historico real (Windows)
    python scripts/backtest.py --compare --csv win_1min.csv

O LLM fica fora do backtest por padrao: custaria uma chamada de API por
candle e, por nao ser deterministico, o resultado nao seria reproduzivel.
Compare-o em paper trading ao vivo (scripts/run_bot.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from b3bot.backtest import run_backtest
from b3bot.config import load_config
from b3bot.data.synthetic import generate_m1, resample_to_m5
from b3bot.decision import build_all_decisors, build_decisor
from b3bot.features import build_feature_frame
from b3bot.journal import setup_logging


def load_m1(args) -> tuple[pd.DataFrame, str]:
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=[0], index_col=0)
        df.columns = [c.lower() for c in df.columns]
        if "tick_volume" not in df.columns:
            df["tick_volume"] = df.get("volume", 1.0)
        return df.sort_index(), f"CSV {args.csv}"
    if args.source == "mt5":
        from b3bot.data.mt5_feed import MT5Feed
        cfg = load_config(args.config)
        feed = MT5Feed(cfg.symbol, cfg.mt5)
        feed.connect()
        try:
            return feed.snapshot(m1_bars=args.bars, m5_bars=args.bars // 5 + 10).m1.df, \
                   f"MT5 {feed.symbol}"
        finally:
            feed.shutdown()
    return generate_m1(n_bars=args.bars, seed=args.seed), "SINTETICO"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--decisor", default=None, help="rules | ml | llm | hybrid")
    ap.add_argument("--compare", action="store_true", help="roda rules, ml e hybrid")
    ap.add_argument("--source", choices=["synthetic", "mt5"], default="synthetic")
    ap.add_argument("--csv")
    ap.add_argument("--bars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging("INFO" if args.verbose else "WARNING")
    cfg = load_config(args.config)

    m1, origem = load_m1(args)
    frame = build_feature_frame(m1, resample_to_m5(m1))

    print("=" * 78)
    print(f"BACKTEST | fonte: {origem} | {len(frame)} candles de 1min")
    print(f"periodo: {frame.index[0]} -> {frame.index[-1]}")
    print(f"direcao: long={cfg.direction.allow_long} short={cfg.direction.allow_short} | "
          f"max {cfg.risk.max_contracts} contrato(s) | slippage {args.slippage_ticks} tick(s)")
    if origem == "SINTETICO":
        print("\n*** DADOS SINTETICOS: valida o pipeline, NAO valida estrategia. ***")
        print("*** Resultado aqui nao diz nada sobre o mercado real.           ***")
    print("=" * 78)

    decisors = build_all_decisors(cfg) if args.compare else \
        {args.decisor or cfg.active_decisor: build_decisor(cfg, args.decisor)}

    results = {}
    for name, d in decisors.items():
        res = run_backtest(cfg, frame, d, slippage_ticks=args.slippage_ticks,
                           verbose=args.verbose)
        results[name] = res
        print()
        print(res.summary())

    if len(results) > 1:
        print("\n" + "=" * 78)
        print("COMPARATIVO")
        print("=" * 78)
        hdr = f"{'decisor':10} {'trades':>7} {'liquido':>12} {'esper/trade':>12} {'acerto':>8} {'PF':>7} {'maxDD':>10}"
        print(hdr); print("-" * len(hdr))
        for name, r in sorted(results.items(), key=lambda kv: -kv[1].net):
            print(f"{name:10} {r.n:7} {r.net:12.2f} {r.expectancy:12.2f} "
                  f"{100*r.win_rate:7.1f}% {r.profit_factor:7.2f} {r.max_drawdown:10.2f}")
        base = results.get("rules")
        if base is not None and base.n:
            print(f"\nBaseline (rules): R$ {base.net:+.2f} em {base.n} trades.")
            for name, r in results.items():
                if name == "rules":
                    continue
                delta = r.net - base.net
                veredito = "BATEU o baseline" if delta > 0 else "NAO bateu o baseline"
                print(f"  {name:8}: R$ {delta:+9.2f} vs baseline -> {veredito}")
            print("\nSe o ML/hibrido nao bate as regras liquido de custos, nao use o modelo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
