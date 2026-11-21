#!/usr/bin/env python3
"""Treina o modelo de ML e imprime o relatorio de validacao walk-forward.

Uso:
    python scripts/train_ml.py                      # dados sinteticos (so para testar o pipeline)
    python scripts/train_ml.py --source mt5         # historico real do MT5 (Windows)
    python scripts/train_ml.py --csv dados_win.csv  # CSV proprio com OHLCV de 1min
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from b3bot.config import load_config
from b3bot.data.synthetic import generate_m1, resample_to_m5
from b3bot.decision.ml_local import save_bundle, walk_forward_train
from b3bot.features import build_feature_frame
from b3bot.journal import setup_logging


def load_m1(args) -> pd.DataFrame:
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=[0], index_col=0)
        df.columns = [c.lower() for c in df.columns]
        need = {"open", "high", "low", "close"}
        if not need.issubset(df.columns):
            raise SystemExit(f"CSV precisa das colunas {need}, achei {set(df.columns)}")
        if "tick_volume" not in df.columns:
            df["tick_volume"] = df.get("volume", 1.0)
        return df.sort_index()

    if args.source == "mt5":
        from b3bot.data.mt5_feed import MT5Feed
        cfg = load_config(args.config)
        feed = MT5Feed(cfg.symbol, cfg.mt5)
        feed.connect()
        try:
            snap = feed.snapshot(m1_bars=args.bars, m5_bars=args.bars // 5 + 10)
            return snap.m1.df
        finally:
            feed.shutdown()

    print("AVISO: treinando com dados SINTETICOS. Serve para validar o pipeline,")
    print("       NAO para validar estrategia. Use --source mt5 ou --csv para dados reais.\n")
    return generate_m1(n_bars=args.bars, seed=args.seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", choices=["synthetic", "mt5"], default="synthetic")
    ap.add_argument("--csv")
    ap.add_argument("--bars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()

    setup_logging("INFO")
    cfg = load_config(args.config)
    ml_params = cfg.decisor_params("ml")
    out = args.out or ml_params.get("model_path", "models/win_m1.pkl")

    m1 = load_m1(args)
    m5 = resample_to_m5(m1)
    frame = build_feature_frame(m1, m5)
    print(f"frame de treino: {len(frame)} barras de 1min "
          f"({frame.index[0]} -> {frame.index[-1]})\n")

    bundle, report = walk_forward_train(
        frame,
        horizon_bars=int(ml_params.get("horizon_bars", 5)),
        threshold_atr=float(ml_params.get("label_threshold_atr", 0.5)),
        n_folds=args.folds,
    )
    print(report.summary())
    save_bundle(bundle, out)
    print(f"\nmodelo salvo em: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
