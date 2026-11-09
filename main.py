"""Command-line entry point.

Usage:
    python main.py generate [--n 1500]   # create data/students.csv (synthetic)
    python main.py train [--fast]        # train + evaluate + save models
    python main.py analyze               # EDA figures, feature importance, insights
    python main.py predict               # predict for a demo student
    python main.py predict --csv new_students.csv --out predictions.csv
    python main.py all                   # generate -> train -> analyze -> predict demo
"""
import argparse
import json

import pandas as pd

from src import config
from src.analysis import run_eda, run_feature_importance, write_insights
from src.data_generator import save_dataset
from src.predict import DEMO_STUDENT, StudentPredictor
from src.preprocessing import load_data
from src.train import run_training


def cmd_generate(args):
    df = save_dataset(n=args.n)
    print(f"Saved {len(df)} students to {config.DATA_FILE}")


def cmd_train(args):
    metrics = run_training(fast=args.fast)
    c, r = metrics["classification"], metrics["regression"]
    print(f"Classifier: {c['best_model']}  accuracy={c['accuracy']:.3f}  "
          f"macro-F1={c['f1_macro']:.3f}")
    print(f"Regressor : {r['best_model']}  MAE={r['mae']:.2f}  "
          f"RMSE={r['rmse']:.2f}  R2={r['r2']:.3f}")
    print(f"Models saved to {config.MODELS_DIR}, metrics to {config.REPORTS_DIR / 'metrics.json'}")


def cmd_analyze(_args):
    df = load_data()
    run_eda(df)
    imp = run_feature_importance(df)
    write_insights(df, imp)
    print(f"Figures and reports written to {config.REPORTS_DIR}")
    print(imp.head(5).to_string(index=False))


def cmd_predict(args):
    predictor = StudentPredictor()
    if args.csv:
        out = predictor.predict_df(pd.read_csv(args.csv))
        out.to_csv(args.out, index=False)
        print(f"Wrote {len(out)} predictions to {args.out}")
        print(out["predicted_level"].value_counts().to_string())
    else:
        result = predictor.predict(DEMO_STUDENT)
        print(json.dumps(result, indent=2))


def cmd_all(args):
    cmd_generate(args)
    cmd_train(args)
    cmd_analyze(args)
    cmd_predict(argparse.Namespace(csv=None, out=None))


def main():
    p = argparse.ArgumentParser(description="Smart Student Success Prediction System")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate synthetic dataset")
    g.add_argument("--n", type=int, default=1500)
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="train and evaluate models")
    t.add_argument("--fast", action="store_true", help="smaller models, quicker run")
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("analyze", help="EDA and feature importance")
    a.set_defaults(func=cmd_analyze)

    pr = sub.add_parser("predict", help="predict for demo student or a CSV")
    pr.add_argument("--csv", help="CSV of students with the feature columns")
    pr.add_argument("--out", default="predictions.csv")
    pr.set_defaults(func=cmd_predict)

    al = sub.add_parser("all", help="run the full pipeline")
    al.add_argument("--n", type=int, default=1500)
    al.add_argument("--fast", action="store_true")
    al.set_defaults(func=cmd_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
