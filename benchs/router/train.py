"""
CLI entry point for training and evaluating the ANN router.

Usage:
    python -m benchs.router.train --results-dir benchs/results_router
    python -m benchs.router.train --results-dir benchs/results_router --evaluate
    python -m benchs.router.train --results-dir benchs/results_router \\
        --output my_router.pkl --recall-targets 0.8 0.9 0.95

After training, test inference with:
    python -m benchs.router.train --load my_router.pkl --predict-demo
"""

import argparse
import sys

from .router import ANNRouter
from .evaluate import lodo_cv
from .data_loader import RECALL_TARGETS


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchs.router.train",
        description="Train and evaluate the ANN index router.",
    )
    p.add_argument(
        "--results-dir", default="benchs/results_router",
        help="Directory containing results_<dataset>.json files.",
    )
    p.add_argument(
        "--output", default="router_model.pkl",
        help="Path to save the trained ANNRouter (joblib).",
    )
    p.add_argument(
        "--recall-targets", nargs="+", type=float, default=None,
        metavar="R",
        help=f"Recall thresholds to train on (default: {RECALL_TARGETS}).",
    )
    p.add_argument(
        "--k", type=int, default=20,
        help="Recall@k to use from the benchmark curves (default: 20).",
    )
    p.add_argument(
        "--evaluate", action="store_true",
        help="Run LODO cross-validation and print metrics.",
    )
    p.add_argument(
        "--evaluate-only", action="store_true",
        help="Run LODO CV without saving a model.",
    )
    p.add_argument(
        "--load", default=None,
        help="Load an existing router instead of training.",
    )
    p.add_argument(
        "--predict-demo", action="store_true",
        help="Run a quick inference demo using hard-coded SIFT-1M meta-features.",
    )
    return p


def _demo_predict(router: ANNRouter) -> None:
    print("\n--- Inference demo (SIFT-1M @ recall=0.90) ---")
    result = router.predict(
        meta_features={
            "n":                         1_000_000,
            "d":                         128,
            "lid_mle":                   18.88,
            "pdist_mean":                538.4,
            "pdist_std":                 106.9,
            "kmeans_inertia_ratio_16":   0.513,
        },
        target_recall=0.90,
    )
    print(f"  Selected index  : {result['index']}")
    print(f"  Recommended {result['param_name']:20s}: {result['param']:.5g}")
    print(f"  Predicted QPS   : {result['predicted_qps']:.1f}")

    print("\n--- Full ranking @ recall=0.90 ---")
    ranking = router.predict_ranking(
        meta_features={
            "n":                         1_000_000,
            "d":                         128,
            "lid_mle":                   18.88,
            "pdist_mean":                538.4,
            "pdist_std":                 106.9,
            "kmeans_inertia_ratio_16":   0.513,
        },
        target_recall=0.90,
    )
    print(ranking.to_string(index=False))

    print("\n--- Feature importances (index selector) ---")
    fi = router.feature_importances
    print(fi.to_string())


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    if args.load:
        print(f"Loading router from {args.load} ...")
        router = ANNRouter.load(args.load)
    elif not args.evaluate_only:
        print(f"Training router on {args.results_dir} ...")
        router = ANNRouter()
        router.fit(
            args.results_dir,
            recall_targets=args.recall_targets,
            k=args.k,
        )
        print(f"Saving router to {args.output} ...")
        router.save(args.output)
        print(f"Candidate indexes: {router.candidate_indexes}")
    else:
        router = None

    if args.predict_demo and router is not None:
        _demo_predict(router)

    if args.evaluate or args.evaluate_only:
        print(f"\nRunning LODO CV on {args.results_dir} ...")
        lodo_cv(
            args.results_dir,
            recall_targets=args.recall_targets,
            k=args.k,
            verbose=True,
        )


if __name__ == "__main__":
    main()
