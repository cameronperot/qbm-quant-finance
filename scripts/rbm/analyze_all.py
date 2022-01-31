import json
import subprocess

from qbm.utils import get_project_dir, load_artifact

from analyze_ensemble import main as analyze_ensemble
from analyze_autocorrelation import main as analyze_autocorrelation


def main(analyze_ensemble_bool, analyze_autocorrelation_bool):
    project_dir = get_project_dir()
    models = load_artifact(project_dir / "scripts/rbm/models.json")

    for model_name, model_info in models.items():
        if analyze_ensemble_bool:
            print(model_info)
            analyze_ensemble(model_info["id"])
        if analyze_autocorrelation_bool:
            analyze_autocorrelation(model_info["id"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze the model outputs.")
    parser.add_argument(
        "--no_analyze_autocorrelation",
        default=False,
        action="store_true",
        help="Do you want to analyze the autocorrelations?",
    )
    parser.add_argument(
        "--no_analyze_ensemble",
        default=False,
        action="store_true",
        help="Do you want to analyze the ensemble?",
    )
    args = parser.parse_args()
    analyze_ensemble_bool = not args.no_analyze_ensemble
    analyze_autocorrelation_bool = not args.no_analyze_autocorrelation

    main(analyze_ensemble_bool, analyze_autocorrelation_bool)
