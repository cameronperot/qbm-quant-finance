import json
import subprocess

from qbm.utils import get_project_dir, load_artifact

from analyze_ensemble import main as analyze_ensemble
from analyze_autocorrelation import main as analyze_autocorrelation


def main(analyze_ensemble, analyze_autocorrelation):
    project_dir = get_project_dir()
    models = load_artifact(project_dir / "scripts/rbm/models.json")

    for model_name, model_info in models.items():
        if analyze_ensemble:
            analyze_ensemble(model_info["id"])
        if analyze_autocorrelation:
            analyze_autocorrelation(model_info["id"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze the model outputs.")
    parser.add_argument(
        "--analyze_autocorrelation",
        type=bool,
        default=True,
        help="Do you want to analyze the autocorrelations?",
    )
    parser.add_argument(
        "--analyze_ensemble",
        type=bool,
        default=True,
        help="Do you want to analyze the ensemble?",
    )
    args = parser.parse_args()

    main(analyze_ensemble, analyze_autocorrelation)
