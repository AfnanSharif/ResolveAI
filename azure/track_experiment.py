"""Log the offline support evaluation as an MLflow/Azure ML experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from azure.evaluate import evaluate


def tracking_payload(project: str | Path, cases: str | Path) -> dict[str, Any]:
    """Build deterministic metrics before any tracking service is contacted."""
    metrics = evaluate(project, cases)
    return {
        "params": {
            "provider": "local",
            "retriever": "bm25",
            "cases_file": Path(cases).name,
        },
        "metrics": {
            "cases": int(metrics["cases"]),
            "category_accuracy": float(metrics["category_accuracy"]),
            "review_accuracy": float(metrics["review_accuracy"]),
        },
    }


def configure_azure_tracking() -> str | None:
    """Point MLflow at Azure ML when workspace settings are present."""
    required = ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ML_WORKSPACE")
    if not all(os.environ.get(name) for name in required):
        return None
    from azure.deploy import workspace_client

    workspace = workspace_client().workspaces.get(os.environ["AZURE_ML_WORKSPACE"])
    return workspace.mlflow_tracking_uri


def log_experiment(payload: dict[str, Any], experiment: str, run_name: str, *, tracking_uri: str | None = None) -> str:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("Install mlflow and azureml-mlflow to track experiments") from exc
    uri = tracking_uri or configure_azure_tracking()
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "evaluation.json"
        report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(payload["params"])
            mlflow.log_metrics(payload["metrics"])
            mlflow.log_artifact(str(report), artifact_path="evaluation")
            mlflow.set_tags({"application": "resolve-ai", "evaluation_mode": "offline-curated"})
            return str(run.info.run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Resolve AI and log metrics to MLflow/Azure ML")
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "evaluation_cases.csv")
    parser.add_argument("--experiment", default=os.environ.get("AZURE_EXPERIMENT_NAME", "resolve-ai-evaluation"))
    parser.add_argument("--run-name", default="offline-policy-evaluation")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = tracking_payload(args.project, args.cases)
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    run_id = log_experiment(payload, args.experiment, args.run_name, tracking_uri=args.tracking_uri)
    print(json.dumps({"run_id": run_id, **payload}, indent=2))


if __name__ == "__main__":
    main()

