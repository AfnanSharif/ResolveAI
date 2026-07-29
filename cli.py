from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from customer_support_agent import CustomerSupportAgent
from customer_support_agent.feedback import FeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-first RAG customer support agent")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Analyze a support request")
    ask.add_argument("message")
    ask.add_argument("--subject", default="Support request")
    feedback = sub.add_parser("feedback", help="Record thumbs-up/down feedback")
    feedback.add_argument("ticket_id")
    feedback.add_argument("rating", choices=("up", "down"))
    feedback.add_argument("--note", default="")
    sub.add_parser("stats", help="Summarize local feedback")
    deploy = sub.add_parser("azure-deploy", help="Create/update the Azure ML managed endpoint")
    deploy.add_argument("--dry-run", action="store_true")
    deploy.add_argument("--endpoint")
    deploy.add_argument("--deployment")
    deploy.add_argument("--model-name")
    deploy.add_argument("--environment-name")
    deploy.add_argument("--instance-type")
    deploy.add_argument("--instance-count", type=int)
    deploy.add_argument("--traffic", type=int)
    track = sub.add_parser("track-evaluation", help="Log curated evaluation metrics to MLflow/Azure ML")
    track.add_argument("--dry-run", action="store_true")
    track.add_argument("--experiment")
    track.add_argument("--cases", type=Path, default=ROOT / "data" / "evaluation_cases.csv")
    track.add_argument("--run-name", default="offline-policy-evaluation")
    track.add_argument("--tracking-uri")
    args = parser.parse_args()
    store = FeedbackStore(ROOT / "feedback" / "ratings.jsonl")
    if args.command == "ask":
        output = CustomerSupportAgent().answer(args.message, args.subject).to_dict()
    elif args.command == "feedback":
        output = store.add(args.ticket_id, 1 if args.rating == "up" else -1, args.note)
    elif args.command == "stats":
        output = store.summary()
    elif args.command == "azure-deploy":
        from azure.deploy import DeploymentConfig, create_or_update, deployment_plan, workspace_client

        config = DeploymentConfig(
            endpoint_name=args.endpoint or os.environ.get("AZURE_ENDPOINT_NAME", "resolve-ai-endpoint"),
            deployment_name=args.deployment or os.environ.get("AZURE_DEPLOYMENT_NAME", "blue"),
            model_name=args.model_name or os.environ.get("AZURE_MODEL_NAME", "resolve-ai-model"),
            environment_name=args.environment_name or os.environ.get("AZURE_ENVIRONMENT_NAME", "resolve-ai-environment"),
            instance_type=args.instance_type or os.environ.get("AZURE_INSTANCE_TYPE", "Standard_DS3_v2"),
            instance_count=args.instance_count if args.instance_count is not None else int(os.environ.get("AZURE_INSTANCE_COUNT", "1")),
            traffic_percent=args.traffic if args.traffic is not None else int(os.environ.get("AZURE_TRAFFIC_PERCENT", "100")),
        )
        output = deployment_plan(config) if args.dry_run else create_or_update(workspace_client(), config)
    else:
        from azure.track_experiment import log_experiment, tracking_payload

        payload = tracking_payload(ROOT, args.cases)
        if args.dry_run:
            output = payload
        else:
            run_id = log_experiment(
                payload,
                args.experiment or os.environ.get("AZURE_EXPERIMENT_NAME", "resolve-ai-evaluation"),
                args.run_name,
                tracking_uri=args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI"),
            )
            output = {"run_id": run_id, **payload}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
