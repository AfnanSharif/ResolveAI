"""Create or update a managed Azure ML endpoint for Resolve AI.

The module keeps SDK imports inside the deployment boundary so the local agent
and its tests remain dependency-free.  ``--dry-run`` validates and prints the
exact asset plan without authenticating or creating billable resources.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DeploymentConfig:
    endpoint_name: str = "resolve-ai-endpoint"
    deployment_name: str = "blue"
    model_name: str = "resolve-ai-model"
    environment_name: str = "resolve-ai-environment"
    instance_type: str = "Standard_DS3_v2"
    instance_count: int = 1
    traffic_percent: int = 100

    def validate(self) -> "DeploymentConfig":
        online_name = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,31}")
        if any(not online_name.fullmatch(value) or "--" in value for value in (self.endpoint_name, self.deployment_name)):
            raise ValueError("Azure endpoint/deployment names must be 3-32 letters, numbers, or single dashes and start with a letter")
        if any(not value.strip() or len(value) > 255 for value in (self.model_name, self.environment_name)):
            raise ValueError("Azure model/environment names must contain between 1 and 255 characters")
        if self.instance_count < 1:
            raise ValueError("instance_count must be positive")
        if not 0 <= self.traffic_percent <= 100:
            raise ValueError("traffic_percent must be between 0 and 100")
        if not self.instance_type.strip():
            raise ValueError("instance_type is required")
        return self


def deployment_plan(config: DeploymentConfig, project_root: str | Path = ROOT) -> dict[str, Any]:
    """Return a serializable, testable description of the Azure assets."""
    config.validate()
    root = Path(project_root).resolve()
    required = (root / "src", root / "data" / "support_knowledge.csv", root / "azure" / "score.py", root / "azure" / "conda.yml", root / "azure" / "sample-request.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError("Deployment assets are missing: " + ", ".join(missing))
    return {
        **asdict(config),
        "project_root": str(root),
        "scoring_script": "azure/score.py",
        "conda_file": str(root / "azure" / "conda.yml"),
        "knowledge_file": str(root / "data" / "support_knowledge.csv"),
        "sample_request": str(root / "azure" / "sample-request.json"),
    }


def traffic_plan(existing: dict[str, int] | None, deployment_name: str, traffic_percent: int) -> dict[str, int]:
    """Preserve and proportionally rebalance existing deployments for a rollout."""
    if not 0 <= traffic_percent <= 100:
        raise ValueError("traffic_percent must be between 0 and 100")
    if traffic_percent == 100:
        return {deployment_name: 100}
    others = {
        str(name): int(percent)
        for name, percent in (existing or {}).items()
        if name != deployment_name and int(percent) > 0
    }
    if not others:
        raise ValueError("traffic below 100% requires an existing alternate deployment")
    remaining = 100 - traffic_percent
    total = sum(others.values())
    exact = {name: remaining * percent / total for name, percent in others.items()}
    allocations = {name: math.floor(percent) for name, percent in exact.items()}
    shortfall = remaining - sum(allocations.values())
    ranked = sorted(others, key=lambda name: (-(exact[name] - allocations[name]), name))
    for name in ranked[:shortfall]:
        allocations[name] += 1
    allocations[deployment_name] = traffic_percent
    return allocations


def workspace_client():
    """Authenticate with DefaultAzureCredential and return an MLClient."""
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise RuntimeError("Install azure-ai-ml and azure-identity for deployment") from exc
    required = ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ML_WORKSPACE")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Missing Azure settings: " + ", ".join(missing))
    return MLClient(DefaultAzureCredential(), os.environ[required[0]], os.environ[required[1]], os.environ[required[2]])


def create_or_update(client: Any, config: DeploymentConfig, project_root: str | Path = ROOT) -> dict[str, Any]:
    """Register assets, deploy code, and atomically direct endpoint traffic.

    Azure SDK objects are created here instead of at import time. This makes the
    deployment callable from CI, a workstation, or an Azure ML command job.
    """
    plan = deployment_plan(config, project_root)
    root = Path(plan["project_root"])
    try:
        from azure.core.exceptions import ResourceNotFoundError
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import CodeConfiguration, Environment, ManagedOnlineDeployment, ManagedOnlineEndpoint, Model
    except ImportError as exc:
        raise RuntimeError("Install azure-ai-ml before deploying") from exc

    try:
        endpoint = client.online_endpoints.get(name=config.endpoint_name)
    except ResourceNotFoundError:
        endpoint = None
    existing_traffic = dict(getattr(endpoint, "traffic", {}) or {})
    planned_traffic = traffic_plan(existing_traffic, config.deployment_name, config.traffic_percent)

    model = client.models.create_or_update(Model(
        path=str(root),
        type=AssetTypes.CUSTOM_MODEL,
        name=config.model_name,
        description="Resolve AI source, support policy, and endpoint scorer",
    ))
    environment = client.environments.create_or_update(Environment(
        name=config.environment_name,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(root / "azure" / "conda.yml"),
        description="Pinned runtime contract for the Resolve AI endpoint",
    ))
    if endpoint is None:
        endpoint = ManagedOnlineEndpoint(
            name=config.endpoint_name,
            auth_mode="key",
            description="Evidence-grounded customer-support response endpoint",
        )
        client.online_endpoints.begin_create_or_update(endpoint).result()
    deployment = ManagedOnlineDeployment(
        name=config.deployment_name,
        endpoint_name=config.endpoint_name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(code=str(root), scoring_script="azure/score.py"),
        instance_type=config.instance_type,
        instance_count=config.instance_count,
    )
    client.online_deployments.begin_create_or_update(deployment).result()
    endpoint.traffic = planned_traffic
    client.online_endpoints.begin_create_or_update(endpoint).result()
    return {
        "endpoint": config.endpoint_name,
        "deployment": config.deployment_name,
        "model": getattr(model, "id", config.model_name),
        "environment": getattr(environment, "id", config.environment_name),
        "traffic": planned_traffic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or deploy the Resolve AI Azure ML endpoint")
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_ENDPOINT_NAME", "resolve-ai-endpoint"))
    parser.add_argument("--deployment", default=os.environ.get("AZURE_DEPLOYMENT_NAME", "blue"))
    parser.add_argument("--model-name", default=os.environ.get("AZURE_MODEL_NAME", "resolve-ai-model"))
    parser.add_argument("--environment-name", default=os.environ.get("AZURE_ENVIRONMENT_NAME", "resolve-ai-environment"))
    parser.add_argument("--instance-type", default=os.environ.get("AZURE_INSTANCE_TYPE", "Standard_DS3_v2"))
    parser.add_argument("--instance-count", type=int, default=int(os.environ.get("AZURE_INSTANCE_COUNT", "1")))
    parser.add_argument("--traffic", type=int, default=int(os.environ.get("AZURE_TRAFFIC_PERCENT", "100")))
    parser.add_argument("--dry-run", action="store_true", help="validate assets and print the plan without authenticating")
    args = parser.parse_args()
    config = DeploymentConfig(args.endpoint, args.deployment, args.model_name, args.environment_name, args.instance_type, args.instance_count, args.traffic)
    if args.dry_run:
        print(json.dumps(deployment_plan(config), indent=2))
        return
    result = create_or_update(workspace_client(), config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
