from __future__ import annotations

import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from customer_support_agent import CustomerSupportAgent
from customer_support_agent.feedback import FeedbackStore
from customer_support_agent.providers import LocalCandidateProvider
from customer_support_agent.knowledge import HashingEmbedder
from azure.evaluate import evaluate
from azure.validate_knowledge import validate
from azure.deploy import DeploymentConfig, deployment_plan, traffic_plan
from azure.score import init as score_init, run as score_run
from azure.track_experiment import log_experiment, tracking_payload
import json


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = CustomerSupportAgent(provider=LocalCandidateProvider())

    def test_shipping_response_is_grounded(self) -> None:
        result = self.agent.answer("Tracking has not moved for four days", "Late parcel")
        self.assertEqual(result.category, "shipping")
        self.assertIn("SHP-201", result.citations)
        self.assertEqual(len(result.candidates), 3)

    def test_security_ticket_needs_review(self) -> None:
        result = self.agent.answer("I think my account was stolen in a breach")
        self.assertEqual(result.urgency, "urgent")
        self.assertTrue(result.human_review)

    def test_short_message_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.answer("no")

    def test_feedback_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.jsonl")
            store.add("TKT-1", 1)
            store.add("TKT-2", -1)
            self.assertEqual(store.summary()["satisfaction"], 0.5)

    def test_azure_pipeline_steps_run_offline(self) -> None:
        validation = validate(ROOT / "data" / "support_knowledge.csv")
        metrics = evaluate(ROOT, ROOT / "data" / "evaluation_cases.csv")
        self.assertTrue(validation["valid"])
        self.assertGreaterEqual(metrics["category_accuracy"], 0.75)
        self.assertEqual(metrics["review_accuracy"], 1.0)

    def test_hash_embeddings_are_deterministic(self) -> None:
        vectors = HashingEmbedder(64).embed(["return damaged item", "return damaged item"])
        self.assertEqual(vectors[0], vectors[1])
        self.assertAlmostEqual(sum(value * value for value in vectors[0]), 1.0)

    def test_azure_endpoint_assets_and_scoring_contract(self) -> None:
        plan = deployment_plan(DeploymentConfig(), ROOT)
        self.assertEqual(plan["scoring_script"], "azure/score.py")
        score_init()
        response = json.loads(score_run({"subject": "Late parcel", "message": "Tracking has not moved for four days"}))
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["category"], "shipping")

    def test_azure_endpoint_names_are_validated_before_authentication(self) -> None:
        with self.assertRaises(ValueError):
            deployment_plan(DeploymentConfig(endpoint_name="invalid endpoint"), ROOT)

    def test_rollout_traffic_preserves_existing_deployments(self) -> None:
        self.assertEqual(traffic_plan({"green": 70, "canary": 30}, "blue", 20), {"green": 56, "canary": 24, "blue": 20})
        with self.assertRaises(ValueError):
            traffic_plan({}, "blue", 20)

    def test_tracking_payload_contains_curated_metrics(self) -> None:
        payload = tracking_payload(ROOT, ROOT / "data" / "evaluation_cases.csv")
        self.assertGreaterEqual(payload["metrics"]["category_accuracy"], 0.75)
        self.assertEqual(payload["params"]["provider"], "local")

    def test_mlflow_logging_boundary_records_the_evaluation(self) -> None:
        calls: dict[str, object] = {}

        class Run:
            info = types.SimpleNamespace(run_id="run-123")
            def __enter__(self): return self
            def __exit__(self, *_): return False

        fake_mlflow = types.SimpleNamespace(
            set_tracking_uri=lambda value: calls.update(tracking_uri=value),
            set_experiment=lambda value: calls.update(experiment=value),
            start_run=lambda run_name: calls.update(run_name=run_name) or Run(),
            log_params=lambda value: calls.update(params=value),
            log_metrics=lambda value: calls.update(metrics=value),
            log_artifact=lambda path, artifact_path: calls.update(artifact=(Path(path).name, artifact_path)),
            set_tags=lambda value: calls.update(tags=value),
        )
        payload = {"params": {"provider": "local"}, "metrics": {"accuracy": 1.0}}
        with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            run_id = log_experiment(payload, "resolve-test", "offline-test", tracking_uri="file:///tmp/mlruns")
        self.assertEqual(run_id, "run-123")
        self.assertEqual(calls["experiment"], "resolve-test")
        self.assertEqual(calls["artifact"], ("evaluation.json", "evaluation"))


if __name__ == "__main__":
    unittest.main()
