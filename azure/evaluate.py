from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def evaluate(project: str | Path, cases_path: str | Path) -> dict[str, float | int]:
    root = Path(project).resolve(); sys.path.insert(0, str(root / "src"))
    from customer_support_agent import CustomerSupportAgent
    from customer_support_agent.providers import LocalCandidateProvider
    agent = CustomerSupportAgent(root / "data" / "support_knowledge.csv", provider=LocalCandidateProvider())
    with Path(cases_path).open(encoding="utf-8-sig", newline="") as source: cases = list(csv.DictReader(source))
    correct = escalated = 0
    for case in cases:
        result = agent.answer(case["message"], case["subject"])
        correct += result.category == case["expected_category"]
        escalated += result.human_review == (case["expected_review"].lower() == "true")
    total = len(cases)
    return {"cases": total, "category_accuracy": round(correct / total, 4) if total else 0.0, "review_accuracy": round(escalated / total, 4) if total else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project", default="."); parser.add_argument("--cases", required=True); parser.add_argument("--report", required=True)
    args = parser.parse_args(); result = evaluate(args.project, args.cases)
    destination = Path(args.report); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
