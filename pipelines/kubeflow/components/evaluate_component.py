"""KFP evaluation component: compare a candidate with the recorded champion threshold."""
from kfp import dsl


@dsl.component(base_image="sentiment-ml-pipeline:local")
def evaluate_component(
    candidate: dsl.Input[dsl.Artifact],
    champion_val_f1: float,
    min_f1: float,
    decision: dsl.Output[dsl.Artifact],
) -> None:
    """Write a promotion recommendation; serving changes remain deliberately out of scope."""
    import json
    from pathlib import Path

    report = json.loads(Path(candidate.path).read_text(encoding="utf-8"))
    val_f1 = float(report["val_f1"])
    result = {
        **report,
        "champion_val_f1": champion_val_f1,
        "quality_gate_passed": val_f1 >= min_f1,
        "beats_champion": val_f1 > champion_val_f1,
        "eligible_for_bentoml_promotion": val_f1 >= min_f1 and val_f1 > champion_val_f1,
        "serving_action": "none_until_bentoml_health_checks_pass",
    }
    Path(decision.path).write_text(json.dumps(result, indent=2), encoding="utf-8")
