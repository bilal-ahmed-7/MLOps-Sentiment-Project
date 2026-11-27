"""KFP preprocessing component: DVC pull, validate, split, and publish a split artifact."""
from kfp import dsl


@dsl.component(base_image="sentiment-ml-pipeline:local")
def preprocess_component(splits: dsl.Output[dsl.Dataset]) -> None:
    """Create deterministic splits and pass them to later KFP tasks as one tar artifact."""
    import shutil
    import subprocess
    import tarfile
    from pathlib import Path

    from pipelines.kubeflow.components.common import configure_dvc_from_env

    configure_dvc_from_env()
    subprocess.run(["dvc", "pull", "data/raw/imdb_reviews.csv"], check=True)
    subprocess.run(["python", "-m", "src.features.preprocessing"], check=True)
    processed = Path("data/processed")
    if not all((processed / f"{name}.csv").exists() for name in ("train", "val", "test")):
        raise RuntimeError("Preprocessing did not produce all required split files.")
    with tarfile.open(splits.path, "w:gz") as archive:
        archive.add(processed, arcname="processed")
    shutil.rmtree(processed)
