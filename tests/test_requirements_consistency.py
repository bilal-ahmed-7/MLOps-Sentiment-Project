"""Guard: packages that affect the saved model are pinned identically everywhere.

The model file is a joblib/pickle, so training, CI and the API image must use the same
scikit-learn, numpy, scipy and joblib versions.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = ("requirements.txt", "requirements-ci.txt", "requirements-api.txt")
MODEL_PACKAGES = ("scikit-learn", "numpy", "scipy", "joblib")
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s#;]+)")


def read_pins(name: str) -> dict[str, str]:
    pins = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        match = PIN.match(line.strip())
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


@pytest.mark.parametrize("package", MODEL_PACKAGES)
def test_model_affecting_package_is_pinned_identically(package):
    versions = {name: read_pins(name).get(package) for name in FILES}
    assert all(versions.values()), f"{package} must be pinned (==) in every requirements file: {versions}"
    assert len(set(versions.values())) == 1, f"{package} versions differ between files: {versions}"
