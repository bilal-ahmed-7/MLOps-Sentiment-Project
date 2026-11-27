"""Small runtime helpers shared by KFP component containers."""
import os
import subprocess


def configure_dvc_from_env() -> None:
    """Configure the DVC remote from task-secret environment variables, if supplied."""
    user, token = os.getenv("DAGSHUB_USER"), os.getenv("DAGSHUB_TOKEN")
    if not user and not token:
        return
    if not user or not token:
        raise RuntimeError("DAGSHUB_USER and DAGSHUB_TOKEN must both be provided to use the DVC remote.")
    for args in (
        ("remote", "modify", "origin", "--local", "auth", "basic"),
        ("remote", "modify", "origin", "--local", "user", user),
        ("remote", "modify", "origin", "--local", "password", token),
    ):
        subprocess.run(["dvc", *args], check=True)
