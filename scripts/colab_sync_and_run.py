#!/usr/bin/env python3
"""
Google Colab & Hugging Face Hub End-to-End Orchestrator (`scripts/colab_sync_and_run.py`)
for the Google Mumbai Air Quality Forecasting Challenge (Zindi).

Capabilities:
1. `--stage-hf`: Stages `data/raw/`, `data/external/`, and `data/processed/` to a Hugging Face
   Dataset repository (`hf_hub` / `huggingface-cli upload`) so remote Google Colab GPU instances
   can pull the complete pre-cached data snapshot in <15 seconds.
2. `--colab-run`: Uses the official `google-colab-cli` (`colab new`, `colab exec`, `colab download`)
   to provision a remote Colab NVIDIA T4/L4 GPU runtime, execute `notebooks/Mumbai_Winning_Colab.ipynb`
   end-to-end, and download the resulting `submission.csv` and executed `.ipynb` artifact.
3. `--local-exec`: Executes `notebooks/Mumbai_Winning_Colab.ipynb` in-place via `jupyter nbconvert`
   and runs all 7 strict Zindi submission assertions.
4. `--submit`: Uploads the verified `submissions/submission_colab_verified.csv` directly to Zindi
   via the official `zindi` SDK (`from zindi.user import Zindian`).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "Mumbai_Winning_Colab.ipynb"
SAMPLE_SUB_PATH = PROJECT_ROOT / "data" / "raw" / "SampleSubmission.csv"
VERIFIED_SUB_PATH = PROJECT_ROOT / "submissions" / "submission_colab_verified.csv"
DEFAULT_HF_REPO = os.environ.get(
    "HF_DATASET_REPO", "han-jisso/google-mumbai-air-quality-challenge"
)
DEFAULT_ZINDI_ID = os.environ.get(
    "ZINDI_CHALLENGE_ID", "google-mumbai-air-quality-challenge"
)


def verify_submission_file(sub_path: Path, sample_sub_path: Path = SAMPLE_SUB_PATH) -> None:
    """Run all 7 strict Zindi competition assertions on a generated submission CSV."""
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file not found: {sub_path}")
    sub_df = pd.read_csv(sub_path, dtype={"ID": str})
    sample_df = pd.read_csv(sample_sub_path, dtype={"ID": str})

    assert len(sub_df) == 2592, f"[Assertion 1 Failed] Expected 2592 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["ID", "Target"], (
        f"[Assertion 2 Failed] Expected ['ID', 'Target'], got {list(sub_df.columns)}"
    )
    assert sub_df["ID"].notna().all(), "[Assertion 3 Failed] Found NaN values in ID column"
    assert sub_df["ID"].is_unique, "[Assertion 4 Failed] Duplicate IDs detected"
    assert sub_df["ID"].tolist() == sample_df["ID"].tolist(), (
        "[Assertion 5 Failed] ID order does not exactly match SampleSubmission.csv"
    )
    assert sub_df["Target"].notna().all() and np.isfinite(sub_df["Target"].values).all(), (
        "[Assertion 6 Failed] Non-finite or NaN predictions detected in Target"
    )
    assert (sub_df["Target"].values >= 0.0).all(), (
        "[Assertion 7 Failed] Negative PM2.5 predictions detected in Target"
    )
    print(
        f"[VERIFIED] {sub_path.name} passed all 7 strict Zindi assertions "
        f"(rows={len(sub_df):,}, mean={sub_df['Target'].mean():.2f} ug/m3, "
        f"min={sub_df['Target'].min():.2f}, max={sub_df['Target'].max():.2f})."
    )


def stage_data_to_huggingface(repo_id: str = DEFAULT_HF_REPO) -> None:
    """
    Stage `data/raw/`, `data/external/`, and `data/processed/` to Hugging Face Hub
    using `huggingface_hub.HfApi().upload_folder` so Colab runtimes can fetch the snapshot immediately.
    """
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    print(f"[HF Stage] Ensuring dataset repository '{repo_id}' exists...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    for rel_dir in ["data/raw", "data/external", "data/processed"]:
        folder_path = PROJECT_ROOT / rel_dir
        if not folder_path.exists():
            continue
        print(f"[HF Stage] Uploading {folder_path} -> hf://datasets/{repo_id}/{rel_dir} ...")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(folder_path),
            path_in_repo=rel_dir if rel_dir != "data/raw" else ".",
            ignore_patterns=[".cache/**", "*.lock", "*.metadata", "__pycache__/**"],
        )
    print(f"[HF Stage] Successfully staged dataset snapshot to https://huggingface.co/datasets/{repo_id}")


def execute_notebook_locally(notebook_path: Path = NOTEBOOK_PATH) -> None:
    """Execute `notebooks/Mumbai_Winning_Colab.ipynb` in-place via `jupyter nbconvert`."""
    jupyter_bin = PROJECT_ROOT / ".venv" / "bin" / "jupyter"
    cmd_prefix = [str(jupyter_bin)] if jupyter_bin.exists() else [sys.executable, "-m", "jupyter"]
    cmd = [
        *cmd_prefix,
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=1800",
        str(notebook_path),
    ]
    print(f"[Local Exec] Executing standalone notebook: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    verify_submission_file(VERIFIED_SUB_PATH)


def run_on_remote_colab(
    notebook_path: Path = NOTEBOOK_PATH,
    gpu_type: str = "T4",
) -> None:
    """
    Orchestrate remote Google Colab GPU execution via `google-colab-cli` (`colab`).
    Falls back to instructions if `colab` CLI binary is not authenticated on PATH.
    """
    colab_bin = shutil.which("colab")
    if colab_bin is None:
        print(
            "[Colab CLI] `colab` binary not found on PATH.\n"
            "Install and authenticate `google-colab-cli` via:\n"
            "  npm install -g @googlecolab/colab-cli   # or: uv tool install google-colab-cli\n"
            "  colab auth login\n"
            "Then re-run:\n"
            f"  python scripts/colab_sync_and_run.py --colab-run --gpu {gpu_type}"
        )
        return

    print(f"[Colab CLI] Provisioning remote Google Colab {gpu_type} GPU session...")
    subprocess.run([colab_bin, "new", "--gpu", gpu_type, "--name", "Mumbai_Winning_Colab"], check=True)
    print(f"[Colab CLI] Executing {notebook_path.name} on remote {gpu_type} GPU...")
    subprocess.run(
        [colab_bin, "exec", str(notebook_path), "--output", str(notebook_path)],
        check=True,
    )
    print("[Colab CLI] Downloading remote submission artifacts...")
    subprocess.run(
        [colab_bin, "download", "submissions/submission_colab_verified.csv", str(VERIFIED_SUB_PATH)],
        check=True,
    )
    verify_submission_file(VERIFIED_SUB_PATH)


def submit_to_zindi(
    sub_path: Path = VERIFIED_SUB_PATH,
    comment: str = "Standalone Colab Reproducible 8-Model Physics + TimesFM 3.0 + Chronos-2 + GBDT + ST-GAT Ensemble",
) -> bool:
    """Verify and submit `submission_colab_verified.csv` directly to Zindi via `zindi` SDK."""
    verify_submission_file(sub_path)
    from zindi.user import Zindian

    username = os.environ.get("ZINDI_USERNAME", "").strip()
    password = os.environ.get("ZINDI_PASSWORD", "").strip()
    challenge_id = os.environ.get("ZINDI_CHALLENGE_ID", DEFAULT_ZINDI_ID).strip()

    if not username or not password:
        print(
            "[Zindi Submit] ZINDI_USERNAME / ZINDI_PASSWORD not set in environment.\n"
            "Export your credentials and re-run:\n"
            "  export ZINDI_USERNAME='your_zindi_username'\n"
            "  export ZINDI_PASSWORD='your_zindi_password'\n"
            "  python scripts/colab_sync_and_run.py --submit"
        )
        return False

    print(f"[Zindi Submit] Authenticating as '{username}' for '{challenge_id}'...")
    user = Zindian(username=username, fixed_password=password)
    user.select_a_challenge(fixed_index=challenge_id)
    user.submit(filepaths=[str(sub_path)], comments=[comment])
    print(f"[Zindi Submit] Successfully submitted {sub_path} to Zindi!")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage data to Hugging Face, execute Mumbai_Winning_Colab.ipynb locally or on Colab GPU, and submit to Zindi."
    )
    parser.add_argument(
        "--stage-hf",
        action="store_true",
        help="Upload data/raw, data/external, and data/processed to Hugging Face Hub dataset repo.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help=f"Target Hugging Face dataset repo ID (default: {DEFAULT_HF_REPO}).",
    )
    parser.add_argument(
        "--local-exec",
        action="store_true",
        help="Execute notebooks/Mumbai_Winning_Colab.ipynb locally in-place via jupyter nbconvert.",
    )
    parser.add_argument(
        "--colab-run",
        action="store_true",
        help="Execute notebooks/Mumbai_Winning_Colab.ipynb on a remote Google Colab GPU via google-colab-cli.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="T4",
        choices=["T4", "L4", "A100"],
        help="Remote Colab GPU accelerator type (default: T4).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run the 7 strict Zindi submission assertions on submissions/submission_colab_verified.csv.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Upload verified submission_colab_verified.csv to Zindi using the zindi Python SDK.",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default="Standalone Colab Reproducible 8-Model Physics + TimesFM 3.0 + Chronos-2 + GBDT + ST-GAT Ensemble",
        help="Submission comment for Zindi leaderboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not any([args.stage_hf, args.local_exec, args.colab_run, args.verify_only, args.submit]):
        print(
            "No action flag provided. Running default verification on existing submission artifact "
            "(use --help to see --stage-hf, --local-exec, --colab-run, --submit)."
        )
        if VERIFIED_SUB_PATH.exists():
            verify_submission_file(VERIFIED_SUB_PATH)
        return

    if args.stage_hf:
        stage_data_to_huggingface(repo_id=args.hf_repo)
    if args.local_exec:
        execute_notebook_locally(NOTEBOOK_PATH)
    if args.colab_run:
        run_on_remote_colab(NOTEBOOK_PATH, gpu_type=args.gpu)
    if args.verify_only:
        verify_submission_file(VERIFIED_SUB_PATH)
    if args.submit:
        submit_to_zindi(VERIFIED_SUB_PATH, comment=args.comment)


if __name__ == "__main__":
    main()
