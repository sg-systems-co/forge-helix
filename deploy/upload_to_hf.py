#!/usr/bin/env python3
"""Upload Falcon-H1-7B-FORGE-v2 to the Hugging Face Hub.

Publishing is public and effectively irreversible -- a pushed commit stays in
the repo history even after deletion, and the weights may be mirrored within
minutes. So this script does nothing without an explicit --confirm, and prints
exactly what it would send first.

    python deploy/upload_to_hf.py --dry-run          # show the plan
    python deploy/upload_to_hf.py --confirm          # actually upload

Auth comes from `huggingface-cli login` or $HF_TOKEN. The token needs write
access to the target namespace.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# HF repo ids are case-sensitive: "SG-Systems" is a 404, the org's canonical id
# is "sgsystems". Kept as a flag so a personal namespace can be targeted too.
DEFAULT_REPO = "sgsystems/Falcon-H1-7B-FORGE-v2"
# FORGE writes artifacts to its own checkout's out/ directory. This repo is the
# overview, not a monorepo, so the default assumes forge is a sibling checkout.
DEFAULT_ARTIFACT_DIR = Path(
    os.environ.get("FORGE_OUT_DIR",
                   Path(__file__).resolve().parents[2] / "forge" / "out")
)
MODEL_FILE = "falcon-h1-7b-forge-v2.gguf"
SIDECAR_FILE = "falcon-h1-7b-forge-v2.forge.json"
MODEL_CARD = Path(__file__).with_name("MODEL_CARD.md")

# Sanity bound: the v2 build is 3.48 GB. Anything far off that is a wrong or
# truncated file, and uploading it would be worse than failing.
EXPECTED_GGUF_BYTES = 3_480_192_416
SIZE_TOLERANCE = 0.02


def human(n: int) -> str:
    return f"{n / 1_000_000_000:.2f} GB" if n >= 1_000_000_000 else f"{n / 1_000_000:.1f} MB"


def check_artifacts(artifact_dir: Path) -> list[tuple[Path, str]]:
    """Validate before touching the network. Returns [(local_path, repo_path)]."""
    gguf = artifact_dir / MODEL_FILE
    sidecar = artifact_dir / SIDECAR_FILE

    problems = []
    if not gguf.is_file():
        problems.append(f"missing model: {gguf}")
    if not sidecar.is_file():
        problems.append(f"missing sidecar: {sidecar}")
    if not MODEL_CARD.is_file():
        problems.append(f"missing model card: {MODEL_CARD}")

    if gguf.is_file():
        with gguf.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                problems.append(f"{gguf.name} does not start with the GGUF magic")
        size = gguf.stat().st_size
        drift = abs(size - EXPECTED_GGUF_BYTES) / EXPECTED_GGUF_BYTES
        if drift > SIZE_TOLERANCE:
            problems.append(
                f"{gguf.name} is {human(size)}, expected ~{human(EXPECTED_GGUF_BYTES)} "
                f"({drift:.0%} off) -- wrong or truncated build?"
            )

    if problems:
        for p in problems:
            print(f"  ERROR: {p}", file=sys.stderr)
        sys.exit(1)

    return [
        (gguf, MODEL_FILE),
        (sidecar, SIDECAR_FILE),
        (MODEL_CARD, "README.md"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"target repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR,
                    help="directory holding the .gguf and .forge.json")
    ap.add_argument("--private", action="store_true",
                    help="create the repo private (no effect if it exists)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and print the plan, upload nothing")
    ap.add_argument("--confirm", action="store_true",
                    help="required to actually upload")
    args = ap.parse_args()

    files = check_artifacts(args.artifacts)
    total = sum(p.stat().st_size for p, _ in files)

    print(f"target repo : {args.repo}  ({'private' if args.private else 'PUBLIC'})")
    print(f"total upload: {human(total)}")
    for local, remote in files:
        print(f"  {human(local.stat().st_size):>9}  {local.name}  ->  {remote}")

    if args.dry_run or not args.confirm:
        print("\nNothing uploaded.", end=" ")
        print("Re-run with --confirm to publish."
              if not args.dry_run else "(--dry-run)")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\nERROR: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    try:
        who = api.whoami()
        print(f"\nauthenticated as: {who.get('name', '?')}")
    except Exception as exc:
        print(f"\nERROR: not authenticated ({exc}).\n"
              "Run `huggingface-cli login` or set $HF_TOKEN.", file=sys.stderr)
        sys.exit(1)

    api.create_repo(repo_id=args.repo, repo_type="model",
                    private=args.private, exist_ok=True)
    print(f"repo ready: https://huggingface.co/{args.repo}")

    # One commit rather than three, so the repo is never in a state where the
    # weights are public but the card documenting the required sampling
    # parameters is not.
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
           for local, remote in files]

    api.create_commit(
        repo_id=args.repo,
        repo_type="model",
        operations=ops,
        commit_message="Falcon-H1-7B-FORGE-v2: 2.06 bpw mixed-precision GGUF",
        commit_description=(
            "Mixed-precision quantization of tiiuae/Falcon-H1-7B-Instruct.\n"
            "ssm_out and ffn_down held at Q6_K; everything else ternary.\n"
            "3.48 GB on disk, 3.68 GB peak RSS at n_ctx=4096.\n"
            "Requires repeat_last_n=2048 -- see the model card."
        ),
    )
    print(f"\ndone: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
