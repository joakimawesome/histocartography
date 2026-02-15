#!/usr/bin/env python
"""
Local launcher for nuclei-graph extraction (no SLURM).

This mirrors the intent of `slurm_nuclei_graphs.sh` by running `batch_runner.py`
either:
  - once in serial mode (process all slides in a single process), or
  - as a local "array" (one process per slide index, with a max concurrency),
    similar to SLURM's `--array=0-31%10`.

Examples (PowerShell):
  .venv/Scripts/python -B run_nuclei_graphs_local.py `
    --manifest "C:\\path\\to\\manifest.csv" `
    --slides_root "C:\\path\\to\\dataset_root" `
    --out_dir "C:\\path\\to\\wsi_processed" `
    --model_path "checkpoints/hovernet_pannuke.pth" `
    --slide_col "path" `
    --graph_method knn --k 5 --feat_mode stats `
    --array '0-31%2'

  # Serial (all slides in one process):
  .venv/Scripts/python -B run_nuclei_graphs_local.py --manifest ... --out_dir ...
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional


def _sanitize_token(s: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.=")
    s = (s or "").strip()
    if not s:
        return "run"
    return "".join(c if c in allowed else "_" for c in s)


def _count_manifest_rows(manifest_path: Path) -> int:
    # Count CSV rows excluding the header row.
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


_ARRAY_RE = re.compile(
    r"^(?P<spec>all|\d+(?:-\d+)?)\s*(?:%\s*(?P<max>\d+))?\s*$",
    flags=re.IGNORECASE,
)


def _parse_array(array_spec: str, manifest_path: Path) -> tuple[list[int], Optional[int]]:
    m = _ARRAY_RE.match((array_spec or "").strip())
    if not m:
        raise ValueError(
            f"Invalid --array '{array_spec}'. Expected formats like '0-31%10', '0-31', '8', or 'all%4'."
        )

    spec = m.group("spec").lower()
    max_str = m.group("max")
    max_concurrent = int(max_str) if max_str else None
    if max_concurrent is not None and max_concurrent < 1:
        raise ValueError(f"Invalid --array max concurrency '{max_str}'; must be >= 1.")

    if spec == "all":
        n = _count_manifest_rows(manifest_path)
        return list(range(n)), max_concurrent

    if "-" in spec:
        start_s, end_s = spec.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if start < 0:
            raise ValueError("Array start must be >= 0.")
        if end < start:
            raise ValueError("Array end must be >= start.")
        return list(range(start, end + 1)), max_concurrent

    idx = int(spec)
    if idx < 0:
        raise ValueError("Array index must be >= 0.")
    return [idx], max_concurrent


def _resolve_path(p: Optional[str], base_dir: Path) -> Optional[str]:
    if p is None:
        return None
    p = str(p)
    pp = Path(p)
    if not pp.is_absolute():
        pp = (base_dir / pp).resolve()
    return str(pp)


def _resolve_executable(exe: str, base_dir: Path) -> str:
    """Resolve an executable passed as a path-ish string or command name."""
    exe = str(exe)

    # If the user gave a command name (e.g. "python"), prefer PATH resolution.
    looks_like_path = any(sep in exe for sep in ("/", "\\"))
    if not looks_like_path and not Path(exe).is_absolute():
        resolved = shutil.which(exe)
        return resolved or exe

    # Otherwise treat it as a path (absolute or relative to repo root).
    resolved = Path(_resolve_path(exe, base_dir) or exe)
    if resolved.exists():
        return str(resolved)

    # Windows convenience: allow ".../python" to resolve to ".../python.exe".
    if os.name == "nt" and resolved.suffix == "":
        candidate = resolved.with_suffix(".exe")
        if candidate.exists():
            return str(candidate)

    return str(resolved)


def _build_batch_runner_cmd(args: argparse.Namespace, *, python: str, batch_runner: str, idx: Optional[int]) -> list[str]:
    cmd = [
        python,
        "-u",
        batch_runner,
        "--manifest",
        args.manifest,
        "--out_dir",
        args.out_dir,
        "--model_path",
        args.model_path,
        "--slide_col",
        args.slide_col,
        "--graph_method",
        args.graph_method,
        "--k",
        str(args.k),
        "--r",
        str(args.r),
        "--feat_mode",
        args.feat_mode,
    ]

    if args.slides_root:
        cmd += ["--slides_root", args.slides_root]
    if args.run_name:
        cmd += ["--run_name", args.run_name]
    if args.gnn_model_path:
        cmd += ["--gnn_model_path", args.gnn_model_path]
    if args.feat_batch_size is not None:
        cmd += ["--feat_batch_size", str(args.feat_batch_size)]
    if args.feat_num_workers is not None:
        cmd += ["--feat_num_workers", str(args.feat_num_workers)]
    if args.feat_pin_memory is not None:
        cmd += ["--feat_pin_memory" if args.feat_pin_memory else "--no-feat_pin_memory"]
    if args.seg_device:
        cmd += ["--seg_device", args.seg_device]
    if args.seg_batch_size is not None:
        cmd += ["--seg_batch_size", str(args.seg_batch_size)]
    if args.seg_tile_size is not None:
        cmd += ["--seg_tile_size", str(args.seg_tile_size)]
    if args.seg_overlap is not None:
        cmd += ["--seg_overlap", str(args.seg_overlap)]
    if args.seg_level is not None:
        cmd += ["--seg_level", str(args.seg_level)]
    if args.seg_min_nucleus_area is not None:
        cmd += ["--seg_min_nucleus_area", str(args.seg_min_nucleus_area)]
    if args.skip_errors:
        cmd += ["--skip_errors"]
    if args.force_rerun:
        cmd += ["--force_rerun"]
    if idx is not None:
        cmd += ["--slurm_array_idx", str(idx)]
    return cmd


@dataclass(frozen=True)
class _Running:
    idx: Optional[int]
    proc: subprocess.Popen[bytes]
    stdout_path: Path
    stderr_path: Path
    stdout_f: object
    stderr_f: object


def _poll_any_finished(running: list[_Running]) -> Optional[tuple[int, int]]:
    """Return (pos_in_list, exit_code) for a finished proc; otherwise None."""
    for i, r in enumerate(running):
        exit_code = r.proc.poll()
        if exit_code is not None:
            return i, int(exit_code)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local launcher that mirrors slurm_nuclei_graphs.sh using batch_runner.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core inputs/outputs
    parser.add_argument("--manifest", default="../Pediatric-Brain-Tumor/data/manifests/v1.0.1/manifest.csv", help="Path to CSV manifest.")
    parser.add_argument("--slide_col", default="path", help="Column name for slide path in manifest.")
    parser.add_argument("--slides_root", default="../Pediatric-Brain-Tumor", help="Optional root dir to resolve relative slide paths.")
    parser.add_argument("--out_dir", default="../Pediatric-Brain-Tumor/data/wsi_processed/", help="Output directory (batch_runner will create a run subdir).")

    # Models / pipeline args (mirror batch_runner.py)
    parser.add_argument("--model_path", default="checkpoints/hovernet_pannuke.pth", help="HoVerNet checkpoint path.")
    parser.add_argument("--gnn_model_path", default=None, help="Optional GNN checkpoint path.")
    parser.add_argument("--graph_method", default="knn", choices=["knn", "radius"], help="Graph construction method.")
    parser.add_argument("--k", type=int, default=5, help="k for kNN graph.")
    parser.add_argument("--r", type=float, default=50.0, help="r for radius graph.")
    parser.add_argument(
        "--feat_mode",
        default="stats",
        choices=["handcrafted", "deep", "hybrid", "stats", "gnn"],
        help="Feature extraction mode.",
    )
    parser.add_argument("--feat_batch_size", type=int, default=None, help="Batch size for deep or hybrid features.")
    parser.add_argument("--feat_num_workers", type=int, default=None, help="DataLoader workers for deep or hybrid features.")
    parser.add_argument(
        "--feat_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable DataLoader pin_memory for deep or hybrid features.",
    )
    parser.add_argument("--run_name", default=None, help="Optional run subdirectory name under --out_dir.")
    parser.add_argument(
        "--skip_errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue even if a slide fails.",
    )
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun even if cached outputs exist.")

    # Segmentation overrides (passed through to batch_runner.py)
    parser.add_argument("--seg_device", default=None, choices=["cuda", "cpu"], help="Override segmentation device.")
    parser.add_argument(
        "--seg_batch_size",
        type=int,
        default=1,
        help="Segmentation inference batch size (set low by default to avoid CUDA OOM on local GPUs).",
    )
    parser.add_argument("--seg_tile_size", type=int, default=None, help="Segmentation tile size (pixels).")
    parser.add_argument("--seg_overlap", type=int, default=None, help="Segmentation tile overlap (pixels).")
    parser.add_argument("--seg_level", type=int, default=None, help="Segmentation WSI pyramid level (0 = full-res).")
    parser.add_argument("--seg_min_nucleus_area", type=int, default=None, help="Minimum nucleus area filter.")

    # Execution / logging
    parser.add_argument(
        "--array",
        default=None,
        help="Optional SLURM-like array spec: '0-31%%10', '8', or 'all%%4'. If omitted, runs serially in one process.",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=None,
        help="Max concurrent workers in --array mode. Overrides the %%N in --array, if provided.",
    )
    parser.add_argument("--logs_dir", default="logs", help="Directory for stdout/stderr logs.")
    parser.add_argument("--job_name", default="nuclei_graphs_local", help="Prefix for log filenames.")
    parser.add_argument("--dry_run", action="store_true", help="Print the commands that would run, then exit.")
    parser.add_argument(
        "--pytorch_alloc_conf",
        default=None,
        help="If set, exports PYTORCH_ALLOC_CONF for child processes (e.g. 'expandable_segments:True').",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable to use for child processes (defaults to this interpreter).",
    )
    parser.add_argument(
        "--batch_runner",
        default=None,
        help="Path to batch_runner.py (defaults to repo_root/batch_runner.py).",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    # Resolve/normalize key paths so we can run this script from anywhere.
    args.manifest = _resolve_path(args.manifest, repo_root)  # type: ignore[assignment]
    args.out_dir = _resolve_path(args.out_dir, repo_root)  # type: ignore[assignment]
    args.model_path = _resolve_path(args.model_path, repo_root)  # type: ignore[assignment]
    args.gnn_model_path = _resolve_path(args.gnn_model_path, repo_root)  # type: ignore[assignment]
    args.slides_root = _resolve_path(args.slides_root, repo_root)  # type: ignore[assignment]
    args.logs_dir = _resolve_path(args.logs_dir, repo_root)  # type: ignore[assignment]

    python = _resolve_executable(args.python or sys.executable, repo_root)

    batch_runner = args.batch_runner or str((repo_root / "batch_runner.py").resolve())
    batch_runner = _resolve_path(batch_runner, repo_root) or batch_runner

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    if not Path(args.model_path).is_file():
        raise SystemExit(f"Model checkpoint not found: {args.model_path}")

    if args.gnn_model_path and not Path(args.gnn_model_path).is_file():
        raise SystemExit(f"GNN checkpoint not found: {args.gnn_model_path}")

    if not Path(batch_runner).is_file():
        raise SystemExit(f"batch_runner.py not found: {batch_runner}")

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_token = _sanitize_token(args.job_name)
    run_token = _sanitize_token(args.run_name) if args.run_name else "auto"
    log_prefix = f"{job_token}__{run_token}__{run_id}"

    env = os.environ.copy()
    # Ensure child output is unbuffered in addition to `-u` (some libs respect this).
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.pytorch_alloc_conf:
        # PyTorch 2.10+ supports PYTORCH_ALLOC_CONF. Older builds used PYTORCH_CUDA_ALLOC_CONF.
        env["PYTORCH_ALLOC_CONF"] = args.pytorch_alloc_conf
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.pytorch_alloc_conf

    if not args.array:
        # Serial mode: let batch_runner handle processing the full manifest in one process.
        stdout_path = logs_dir / f"{log_prefix}.out"
        stderr_path = logs_dir / f"{log_prefix}.err"
        cmd = _build_batch_runner_cmd(args, python=python, batch_runner=batch_runner, idx=None)
        print(f"[serial] logs: {stdout_path.name} / {stderr_path.name}")
        if args.dry_run:
            print(cmd)
            return 0
        with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
            p = subprocess.Popen(cmd, cwd=str(repo_root), stdout=out_f, stderr=err_f, env=env)
            return int(p.wait())

    # Array mode: one process per index, like SLURM array tasks.
    indices, max_from_spec = _parse_array(args.array, manifest_path)
    if not indices:
        print("[array] no slides found in manifest; nothing to do.")
        return 0

    num_slides = _count_manifest_rows(manifest_path)
    in_range = [i for i in indices if 0 <= i < num_slides]
    in_range_set = set(in_range)
    dropped = [i for i in indices if i not in in_range_set]
    if dropped:
        print(f"[array] skipping out-of-range indices (manifest rows={num_slides}): {dropped[:20]}")
    indices = in_range
    if not indices:
        print("[array] all requested indices were out of range; nothing to do.")
        return 0

    max_concurrent = args.max_concurrent or max_from_spec or 1
    if max_concurrent < 1:
        raise SystemExit("--max_concurrent must be >= 1.")

    print(f"[array] indices={indices[0]}..{indices[-1]} (n={len(indices)}), max_concurrent={max_concurrent}")

    running: list[_Running] = []
    results: dict[int, int] = {}

    def launch(idx: int) -> None:
        stdout_path = logs_dir / f"{log_prefix}__idx{idx}.out"
        stderr_path = logs_dir / f"{log_prefix}__idx{idx}.err"
        cmd = _build_batch_runner_cmd(args, python=python, batch_runner=batch_runner, idx=idx)
        print(f"[start] idx={idx} logs: {stdout_path.name} / {stderr_path.name}")
        if args.dry_run:
            print(cmd)
            results[idx] = 0
            return
        out_f = stdout_path.open("wb")
        err_f = stderr_path.open("wb")
        try:
            proc = subprocess.Popen(cmd, cwd=str(repo_root), stdout=out_f, stderr=err_f, env=env)
        except Exception:
            out_f.close()
            err_f.close()
            raise
        running.append(_Running(idx=idx, proc=proc, stdout_path=stdout_path, stderr_path=stderr_path, stdout_f=out_f, stderr_f=err_f))

    try:
        for idx in indices:
            while not args.dry_run and len(running) >= max_concurrent:
                finished = _poll_any_finished(running)
                if finished is None:
                    time.sleep(0.25)
                    continue
                pos, exit_code = finished
                r = running.pop(pos)
                try:
                    r.stdout_f.close()
                    r.stderr_f.close()
                except Exception:
                    pass
                results[int(r.idx)] = int(exit_code)
                print(f"[done]  idx={r.idx} exit_code={exit_code}")

            launch(idx)

        while not args.dry_run and running:
            finished = _poll_any_finished(running)
            if finished is None:
                time.sleep(0.25)
                continue
            pos, exit_code = finished
            r = running.pop(pos)
            try:
                r.stdout_f.close()
                r.stderr_f.close()
            except Exception:
                pass
            results[int(r.idx)] = int(exit_code)
            print(f"[done]  idx={r.idx} exit_code={exit_code}")
    except KeyboardInterrupt:
        if not args.dry_run:
            print("\n[interrupt] terminating running tasks...")
            for r in running:
                try:
                    r.proc.terminate()
                except Exception:
                    pass
            for r in running:
                try:
                    r.proc.wait(timeout=10)
                except Exception:
                    pass
                try:
                    r.stdout_f.close()
                    r.stderr_f.close()
                except Exception:
                    pass
        raise

    failed = {idx: code for idx, code in results.items() if code != 0}
    if failed:
        print(f"[summary] failed {len(failed)}/{len(results)}: {sorted(failed.items())[:10]}")
        return 1

    print(f"[summary] success {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
