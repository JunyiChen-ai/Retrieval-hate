#!/usr/bin/env python3
"""Generate OmniVTG intervals for a complete positive test cohort.

This producer reads only frozen split membership and video-level labels.  It
does not import frame/span ground truth or any evaluator.  The model stays
resident while videos are processed sequentially, and the JSONL is resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))

from hate_common import data as hdata  # noqa: E402
from protocol import (  # noqa: E402
    CODE_VERSION_DESCRIPTION,
    CONTRACT_VERSION,
    FORMAL_RUNTIME_VERSIONS,
    MODEL_ID,
    QUERY,
    ROW_FIELDS,
    parse_interval,
    positive_test_cohort,
)
from smoke import prepare_inputs  # noqa: E402


CORPUS_ROOTS = {
    "hatemm": Path("/home/jehc223/data/HateMM/video"),
    "hateclipseg": Path("/home/jehc223/data/HateClipSeg/videos"),
}
CANONICAL_RUN_ROOT = REPO / "runs/20260831_omnivtg_grounder_diagnostic/formal"


def load_completed(
    path: Path,
    corpus: str,
    model: str,
    cohort: set[str],
    video_paths: dict[str, Path],
) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = row.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                raise RuntimeError(f"invalid video_id at {path}:{line_number}")
            if video_id in completed:
                raise RuntimeError(f"duplicate video_id in resumed output: {video_id}")
            if set(row) != ROW_FIELDS:
                raise RuntimeError(f"invalid row schema at {path}:{line_number}")
            if (
                row["contract_version"] != CONTRACT_VERSION
                or row["corpus"] != corpus
                or row["split"] != "test"
                or row["model"] != model
                or row["query"] != QUERY
                or video_id not in cohort
                or Path(row["source_video"]).resolve() != video_paths[video_id].resolve()
                or not isinstance(row["parse_ok"], bool)
            ):
                raise RuntimeError(f"row provenance mismatch at {path}:{line_number}")
            parsed = parse_interval(row["completion"])
            if row["parse_ok"]:
                if (
                    parsed is None
                    or row["interval_seconds"] != parsed
                    or row["error_type"] is not None
                    or row["error_message"] is not None
                    or row["traceback"] is not None
                ):
                    raise RuntimeError(f"invalid successful row at {path}:{line_number}")
            elif row["interval_seconds"] is not None:
                raise RuntimeError(f"failed row carries interval at {path}:{line_number}")
            elif row["completion"] is not None:
                if (
                    parsed is not None
                    or row["error_type"] != "ParseFailure"
                    or not isinstance(row["error_message"], str)
                    or not row["error_message"]
                    or row["traceback"] is not None
                ):
                    raise RuntimeError(f"invalid parse-failure row at {path}:{line_number}")
            elif not all(
                isinstance(row[key], str) and row[key]
                for key in ("error_type", "error_message", "traceback")
            ):
                raise RuntimeError(f"invalid inference-failure row at {path}:{line_number}")
            completed.add(video_id)
    return completed


def resolve_video(corpus: str, video_id: str) -> Path:
    root = CORPUS_ROOTS[corpus]
    matches = sorted(root.glob(f"{video_id}.*"))
    files = [path for path in matches if path.is_file()]
    if len(files) != 1:
        raise RuntimeError(
            f"expected exactly one source video for {corpus}/{video_id}, got {len(files)}"
        )
    return files[0]


def build_model(model: str):
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=32768,
        max_num_batched_tokens=32768,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 0, "video": 768},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 16 * 28 * 28,
        },
    )
    sampling = SamplingParams(
        repetition_penalty=1.05,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        stop_token_ids=[151645, 151643],
        max_tokens=1024,
        include_stop_str_in_output=False,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )
    return processor, llm, sampling


def infer_one(video_path: Path, processor, llm, sampling):
    prompt_token_ids, mm_data = prepare_inputs(video_path, processor)
    outputs = llm.generate(
        prompts=[{
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"video": mm_data},
        }],
        sampling_params=sampling,
        use_tqdm=False,
    )
    completion = outputs[0].outputs[0].text.strip()
    interval = parse_interval(completion)
    return completion, interval


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def ensure_run_metadata(run_dir: Path, corpus: str, output: Path) -> None:
    observed_runtime = {
        package: version(package) for package in FORMAL_RUNTIME_VERSIONS
    }
    if observed_runtime != FORMAL_RUNTIME_VERSIONS:
        raise RuntimeError(
            f"formal runtime version mismatch: {observed_runtime}"
        )
    config = {
        "contract_version": CONTRACT_VERSION,
        "corpus": corpus,
        "split": "test",
        "cohort": "video-level-positive fixed evaluator cohort",
        "model": MODEL_ID,
        "query": QUERY,
        "predictions": str(output.resolve()),
        "runtime_versions": FORMAL_RUNTIME_VERSIONS,
        "engine_mode": "vLLM multimodal, enforce_eager=True",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
    }
    config_path = run_dir / "config.json"
    version_path = run_dir / "code_version.txt"
    if output.exists() and (not config_path.exists() or not version_path.exists()):
        raise RuntimeError("prediction rows exist without complete run metadata")
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError(f"run config mismatch: {config_path}")
    else:
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        os.replace(temporary, config_path)
    expected_version = CODE_VERSION_DESCRIPTION + "\n"
    if version_path.exists():
        if version_path.read_text() != expected_version:
            raise RuntimeError(f"code version description mismatch: {version_path}")
    else:
        temporary = version_path.with_suffix(".txt.tmp")
        temporary.write_text(expected_version)
        os.replace(temporary, version_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=sorted(CORPUS_ROOTS), required=True)
    args = parser.parse_args()
    output = CANONICAL_RUN_ROOT / args.corpus / "predictions.jsonl"

    split_ids = hdata.load_split(args.corpus, "test")
    labels = hdata.load_labels(args.corpus)
    cohort = positive_test_cohort(args.corpus, split_ids, labels)
    if not cohort:
        raise RuntimeError(f"empty positive test cohort for {args.corpus}")
    video_paths = {video_id: resolve_video(args.corpus, video_id) for video_id in cohort}
    ensure_run_metadata(output.parent, args.corpus, output)
    completed = load_completed(
        output, args.corpus, MODEL_ID, set(cohort), video_paths
    )

    pending = [video_id for video_id in cohort if video_id not in completed]
    print(
        json.dumps({
            "event": "cohort",
            "corpus": args.corpus,
            "split": "test",
            "output": str(output),
            "positive_total": len(cohort),
            "already_complete": len(completed),
            "pending": len(pending),
        }),
        flush=True,
    )
    if not pending:
        return

    processor, llm, sampling = build_model(MODEL_ID)
    for index, video_id in enumerate(pending, 1):
        video_path = video_paths[video_id]
        try:
            completion, interval = infer_one(video_path, processor, llm, sampling)
            row = {
                "contract_version": CONTRACT_VERSION,
                "video_id": video_id,
                "corpus": args.corpus,
                "split": "test",
                "model": MODEL_ID,
                "query": QUERY,
                "source_video": str(video_path.resolve()),
                "parse_ok": interval is not None,
                "interval_seconds": interval,
                "completion": completion,
                "error_type": None if interval is not None else "ParseFailure",
                "error_message": (
                    None if interval is not None
                    else "completion did not contain one valid final interval"
                ),
                "traceback": None,
            }
        except Exception as error:  # Keep failures in the denominator.
            row = {
                "contract_version": CONTRACT_VERSION,
                "video_id": video_id,
                "corpus": args.corpus,
                "split": "test",
                "model": MODEL_ID,
                "query": QUERY,
                "source_video": str(video_path.resolve()),
                "parse_ok": False,
                "interval_seconds": None,
                "completion": None,
                "error_type": type(error).__name__,
                "error_message": str(error) or repr(error),
                "traceback": traceback.format_exc(),
            }
        append_row(output, row)
        print(
            json.dumps({
                "event": "video_complete",
                "corpus": args.corpus,
                "index": len(completed) + index,
                "total": len(cohort),
                "video_id": video_id,
                "parse_ok": row["parse_ok"],
                "interval_seconds": row["interval_seconds"],
                "error_type": row["error_type"],
            }),
            flush=True,
        )
    final_completed = load_completed(
        output, args.corpus, MODEL_ID, set(cohort), video_paths
    )
    if final_completed != set(cohort):
        raise RuntimeError(f"{args.corpus}: producer ended without complete cohort")
    print(json.dumps({
        "event": "cohort_complete",
        "corpus": args.corpus,
        "split": "test",
        "positive_total": len(cohort),
    }), flush=True)


if __name__ == "__main__":
    main()
