#!/usr/bin/env python
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "reproduction_baselines"))
from hate_common import data as hdata  # noqa: E402

CORPORA = ("hatemm", "hateclipseg")
FRACTION = 0.15


def main():
    final = {
        "developmental_test_evidence": True,
        "selector": "top ceil(0.15*T) lexical frames restricted to speech-present",
        "corpora": {},
    }
    for corpus in CORPORA:
        score_path = (ROOT / "runs/20260831_video_label_lexical_locality/premise" /
                      corpus / "scores.jsonl")
        rows = {}
        with score_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                rows[str(row["video_id"])] = row
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        video_rows = []
        for video_id, target in gt.items():
            target = np.asarray(target, dtype=int)
            if labels.get(video_id) != 1 or len(np.unique(target)) != 2:
                continue
            lexical = np.asarray(rows[video_id]["score_lexical"], dtype=float)
            speech = np.asarray(rows[video_id]["score_speech"], dtype=float) > 0
            count = max(1, int(math.ceil(FRACTION * len(target))))
            eligible = np.flatnonzero(speech)
            selected = (eligible[np.argsort(lexical[eligible])[-min(count, len(eligible)):]]
                        if len(eligible) else np.asarray([], dtype=int))
            base = float(target.mean())
            precision = float(target[selected].mean()) if len(selected) else None
            video_rows.append({
                "video_id": video_id,
                "base_positive_rate": base,
                "selected_precision": precision,
                "selected_count": int(len(selected)),
                "selected_positive_recall": float(target[selected].sum() / target.sum())
                if len(selected) else 0.0,
                "gt_positive_speech_fraction": float(
                    ((target == 1) & speech).sum() / (target == 1).sum()),
            })
        defined = [row for row in video_rows if row["selected_precision"] is not None]
        base = float(np.mean([row["base_positive_rate"] for row in video_rows]))
        precision = float(np.mean([row["selected_precision"] for row in defined]))
        final["corpora"][corpus] = {
            "n_eligible_positive_videos": len(video_rows),
            "n_zero_selection": len(video_rows) - len(defined),
            "macro_base_positive_rate": base,
            "macro_selected_precision": precision,
            "macro_selected_precision_minus_base": float(np.mean([
                row["selected_precision"] - row["base_positive_rate"]
                for row in defined])),
            "macro_selected_positive_recall": float(np.mean([
                row["selected_positive_recall"] for row in video_rows])),
            "macro_gt_positive_speech_fraction": float(np.mean([
                row["gt_positive_speech_fraction"] for row in video_rows])),
        }
    final["gate"] = {
        "both_corpora_positive_enrichment": all(
            row["macro_selected_precision_minus_base"] > 0
            for row in final["corpora"].values())
    }
    final["decision"] = (
        "PROCEED_TO_NOVELTY" if final["gate"]["both_corpora_positive_enrichment"]
        else "STOP_BEFORE_IMPLEMENTATION")
    out = ROOT / "runs/20260831_carrier_sar_pu_premise/metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
