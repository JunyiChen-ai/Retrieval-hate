#!/usr/bin/env python
"""B-SRTD teacher scorer — score every lattice cell with a frozen open-weight teacher.

Rules frozen in research-wiki/EXP_bsrtd_prereg.md.  This script produces the ONLY teacher
artefact the pilot is allowed to read:

    data/Counterfactual/BSRTD/teacher_scores.jsonl

Contract (append-only, idempotent, resumable):
    {"key": "<lang>|<split>|<seed_id>|<cell>", "lang", "split", "seed_id", "cell",
     "text_sha256", "model", "engine", "prompt_id", "score" (float in [0,1]),
     "raw" (the model's literal reply), "ts"}
A row is considered DONE — and the API call skipped — iff a row with the same key,
text_sha256, model and prompt_id already exists in the file.  Re-running after any
interruption resumes exactly where it stopped; re-running after the lattice text changes
re-scores only the changed cells.

TEACHER (frozen order; see prereg §4).  The generator of the lattice is Claude, so Claude
is INELIGIBLE as teacher: a teacher that is the generator would only re-read the
generator's own design intent, which is already present in `cell_expected_labels`.
    primary      Qwen/Qwen2.5-72B-Instruct
    alternate 1  meta-llama/Llama-3.3-70B-Instruct
    alternate 2  deepseek-ai/DeepSeek-V3
All three are open-weight, so any third party can reproduce the scores; all three are >=
10x the student encoder's parameter count (Qwen2.5-VL-7B), so this is not self-distillation.

SPLITS: train and val ONLY.  `test` is rejected by assertion.

Access is via any OpenAI-compatible chat-completions endpoint:
    BSRTD_TEACHER_BASE_URL   e.g. https://openrouter.ai/api/v1
    BSRTD_TEACHER_API_KEY
    BSRTD_TEACHER_MODEL      overrides --model
`--engine synthetic` fabricates deterministic scores for rehearsal; rows are tagged
engine="synthetic" and the pilot refuses to emit a primary verdict from them.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bsrtd_lattice import (  # noqa: E402
    BSRTD_DIR, CELLS, PROMPT_ID, TEACHER_PATH, load_lattices, load_teacher_scores,
    normalise_row, teacher_key, text_sha)

TEACHERS = ["Qwen/Qwen2.5-72B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "deepseek-ai/DeepSeek-V3"]

# ---------------------------------------------------------------- frozen prompt (BSRTD-T1)
SYSTEM = (
    "You are a careful forensic rater assisting a hateful-video detection system. You are "
    "shown the TITLE and TRANSCRIPT of one short video. Your ONLY job is to rate how much "
    "HATE evidence the text carries. Hate evidence = content that attacks, dehumanises, "
    "demeans, threatens or incites against people for a protected attribute (race, "
    "ethnicity, religion, nationality, gender, sexual orientation, disability, caste, "
    "immigration status, etc.). Content that REPORTS, QUOTES, CONDEMNS or ARGUES AGAINST "
    "hate is NOT hate evidence, even when it repeats hateful wording verbatim. Content that "
    "is merely rude, crude or offensive without a protected-attribute target is NOT hate "
    "evidence. Rate only what the text itself carries; do not speculate about the video."
)

RUBRIC = (
    "Output a single integer from 0 to 100.\n"
    "  0-10   no hate evidence at all (neutral, benign, unrelated).\n"
    "  11-30  faint or ambiguous cue that could relate to hate but is unclear; or hateful "
    "wording that is clearly quoted in order to condemn it.\n"
    "  31-55  hate-related content targeting a protected group, but implicit, coded, or "
    "carried mostly by insinuation.\n"
    "  56-80  clear hate content targeting a protected group: stereotyping, demeaning "
    "generalisation, mockery of the group as a group.\n"
    "  81-100 explicit, unambiguous hate: slurs used as attacks, dehumanisation, threats, "
    "or incitement against a protected group.\n"
    "Answer with the integer and nothing else."
)


def build_prompt(text):
    return (RUBRIC + "\n\nTEXT:\n" + (text.strip() if text and text.strip() else "(none)")
            + "\n\nRating (0-100):")


_INT = re.compile(r"-?\d+")


def parse_score(raw):
    """Frozen parse: first integer in the reply, clipped to [0, 100], /100.  None on failure."""
    if raw is None:
        return None
    m = _INT.search(str(raw))
    if not m:
        return None
    v = int(m.group(0))
    return float(min(100, max(0, v))) / 100.0


# ---------------------------------------------------------------------------- engines
def make_client(base_url, api_key):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def score_openai(client, model, text, max_retries=5):
    """Deterministic single call: temperature 0, top_p 1, 8 output tokens."""
    last = None
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": build_prompt(text)}],
                temperature=0.0, top_p=1.0, max_tokens=8, seed=20260810)
            raw = (r.choices[0].message.content or "").strip()
            s = parse_score(raw)
            if s is not None:
                return s, raw
            last = f"unparseable:{raw!r}"
        except Exception as e:  # noqa: BLE001 - transport/rate-limit; retry with backoff
            last = repr(e)
        time.sleep(min(30.0, 2.0 ** attempt))
    raise RuntimeError(f"teacher call failed after {max_retries} attempts: {last}")


def score_synthetic(model, text):
    """Deterministic hash-based stand-in.  REHEARSAL ONLY."""
    import hashlib
    h = int(hashlib.sha256((model + "|" + text).encode("utf-8")).hexdigest()[:8], 16)
    v = h % 101
    return float(v) / 100.0, str(v)


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="B-SRTD teacher scorer (idempotent, resumable)")
    ap.add_argument("--lattice-root", default=BSRTD_DIR,
                    help="directory holding {train,val}_lattices_{en,zh}.jsonl")
    ap.add_argument("--out", default=TEACHER_PATH)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--langs", default="en,zh")
    ap.add_argument("--engine", choices=["openai", "synthetic"], default="openai")
    ap.add_argument("--model", default=os.environ.get("BSRTD_TEACHER_MODEL", TEACHERS[0]))
    ap.add_argument("--base-url", default=os.environ.get("BSRTD_TEACHER_BASE_URL",
                                                         "https://openrouter.ai/api/v1"))
    ap.add_argument("--limit", type=int, default=0, help=">0: only the first N cells (debug)")
    ap.add_argument("--progress-every", type=int, default=25)
    a = ap.parse_args()

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    assert "test" not in splits, "REFUSING: the teacher never sees the test split"
    langs = [s.strip() for s in a.langs.split(",") if s.strip()]

    done = load_teacher_scores(a.out)
    todo = []
    for lang in langs:
        for split in splits:
            for r0 in load_lattices(split, lang, root=a.lattice_root):
                r = normalise_row(r0, split, lang)
                for ci, c in enumerate(CELLS):
                    k = teacher_key(lang, split, r["seed_id"], c)
                    sha = text_sha(r["texts"][ci])
                    prev = done.get(k)
                    if (prev and prev.get("text_sha256") == sha
                            and prev.get("model") == a.model
                            and prev.get("prompt_id") == PROMPT_ID):
                        continue
                    todo.append((lang, split, r["seed_id"], c, r["texts"][ci], sha))
    if a.limit:
        todo = todo[:a.limit]

    print(f"teacher={a.model} engine={a.engine} prompt={PROMPT_ID}", flush=True)
    print(f"cached={len(done)}  to_score={len(todo)}  out={a.out}", flush=True)
    if not todo:
        print("nothing to do (cache complete)", flush=True)
        return

    client = None
    if a.engine == "openai":
        key = os.environ.get("BSRTD_TEACHER_API_KEY")
        assert key, ("BSRTD_TEACHER_API_KEY is not set; export it (and optionally "
                     "BSRTD_TEACHER_BASE_URL / BSRTD_TEACHER_MODEL) before scoring")
        client = make_client(a.base_url, key)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    with open(a.out, "a") as f:
        for i, (lang, split, sid, cell, text, sha) in enumerate(todo):
            if a.engine == "synthetic":
                s, raw = score_synthetic(a.model, text)
            else:
                s, raw = score_openai(client, a.model, text)
            f.write(json.dumps({
                "key": teacher_key(lang, split, sid, cell), "lang": lang, "split": split,
                "seed_id": sid, "cell": cell, "text_sha256": sha, "model": a.model,
                "engine": a.engine, "prompt_id": PROMPT_ID, "score": s, "raw": raw,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
            f.flush()
            if (i + 1) % a.progress_every == 0:
                el = time.time() - t0
                print(f"PROGRESS scored={i+1}/{len(todo)} elapsed={el:.0f}s "
                      f"rate={(i+1)/max(el,1e-9):.2f}/s", flush=True)
    print(f"DONE scored={len(todo)} in {time.time()-t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
