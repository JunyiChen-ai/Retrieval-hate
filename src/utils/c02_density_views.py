#!/usr/bin/env python
"""c02_density_views.py -- FROZEN construction of the C02 evidence-density view orbit.

Record: refine-logs/C02_A0_V9_RECORD.md.  Registry authority:
TARGET_STATE.json::iteration_8_stage0_bounded_extraction_amendment.

THE CONTRACT THE 2026-07-29 REVIEW IMPOSED, AND HOW IT IS MET HERE
    "A label-preserving C02 view must retain the complete native transcript as an
    ordered subsequence and may only add controlled repetition."
    (refine-logs/C02_DESIGN_REVIEW.md, first review, blocking finding 2.)

    Every view produced below is `T` with one contiguous block of `T` duplicated
    in place, separated by a single space.  Nothing is deleted, reordered,
    summarised, translated or paraphrased, so `T` is an ordered subsequence of
    every view by construction.  `assert_subsequence` proves it per item at run
    time and is called by the extractor before any GPU forward.

THE ORBIT (6 views, one extraction, every arm is a sub-orbit of it)
    NAT   : the native text channel T, verbatim.
    RFULL : T + " " + T                          -- exact-content quantity doubling.
    RW1..4: T with contiguous window k duplicated in place -- localized repetition.

    Windows are the K=4 contiguous character quarters of T at cuts
    c_k = (k * len(T)) // 4.  There is no snapping heuristic and no tunable
    parameter: the cut rule is one integer expression, is dataset-symmetric
    (English whitespace text and Chinese non-whitespace text alike), and can only
    ever duplicate a contiguous substring.

WHY THE TEXT CHANNEL AND NOT "THE TRANSCRIPT"
    The object the deployed encoder actually consumes is the `text` field of
    data/gt/<DS>/<split>.jsonl (src/utils/generate_VideoMLLM_embedding_lora_HF.py
    :438-442).  On HateMM that field is the ASR transcript; on MHC-ZH it is
    MultiHateClip's harvested title + " . " + its own speech transcript
    (scripts/prep_mhc.py:73-78, re-verified in ERRPAT_MHC-ZH_2026-07-26.md §5).
    Defining the orbit on the field the encoder is fed is the only definition that
    is protocol-faithful on BOTH datasets; it needs no time alignment and no
    external asset.

IDENTITY (DEGENERATE) CASES -- FAIL CLOSED, COUNTED, NEVER SILENT
    A view may only ever be `T` itself when duplicating would change the prompt in
    a way that is not controlled repetition.  Three declared causes:
      EMPTY_TEXT     T.strip() == "".  Two sub-cases, both handled identically.
                     For T == "" the deployed prompt substitutes "(none)"; T + " " + T
                     would be " ", which is truthy, so the prompt would flip from
                     "(none)" to " " -- a prompt edit, not a density view.  For
                     whitespace-only T (the case that ACTUALLY fires: 39 of 744 HateMM
                     train rows, 0 of 579 MHC-ZH) there is no "(none)" flip, because
                     whitespace is already truthy; the guard fires because repeating
                     whitespace changes token count without changing evidence density,
                     which is not a controlled repetition of any evidence.
                     All six views are set to T.
      LENGTH_GUARD   len(T) > L_MAX.  A CHARACTER budget bounding the growth of the
                     forward's sequence length under doubling.  L_MAX = 12000 is a
                     NEW constant chosen here, not an inherited one:
                     C02_EXPERIMENT_PLAN.md §3.1 excluded and counted items that would
                     truncate under the frozen native TOKENIZER limit, which is the
                     closest precedent and the same spirit but NOT the same criterion.
                     All six views are set to T.
      EMPTY_WINDOW   window k of T is empty (only possible when len(T) < 4).  RW_k
                     alone is set to T; the other views are unaffected.
    In every case the extractor computes the NAT vector ONCE and copies it into the
    degenerate view slots, so the degenerate orbit is bit-identical by construction
    rather than by tolerance.
"""

import hashlib

VIEW_NAMES = ("NAT", "RFULL", "RW1", "RW2", "RW3", "RW4")
NON_NATIVE_VIEWS = ("RFULL", "RW1", "RW2", "RW3", "RW4")
K_WINDOWS = 4
SEP = " "
L_MAX = 12000            # characters of native T above which the orbit is the identity
RANDOM_WINDOW_HASH = "blake2b"
RANDOM_WINDOW_SALT = b"C02-A0-v1/random-window/"

DEGEN_NONE = "NONE"
DEGEN_EMPTY_TEXT = "EMPTY_TEXT"
DEGEN_LENGTH_GUARD = "LENGTH_GUARD"


def window_cuts(length, k_windows=K_WINDOWS):
    """c_0 = 0 <= c_1 <= ... <= c_K = length, the frozen contiguous quarter cuts."""
    return [(k * length) // k_windows for k in range(k_windows + 1)]


def build_views(text):
    """Return (views, meta) for one native text channel string.

    views: dict view_name -> str, always containing every name in VIEW_NAMES.
    meta : dict describing degeneracy, window sizes and the identity set.
    """
    if text is None:
        text = ""
    length = len(text)

    if text.strip() == "":
        degen = DEGEN_EMPTY_TEXT
    elif length > L_MAX:
        degen = DEGEN_LENGTH_GUARD
    else:
        degen = DEGEN_NONE

    if degen != DEGEN_NONE:
        views = {name: text for name in VIEW_NAMES}
        meta = {
            "len_native": length,
            "degenerate": degen,
            "cuts": None,
            "window_lens": None,
            "empty_windows": None,
            "identity_views": list(NON_NATIVE_VIEWS),
        }
        return views, meta

    cuts = window_cuts(length)
    views = {"NAT": text, "RFULL": text + SEP + text}
    window_lens = []
    empty_windows = []
    identity_views = []
    for k in range(1, K_WINDOWS + 1):
        lo, hi = cuts[k - 1], cuts[k]
        window = text[lo:hi]
        window_lens.append(len(window))
        name = "RW{}".format(k)
        if window == "":
            empty_windows.append(k)
            identity_views.append(name)
            views[name] = text
        else:
            views[name] = text[:hi] + SEP + window + text[hi:]

    meta = {
        "len_native": length,
        "degenerate": DEGEN_NONE,
        "cuts": cuts,
        "window_lens": window_lens,
        "empty_windows": empty_windows,
        "identity_views": identity_views,
    }
    return views, meta


def assert_subsequence(text, view):
    """Hard proof that the native text survives in the view as an ORDERED SUBSEQUENCE.

    Raises AssertionError otherwise.  Called per item per view by the extractor
    before any GPU work, so a construction bug can never reach a forward pass.
    """
    it = iter(view)
    ok = all(ch in it for ch in text)
    assert ok, "C02 VIEW CONTRACT VIOLATED: native text is not an ordered subsequence"
    return True


def random_window(item_id, k_windows=K_WINDOWS):
    """Deterministic, label-blind window index in 1..K for RANDOM_WINDOW_REPEAT."""
    h = hashlib.blake2b(RANDOM_WINDOW_SALT + str(item_id).encode("utf-8"),
                        digest_size=8).digest()
    return int.from_bytes(h, "big") % k_windows + 1


def argmax_window(scores):
    """P3 K=4 evidence-density argmax, ties -> lowest index.  1-based."""
    best, arg = None, None
    for i, s in enumerate(scores):
        if best is None or s > best:
            best, arg = s, i
    return arg + 1


def argmin_window(scores):
    """P3 K=4 evidence-density argmin, ties -> lowest index.  1-based."""
    best, arg = None, None
    for i, s in enumerate(scores):
        if best is None or s < best:
            best, arg = s, i
    return arg + 1


def self_test():
    """Pure-string fail-closed self-test.  No data, cache, model, label or GPU access.

    Run at preparation time on the login node AND again inside the SLURM extraction
    job before the model is loaded, so a construction bug can never reach a forward
    pass or burn a queue slot after the encoder is up.
    """
    cases = []

    # 1. ordinary English text: every view repeats and preserves the native order
    t = "alpha beta gamma delta epsilon zeta"
    v, m = build_views(t)
    assert m["degenerate"] == DEGEN_NONE
    assert v["NAT"] == t
    assert v["RFULL"] == t + SEP + t
    assert m["cuts"] == window_cuts(len(t))
    assert sum(m["window_lens"]) == len(t)
    for name in VIEW_NAMES:
        assert_subsequence(t, v[name])
    for k in range(1, K_WINDOWS + 1):
        name = "RW{}".format(k)
        assert v[name] != t, "window {} must actually repeat".format(k)
        assert len(v[name]) == len(t) + len(SEP) + m["window_lens"][k - 1]
    assert len(set(v[n] for n in VIEW_NAMES)) == len(VIEW_NAMES), \
        "the six views of ordinary text must be six distinct strings"
    cases.append("ordinary_english")

    # 2. non-whitespace CJK text behaves identically (dataset symmetry)
    t = "测试文本内容一二三四"
    v, m = build_views(t)
    assert m["degenerate"] == DEGEN_NONE
    for name in VIEW_NAMES:
        assert_subsequence(t, v[name])
    assert v["RW1"].startswith(t[:m["cuts"][1]] + SEP)
    cases.append("cjk")

    # 3. empty text -> full identity orbit, prompt untouched
    for t in ("", "   ", "\n\t "):
        v, m = build_views(t)
        assert m["degenerate"] == DEGEN_EMPTY_TEXT
        assert all(v[n] == t for n in VIEW_NAMES)
        assert m["identity_views"] == list(NON_NATIVE_VIEWS)
    cases.append("empty_text")

    # 4. length guard -> full identity orbit
    t = "x" * (L_MAX + 1)
    v, m = build_views(t)
    assert m["degenerate"] == DEGEN_LENGTH_GUARD
    assert all(v[n] == t for n in VIEW_NAMES)
    t = "y" * L_MAX
    v, m = build_views(t)
    assert m["degenerate"] == DEGEN_NONE, "L_MAX itself must NOT trip the guard"
    cases.append("length_guard")

    # 5. short text -> empty windows are identity for that view only
    t = "ab"
    v, m = build_views(t)
    assert m["degenerate"] == DEGEN_NONE
    assert m["empty_windows"], "len(T)=2 must produce at least one empty window"
    for k in m["empty_windows"]:
        assert v["RW{}".format(k)] == t
    assert v["RFULL"] == t + SEP + t
    for name in VIEW_NAMES:
        assert_subsequence(t, v[name])
    cases.append("short_text_empty_window")

    # 6. subsequence checker actually rejects a deletion
    try:
        assert_subsequence("abcdef", "abdef")
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("assert_subsequence accepted a deletion")
    cases.append("deletion_rejected")

    # 7. selectors are deterministic, label-blind and in range
    for vid in ("hate_video_95", "BV1f8411b7Xz", "non_hate_video_1"):
        r = random_window(vid)
        assert 1 <= r <= K_WINDOWS and r == random_window(vid)
    assert argmax_window([0, 3, 3, 1]) == 2, "argmax ties -> lowest index"
    assert argmin_window([2, 0, 0, 1]) == 2, "argmin ties -> lowest index"
    assert argmax_window([0, 0, 0, 0]) == 1 and argmin_window([3, 3, 3, 3]) == 1
    cases.append("selectors")

    return cases


if __name__ == "__main__":
    print("c02_density_views self-test PASS:", ", ".join(self_test()))
