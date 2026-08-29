"""DESC_CHANNEL -- description-quality and cost readout (FREEZE section 7 items 5-6)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from defect import is_defect, load_gt  # noqa: E402
from gen_desc import FIELDS, OUT, violations  # noqa: E402

# DashScope list price assumed (see STANCE_PILOT_RESULT.md section 7): qwen3-vl-plus
# CNY 0.002 / 1K input, CNY 0.008 / 1K output; Batch API at 50 %.
P_IN, P_OUT, BATCH_DISCOUNT = 0.002, 0.008, 0.5


def main():
    gt = load_gt(ROOT)
    rows = {}
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    n = len(rows)
    ok = [r for r in rows.values() if r["parse"] == "ok"]
    bad = [r for r in rows.values() if r["parse"] != "ok"]
    still = [r for r in rows.values() if r.get("violations")]
    blanked = [r for r in rows.values() if r.get("blanked_fields")]
    regen = [r for r in rows.values() if r.get("regenerated")]
    ti = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows.values())
    to = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows.values())
    empty_fields = sum(1 for r in ok for k in FIELDS if not (r["fields"].get(k) or "").strip())

    print("rows                    %d / 1066" % n)
    print("parse ok                %d" % len(ok))
    print("parse failed            %d  %s" % (len(bad), [r["id"] for r in bad][:10]))
    print("regenerated once        %d" % len(regen))
    print("still violating         %d" % len(still))
    print("fields blanked          %d" % sum(len(r["blanked_fields"]) for r in blanked))
    print("empty fields (any arm)  %d / %d" % (empty_fields, len(ok) * len(FIELDS)))
    print("tokens in / out         %d / %d" % (ti, to))
    print("cost @ batch 50%%        CNY %.2f" % ((ti / 1000 * P_IN + to / 1000 * P_OUT)
                                                 * BATCH_DISCOUNT))
    lens = sorted(len(" ".join((r["fields"].get(k) or "") for k in FIELDS)) for r in ok)
    print("desc chars  min/median/max  %d / %d / %d"
          % (lens[0], lens[len(lens) // 2], lens[-1]))
    dset = [v for v in rows if is_defect(gt[v]["text"])]
    print("DEFECT videos with a description: %d / %d"
          % (sum(1 for v in dset if rows[v]["parse"] == "ok"), len(dset)))

    if len(sys.argv) > 1:
        for vid in sys.argv[1:]:
            r = rows[vid]
            print("\n" + "=" * 78)
            print(vid, "| label", gt[vid]["label"], "| transcript", repr(gt[vid]["text"][:60]))
            for k in FIELDS:
                print("  %-19s %s" % (k, (r["fields"] or {}).get(k, "")))


if __name__ == "__main__":
    main()
