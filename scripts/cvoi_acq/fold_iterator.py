from __future__ import annotations

def iter_registered_folds(outer,inner):
    """Single iterator used by real and formal-shaped synthetic fold assets."""
    if set(outer)!=set(inner):raise RuntimeError("HALT_FOLD_SEED_MISMATCH")
    for seed in sorted(outer,key=int):
        ofolds=outer[seed]["folds"]
        all_v={v for f in ofolds for v in f["video_ids"]};all_g={g for f in ofolds for g in f["group_ids"]}
        if sum(len(f["video_ids"]) for f in ofolds)!=len(all_v):raise RuntimeError("HALT_OUTER_VIDEO_DUPLICATE")
        for outer_fold,q in enumerate(ofolds):
            qv=set(q["video_ids"]);qg=set(q["group_ids"]);train_v=all_v-qv;train_g=all_g-qg
            if qv&train_v or qg&train_g:raise RuntimeError("HALT_OUTER_ISOLATION")
            infolds=inner[seed][str(outer_fold)]["folds"];seen=set();inner_rows=[]
            for f in infolds:
                eg=set(f["group_ids"]);ev=set(f["video_ids"])
                if eg&qg or ev&qv or seen&eg:raise RuntimeError("HALT_INNER_ISOLATION")
                seen|=eg;inner_rows.append({"inner_fold":int(f["fold"]),"eval_groups":tuple(sorted(eg)),"eval_videos":tuple(sorted(ev)),"fit_groups":tuple(sorted(train_g-eg)),"fit_videos":tuple(sorted(train_v-ev))})
            if seen!=train_g:raise RuntimeError("HALT_INNER_GROUP_COVERAGE")
            yield {"split_seed":int(seed),"outer_fold":outer_fold,"query_groups":tuple(sorted(qg)),"query_videos":tuple(sorted(qv)),
                   "train_groups":tuple(sorted(train_g)),"train_videos":tuple(sorted(train_v)),"inner":inner_rows}
