"""Retrieval-consensus segment denoising: E-step pseudo-label assignment.

DESIGN_iter3 SS2 (retrieval-as-annotator). For every auto sub-clip s of a train
video v (video-level label Y_v, NO segment gold labels), we assign a pseudo-label
ROLE from the agreement between Y_v and a per-sub-clip "vote" yhat_s:

  * seg_mode=consensus : yhat_s = similarity-weighted kNN vote of the VIDEO-level
    labels of the whole-video labelled memory (train set), queried with the
    sub-clip embedding in the CURRENT fused space (round 0: raw frozen-CLIP
    space; later EM rounds: the trained head's fused space).
  * seg_mode=selfscore : yhat_s = the model's OWN hate score sigmoid(logit) on
    the sub-clip (MIST / C2FPL-style WSVAD pseudo-labelling; the make-or-break
    control: same agreement table / margin / training pipeline, ONLY the label
    source differs).

Agreement table (margin m_s >= tau required for every confident role):

  Y_v=1, yhat=1  -> ROLE_POS      high-confidence hateful segment (positive)
  Y_v=1, yhat=0  -> ROLE_DRIFT    suspected noisy MIL positive == benign
                                  background segment of a hateful video;
                                  NEVER a positive; optionally used as the
                                  within-video drifting hard negative
  Y_v=0, yhat=0  -> ROLE_NEG      high-confidence benign segment (negative)
  Y_v=0, yhat=1  -> ROLE_CONFLICT confusable benign segment; ignored by
                                  default (optional extra hard negative)
  otherwise      -> ROLE_IGNORE   low margin; only whole-video term applies
"""

import numpy as np
import torch
import faiss

ROLE_IGNORE = -1
ROLE_NEG = 0
ROLE_POS = 1
ROLE_DRIFT = 2
ROLE_CONFLICT = 3

ROLE_NAMES = {
    ROLE_IGNORE: "ignore",
    ROLE_NEG: "neg",
    ROLE_POS: "pos",
    ROLE_DRIFT: "drift",
    ROLE_CONFLICT: "conflict",
}


def _l2n(x):
    return torch.nn.functional.normalize(x.float(), p=2, dim=1)


def _encode_video_fused(model, img_feats, text_feats, args, batch_size=512):
    """Whole-video fused embeddings through the head (eval, no grad). -> cpu [N, proj]"""
    model.eval()
    outs = []
    n = img_feats.shape[0]
    with torch.no_grad():
        for s in range(0, n, batch_size):
            _, e = model(
                img_feats[s: s + batch_size].to(args.device).float(),
                text_feats[s: s + batch_size].to(args.device).float(),
                return_embed=True,
            )
            outs.append(e.detach().cpu())
    return torch.cat(outs, dim=0)


def _knn_vote(query, memory, mem_labels, own_parent, topk):
    """Similarity-weighted kNN vote of video-level labels.

    query      [S, D] cpu float tensor (L2-normalised)
    memory     [N, D] cpu float tensor (L2-normalised)
    mem_labels [N]    numpy int (video-level labels of the memory)
    own_parent [S]    numpy int (row of each query's OWN parent video in memory;
                      excluded from its vote to avoid self-label leakage)
    Returns vote_hate [S] in [0,1] (weighted fraction of hateful neighbours).
    """
    index = faiss.IndexFlatIP(memory.shape[1])
    index.add(np.ascontiguousarray(memory.numpy().astype("float32")))
    k = min(topk + 1, memory.shape[0])  # +1 so we can drop the own parent
    D, I = index.search(np.ascontiguousarray(query.numpy().astype("float32")), k)
    S = query.shape[0]
    vote = np.zeros(S, dtype=np.float64)
    for i in range(S):
        w_sum, w_hate, used = 0.0, 0.0, 0
        for j in range(k):
            if used == topk:
                break
            cand = int(I[i, j])
            if cand == int(own_parent[i]):
                continue
            w = max(float(D[i, j]), 0.0) + 1e-8
            w_sum += w
            if int(mem_labels[cand]) == 1:
                w_hate += w
            used += 1
        vote[i] = w_hate / w_sum if w_sum > 0 else 0.5
    return vote


def assign_roles(video_labels, vote, valid, tau):
    """Agreement table -> roles/margins.

    video_labels [S] numpy int, vote [S] numpy float in [0,1],
    valid [S] numpy bool (decodable sub-clips), tau = margin threshold.
    Returns (roles LongTensor [S], margins FloatTensor [S]).
    """
    S = len(vote)
    yhat = (vote >= 0.5).astype(np.int64)
    margin = np.abs(2.0 * vote - 1.0)
    roles = np.full(S, ROLE_IGNORE, dtype=np.int64)
    conf = valid & (margin >= tau)
    yv = np.asarray(video_labels).astype(np.int64)
    roles[conf & (yv == 1) & (yhat == 1)] = ROLE_POS
    roles[conf & (yv == 1) & (yhat == 0)] = ROLE_DRIFT
    roles[conf & (yv == 0) & (yhat == 0)] = ROLE_NEG
    roles[conf & (yv == 0) & (yhat == 1)] = ROLE_CONFLICT
    return (
        torch.as_tensor(roles, dtype=torch.long),
        torch.as_tensor(margin, dtype=torch.float32),
    )


def summarize_roles(roles, tag=""):
    roles_np = roles.numpy() if torch.is_tensor(roles) else np.asarray(roles)
    counts = {name: int((roles_np == r).sum()) for r, name in ROLE_NAMES.items()}
    total = len(roles_np)
    print("[consensus]{} roles over {} sub-clips: ".format(tag, total)
          + "  ".join("{}={}".format(k, v) for k, v in counts.items()))
    return counts


def flip_rate(prev_roles, new_roles):
    prev = prev_roles.numpy() if torch.is_tensor(prev_roles) else np.asarray(prev_roles)
    new = new_roles.numpy() if torch.is_tensor(new_roles) else np.asarray(new_roles)
    return float((prev != new).mean())


def consensus_estep(segment_cache, train_set, model, args):
    """E-step for seg_mode=consensus.

    Base (consensus_space=clip, default -- pre-W5 behaviour, bit-for-bit):
    model=None -> round 0: vote in the raw frozen-CLIP space (per-modality
    L2-normalised concat, re-normalised). model!=None -> vote in the current
    head's fused space (per-EM-round index rebuild).
    Memory = whole-video TRAIN samples with their video-level labels.

    W5 (consensus_space=archive|blend): the vote runs (partly) in the MLLM
    structured-archive CLIP-text space (segment_cache["archive_feats"],
    [N, Da], aligned to the train cache order). Every sub-clip queries with
    its PARENT video's archive vector -- the W2 attribution showed the clip-
    space vote was de facto video-level relabelling anyway, so this is the
    honest explicit form, and the archive text CAN see speech / on-screen-
    text evidence that the frame space is blind to.
      archive: memory/query = archive vectors only (round-invariant E-step).
      blend  : key = l2n([l2n(base) | a*l2n(archive)]) -> vote similarity
               (cos_base + a^2*cos_archive)/(1+a^2), a=args.consensus_space_alpha.
    """
    sub_img = segment_cache["subclip_img_feats"].float()      # [S, Dv]
    parents = segment_cache["subclip_parent"].long()          # [S]
    inherited = segment_cache["labels"].long()                # [S] == Y_v
    vid_img = train_set.image_feats.float()                   # [N, Dv]
    vid_txt = train_set.text_feats.float()                    # [N, Dt]
    vid_labels_np = (
        train_set.labels.numpy() if torch.is_tensor(train_set.labels)
        else np.asarray(train_set.labels)
    ).astype(np.int64)
    parent_txt = vid_txt.index_select(0, parents)             # [S, Dt]

    space_mode = str(getattr(args, "consensus_space", "clip"))
    if space_mode != "clip":
        arc = segment_cache.get("archive_feats", None)
        if arc is None:
            raise ValueError(
                "consensus_space='{}' requires segment_cache['archive_feats'] "
                "(loaded in run_rac.py from the train archive cache)".format(
                    space_mode))
        arc_vid = _l2n(arc.float())                            # [N, Da]
        arc_sub = arc_vid.index_select(0, parents)             # [S, Da]

    if space_mode == "archive":
        # Pure archive space: no base-space encoding needed; identical vote
        # every EM round (the archive is frozen), so flip rate after round 1
        # is 0 by construction.
        memory, query = arc_vid, arc_sub
        space = "archive-text"
    else:
        if model is None:
            base_mem = _l2n(torch.cat([_l2n(vid_img), _l2n(vid_txt)], dim=1))
            base_query = _l2n(torch.cat([_l2n(sub_img), _l2n(parent_txt)], dim=1))
            base_name = "raw-CLIP"
        else:
            from utils.retrieval import _encode_subclip_fused
            base_mem = _l2n(_encode_video_fused(model, vid_img, vid_txt, args).cpu())
            base_query = _l2n(_encode_subclip_fused(model, sub_img, parent_txt, args).cpu())
            base_name = "fused-head"
        if space_mode == "clip":
            memory, query, space = base_mem, base_query, base_name
        else:  # blend
            a = float(getattr(args, "consensus_space_alpha", 1.0))
            memory = _l2n(torch.cat([base_mem, a * arc_vid], dim=1))
            query = _l2n(torch.cat([base_query, a * arc_sub], dim=1))
            space = "blend({}+archive,a={})".format(base_name, a)

    vote = _knn_vote(
        query, memory, vid_labels_np, parents.numpy(),
        topk=int(getattr(args, "consensus_topk", 10)),
    )
    valid = (sub_img.sum(dim=1) != 0).numpy()
    roles, margins = assign_roles(
        inherited.numpy(), vote, valid, float(getattr(args, "consensus_margin", 0.2)))
    summarize_roles(roles, tag=" ({} space, topk={}, tau={})".format(
        space, getattr(args, "consensus_topk", 10), getattr(args, "consensus_margin", 0.2)))
    return roles, margins


def selfscore_init(segment_cache):
    """Round-0 warm start for seg_mode=selfscore: inherited video labels with full
    confidence (the standard MIST-style stage-1 weak training), so that the first
    M-step has a trained scorer to draw self pseudo-labels from."""
    sub_img = segment_cache["subclip_img_feats"].float()
    inherited = segment_cache["labels"].long().numpy()
    valid = (sub_img.sum(dim=1) != 0).numpy()
    roles = np.where(inherited == 1, ROLE_POS, ROLE_NEG)
    roles = np.where(valid, roles, ROLE_IGNORE).astype(np.int64)
    margins = np.ones(len(roles), dtype=np.float32)
    roles_t = torch.as_tensor(roles, dtype=torch.long)
    summarize_roles(roles_t, tag=" (selfscore warm start = inherited labels)")
    return roles_t, torch.as_tensor(margins)


def selfscore_estep(segment_cache, train_set, model, args, batch_size=512):
    """E-step for seg_mode=selfscore: yhat_s comes from the model's OWN hate
    score on the sub-clip; same agreement table / margin threshold as consensus.
    This is the MIST/C2FPL-style control: NO retrieval neighbours involved.

    Calibration: raw sigmoids of a head trained with VIDEO-level BCE cluster
    tightly around 0.5 at the sub-clip granularity, which would make an absolute
    margin threshold vacuous (everything ROLE_IGNORE). Following the WSVAD
    score-ranking practice (MIST top-k selection), the score is rank-normalised
    to its global percentile over all (valid) train sub-clips, so vote in [0,1]
    is scale-free and tau keeps the same semantics as in consensus."""
    sub_img = segment_cache["subclip_img_feats"].float()
    parents = segment_cache["subclip_parent"].long()
    inherited = segment_cache["labels"].long()
    vid_txt = train_set.text_feats.float()
    parent_txt = vid_txt.index_select(0, parents)

    model.eval()
    probs = []
    n = sub_img.shape[0]
    with torch.no_grad():
        for s in range(0, n, batch_size):
            logit, _ = model(
                sub_img[s: s + batch_size].to(args.device),
                parent_txt[s: s + batch_size].to(args.device),
                return_embed=True,
            )
            probs.append(torch.sigmoid(logit.reshape(-1)).detach().cpu())
    score = torch.cat(probs).numpy().astype(np.float64)

    valid = (sub_img.sum(dim=1) != 0).numpy()
    # global rank-percentile over valid sub-clips (average-rank-free simple form)
    vote = np.full(len(score), 0.5, dtype=np.float64)
    vidx = np.nonzero(valid)[0]
    if len(vidx) > 1:
        order = np.argsort(score[vidx], kind="stable")
        ranks = np.empty(len(vidx), dtype=np.float64)
        ranks[order] = np.arange(len(vidx), dtype=np.float64)
        vote[vidx] = ranks / (len(vidx) - 1)

    roles, margins = assign_roles(
        inherited.numpy(), vote, valid, float(getattr(args, "consensus_margin", 0.2)))
    summarize_roles(roles, tag=" (selfscore rank-normalised, tau={})".format(
        getattr(args, "consensus_margin", 0.2)))
    return roles, margins
