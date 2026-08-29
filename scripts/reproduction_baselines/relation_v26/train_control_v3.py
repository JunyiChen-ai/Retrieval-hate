#!/usr/bin/env python3
"""V26 kill-pilot trainer with the pre-result variable-length control migration.

The frozen reference continues to use the audited v2 loader.  This trainer only
changes the permuted control: a canonical cyclic derangement moves the donor's
complete (T, X, masks, own-OOF-background) tuple without temporal resampling.
"""
import argparse, copy, json, math
from pathlib import Path

import torch

import train as v2
from artifacts import atomic, sha
from core import CTW, DESIGN_SHA, MIGRATION_SHA, ARCH, ch, fold, tensor_ch, availability_counts, intervention_count

SCHEMA = "v26_finite_rf_train_run_v3_variable_length_control"
CKPT_SCHEMA = "v26_finite_rf_checkpoint_v3_variable_length_control"
CONTROL_SCHEMA = "v26_variable_length_derangement_v1"


def source_sha():
    return sha(__file__)


def permutation_v3(rows):
    recipients = sorted(rows, key=lambda r: r["id"])
    donors = recipients[1:] + recipients[:1]
    out, mapping = [], []
    for recipient, donor in zip(recipients, donors):
        moved = {
            **recipient,
            "T": donor["T"],
            "X": donor["X"],
            "masks": donor["masks"],
            "oof_b": donor["oof_b"],
        }
        out.append(moved)
        mapping.append({
            "recipient": recipient["id"],
            "recipient_T": recipient["T"],
            "donor": donor["id"],
            "donor_fold": fold(donor["id"]),
            "donor_T": donor["T"],
            "nonself": recipient["id"] != donor["id"],
            "raw_sha": tensor_ch(donor["X"]),
            "mask_sha": tensor_ch(donor["masks"]),
            "b_sha": tensor_ch(donor["oof_b"]),
            "availability": availability_counts(donor["masks"]),
            "intervention_coverage": intervention_count(donor["masks"]),
        })
    return out, mapping


def save_ckpt(path, model, seed, arm, epoch, steps, inputs):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "schema": CKPT_SCHEMA,
        "design_sha256": DESIGN_SHA,
        "migration_sha256": MIGRATION_SHA,
        "architecture": ARCH,
        "control_schema": CONTROL_SCHEMA,
        "trainer_sha256": source_sha(),
        "seed": seed,
        "arm": arm,
        "epoch": epoch,
        "steps": steps,
        "inputs": inputs,
        "state": state,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os, tempfile
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    os.close(fd)
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return {"path": str(path.resolve()), "sha256": sha(path)}


def train_arm(rows, arm, seed, out, inputs, epochs, initial_state, device):
    v2.seed_all(seed)
    model = CTW(model_seed=seed)
    model.load_state_dict(copy.deepcopy(initial_state))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    checkpoints = [save_ckpt(out / "epoch0.pt", model, seed, arm, 0, 0, inputs)]
    steps, use = 0, rows
    if arm == "permuted":
        use, mapping = permutation_v3(rows)
        donors = [x["donor"] for x in mapping]
        pre_T = sorted(r["T"] for r in rows)
        post_T = sorted(r["T"] for r in use)
        pre_av = sorted(tuple(int(x.sum()) for x in r["masks"]) for r in rows)
        post_av = sorted(tuple(int(x.sum()) for x in r["masks"]) for r in use)
        pre_tuples = sorted((r["T"], tensor_ch(r["X"]), tensor_ch(r["masks"]), tensor_ch(r["oof_b"])) for r in rows)
        post_tuples = sorted((r["T"], tensor_ch(r["X"]), tensor_ch(r["masks"]), tensor_ch(r["oof_b"])) for r in use)
        valid = (
            len(rows) > 1
            and all(x["nonself"] for x in mapping)
            and sorted(donors) == sorted(r["id"] for r in rows)
            and pre_T == post_T
            and pre_av == post_av
            and pre_tuples == post_tuples
            and all(x["intervention_coverage"] == x["donor_T"] for x in mapping)
        )
        if not valid:
            raise RuntimeError("variable-length permutation coverage")
        atomic(out / "permutation.json", {
            "schema": CONTROL_SCHEMA,
            "design_sha256": DESIGN_SHA,
            "migration_sha256": MIGRATION_SHA,
            "trainer_sha256": source_sha(),
            "mapping": mapping,
            "video_fraction": 1.0,
            "instance_fraction": 1.0,
            "donor_bijection": True,
            "T_multiset_sha256": ch(pre_T),
            "availability_multiset_sha256": ch(pre_av),
            "tuple_multiset_sha256": ch(pre_tuples),
        })
    for epoch in range(1, epochs + 1):
        for start in range(0, len(use), 4):
            losses = []
            for row in use[start:start + 4]:
                replacement = [torch.zeros_like(x) for x in row["X"]] if arm == "negative_mean" else row["oof_b"]
                loss, _, _ = v2.ctw_loss(model, row, replacement)
                losses.append(loss)
            opt.zero_grad()
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
        checkpoints.append(save_ckpt(out / f"epoch{epoch}.pt", model, seed, arm, epoch, steps, inputs))
    return checkpoints, steps


def train_probe(rows, out, inputs, device):
    v2.seed_all(26027)
    model = v2.Probe().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(8):
        for start in range(0, len(rows), 4):
            losses = [torch.nn.functional.binary_cross_entropy_with_logits(model(r["X"], r["masks"]), r["y"]) for r in rows[start:start + 4]]
            opt.zero_grad()
            torch.stack(losses).mean().backward()
            opt.step()
    return save_ckpt(Path(out) / "probe.pt", model, 26027, "probe", 8, math.ceil(len(rows) / 4) * 8, inputs)


def run(features, labels, reference, out, seed=234, epochs=8):
    rows, feature_manifest = v2.load_rows(features, labels)
    raw_reference = json.load(open(reference))
    verified_reference = v2.verify_reference(reference, features, labels, raw_reference["inputs"]["val_features"]["path"])
    v2.attach_oof(rows, verified_reference)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    row_sha = ch([(r["id"], r["T"], tensor_ch(r["X"]), tensor_ch(r["masks"]), tensor_ch(r["oof_b"]), float(r["G"]), float(r["y"])) for r in rows])
    inputs = {
        "features_sha256": sha(features),
        "features_root_sha256": feature_manifest["root_sha256"],
        "labels_sha256": sha(labels),
        "reference_manifest_sha256": sha(reference),
        "rows_sha256": row_sha,
        "core_sha256": sha(Path(__file__).with_name("core.py")),
        "base_loader_sha256": sha(Path(__file__).with_name("train.py")),
        "trainer_sha256": source_sha(),
        "design_sha256": DESIGN_SHA,
        "migration_sha256": MIGRATION_SHA,
        "architecture": ARCH,
        "control_schema": CONTROL_SCHEMA,
    }
    atomic(out / "fallback.json", {"schema": "v26_fallback_v1", "scores": {r["id"]: float(r["G"]) for r in rows}, "raw_G_bit_exact": True})
    device = v2.cuda5090()
    v2.preload(rows, device)
    canonical = CTW(model_seed=seed).state_dict()
    epoch0_hash = ch({k: v.tolist() for k, v in canonical.items()})
    arms, steps = {}, {}
    for arm in ("real", "permuted", "negative_mean"):
        arms[arm], steps[arm] = train_arm(rows, arm, seed, out / arm, inputs, epochs, canonical, device)
    probe = train_probe(rows, out, inputs, device)
    manifest = {
        "schema": SCHEMA,
        "design_sha256": DESIGN_SHA,
        "migration_sha256": MIGRATION_SHA,
        "architecture": ARCH,
        "control_schema": CONTROL_SCHEMA,
        "trainer_sha256": source_sha(),
        "epoch0_state_sha256": epoch0_hash,
        "seed": seed,
        "epochs": list(range(epochs + 1)),
        "arms": arms,
        "steps": steps,
        "matched_steps": len(set(steps.values())) == 1,
        "probe": probe,
        "features": {"path": str(Path(features).resolve()), "sha256": sha(features), "root_sha256": feature_manifest["root_sha256"]},
        "labels": {"path": str(Path(labels).resolve()), "sha256": sha(labels)},
        "reference": {"path": str(Path(reference).resolve()), "sha256": sha(reference)},
        "rows_sha256": row_sha,
        "test_read": False,
    }
    atomic(out / "manifest.json", manifest)
    verify_train_run(out / "manifest.json")
    return manifest


def verify_train_run(path):
    manifest = json.load(open(path))
    required = {"schema", "design_sha256", "migration_sha256", "architecture", "control_schema", "trainer_sha256", "epoch0_state_sha256", "seed", "epochs", "arms", "steps", "matched_steps", "probe", "features", "labels", "reference", "rows_sha256", "test_read"}
    if set(manifest) != required or manifest["schema"] != SCHEMA or manifest["trainer_sha256"] != source_sha() or manifest["control_schema"] != CONTROL_SCHEMA or manifest["test_read"] is not False or not manifest["matched_steps"]:
        raise RuntimeError("v3 train manifest")
    for arm, checkpoints in manifest["arms"].items():
        for epoch, entry in enumerate(checkpoints):
            if sha(entry["path"]) != entry["sha256"]:
                raise RuntimeError("v3 checkpoint bytes")
            checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=False)
            if checkpoint["schema"] != CKPT_SCHEMA or checkpoint["trainer_sha256"] != source_sha() or checkpoint["arm"] != arm or checkpoint["epoch"] != epoch or checkpoint["inputs"]["control_schema"] != CONTROL_SCHEMA:
                raise RuntimeError("v3 checkpoint schema")
            if epoch == 0 and ch({k: v.tolist() for k, v in checkpoint["state"].items()}) != manifest["epoch0_state_sha256"]:
                raise RuntimeError("v3 epoch0 mismatch")
    if sha(manifest["probe"]["path"]) != manifest["probe"]["sha256"]:
        raise RuntimeError("v3 probe")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=234, choices=(234, 2025, 3407))
    parser.add_argument("--epochs", type=int, default=8, choices=(8,))
    parser.add_argument("--device", required=True, choices=("cuda",))
    args = parser.parse_args()
    run(args.features, args.labels, args.reference, args.out, args.seed, args.epochs)


if __name__ == "__main__":
    main()
