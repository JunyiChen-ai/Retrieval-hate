#!/usr/bin/env python3
"""Build fixed bilingual BERT directions for POWA primitive grounding."""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


PROMPTS = {
    "en": [
        ("hateful abusive insulting or threatening language", "neutral respectful language"),
        ("a protected identity group targeted by race religion gender or nationality", "no person or group is targeted"),
        ("physical violence injury killing or violent threats", "peaceful nonviolent content"),
        ("sexual abuse explicit sexual content or sexual harassment", "nonsexual ordinary content"),
        ("self harm suicide or encouragement to hurt oneself", "safe content without self harm"),
        ("hateful words quoted reported or condemned for context", "hateful words directly endorsed by the speaker"),
    ],
    "zh": [
        ("仇恨、辱骂、侮辱或威胁性的语言", "中立且尊重的语言"),
        ("以种族、宗教、性别或国籍等受保护身份群体为目标", "没有针对任何个人或群体"),
        ("身体暴力、伤害、杀戮或暴力威胁", "和平且非暴力的内容"),
        ("性虐待、露骨性内容或性骚扰", "普通且不涉及性的内容"),
        ("自残、自杀或鼓励伤害自己", "安全且不涉及自残的内容"),
        ("为了引用、报道或谴责而提及仇恨言论", "说话者直接支持仇恨言论"),
    ],
}
MODELS = {"en": "bert-base-uncased", "zh": "bert-base-chinese"}


def encode(model_id, texts):
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_pretrained(model_id, add_pooling_layer=False,
                                      local_files_only=True).eval()
    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, max_length=64,
                    return_tensors="pt")
        return model(**batch).last_hidden_state[:, 0].float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/reproduction/powa_macil/semantic_prototypes.npz")
    args = ap.parse_args()
    arrays = {}
    for lang, pairs in PROMPTS.items():
        flat = [text for pair in pairs for text in pair]
        vec = encode(MODELS[lang], flat).reshape(len(pairs), 2, -1)
        direction = vec[:, 0] - vec[:, 1]
        direction /= np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-8)
        arrays[lang] = direction.astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(args.out, {k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    main()
