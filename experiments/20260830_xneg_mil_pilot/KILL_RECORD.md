# C3 killed pre-training (2026-08-30)

The mechanism premise was measured directly before spending the runs: applying
the frozen selection rule to TEST hate videos (frame GT available; selection is
feature-only so this is a fair probe) gives pseudo-negative precision AT OR
BELOW the per-corpus benign base rate:

| corpus | selected | precision(benign) | base benign rate |
|---|---:|---:|---:|
| hatemm | 2574 | .394 | .419 |
| mhclip_en | 370 | .197 | .233 |
| mhclip_zh | 39 | .154 | .374 |
| hateclipseg | 3547 | .413 | .411 |

Cross-video kNN proximity to the benign-video frame bank does not identify
benign seconds inside hateful videos — on MHC-ZH it preferentially selects
HATEFUL seconds. Consistent with the skyline finding: the within-video
discriminative signal in these features is temporal-contextual, not pointwise;
pointwise appearance proximity cannot carry the supervision.

Verdict: supervising kNN-selected seconds to 0 would inject noise at best and
anti-signal on ZH. Candidate killed without training. The review of xneg_mil.py
was still in flight when this measurement landed; its outcome does not change
the verdict.

## Follow-up probe: external hate-speech text classifier — also dead

cardiffnlp/twitter-roberta-base-hate-latest over per-second ASR context
([t-2,t+3)), TEST hate videos, within-ROC macro: hatemm .4885 (85),
mhclip_en .4262 (44), hateclipseg .4797 (67) — at or below chance, and below
the trivial has_speech arm (.5468/.5282/.5081). External text classification
of ASR does not order seconds either (probe_text_teacher.md).

Shared root cause with C1's kill (VLM window teacher .578/.514 < .60): neither
pointwise appearance nor coarse 16-s VLM windows order seconds within hate
videos. What DOES order them: temporal-context models with frame supervision
(skyline .75/.77 on HateMM/EN). The next candidate must supply an ordering
signal that (a) is per-second or finer, (b) carries semantic content beyond
frozen-appearance proximity. Leading option: external hate-speech TEXT
classifier over per-second ASR (speech is the dominant hate carrier per
MACIL-SD audio dominance), probed next.
