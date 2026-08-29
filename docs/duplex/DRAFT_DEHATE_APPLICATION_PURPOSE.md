# Draft DeHate Dataset Application Purpose

Status: **DRAFT ONLY — DO NOT SUBMIT WITHOUT USER/PI REVIEW**

This file intentionally contains no applicant, PI, affiliation, or email information and does not accept the dataset terms on anyone's behalf.

## Short form

We request access to DeHate for non-commercial academic research on weakly supervised temporal localization of explicit and implicit hateful content in videos. We will train models using only video-level labels from a preregistered training split, select configurations using a validation split, and evaluate temporal localization on a self-sealed test split. Our study will compare audio, speech/text, and visual evidence while preventing test annotations from entering model training or selection. We will use the dataset only within the approved research team, comply with the DeHate terms and the source platforms' policies, and report only aggregate research results permitted by the data agreement. We will not redistribute the dataset, media, annotations, or restricted derived artifacts.

## Expanded form

The proposed project studies weakly supervised hateful video localization: predicting frame- or segment-level hateful-content scores while training only from coarse video-level labels. DeHate is valuable because it covers both explicit and implicit hate, two source platforms, and segment-level multimodal annotations.

We plan to establish a deterministic, label-independent 70/10/20 train/validation/test split, with exact duplicate media kept in the same split. Training will use only training videos and their video-level labels. Temporal annotations from the validation split will be used for model selection, and test annotations will be isolated behind an aggregate evaluator until model code, checkpoint, and configuration have been frozen. We intend to report aggregate frame-level average precision and ROC-AUC, uncertainty intervals, media-coverage statistics, and—where applicable—aggregate segment-overlap metrics.

The work is solely for non-commercial academic research. Access will be limited to authorized researchers who independently satisfy the dataset terms. We will not redistribute original videos, annotations, dataset subsets, or restricted derived data. Code released for reproducibility will contain no DeHate content or dataset-derived artifacts and will require users to obtain their own authorized access. We will follow applicable institutional ethics procedures and TikTok/BitChute terms, use access-controlled storage, and maintain an internal acquisition/evaluation access ledger.

## Items requiring human confirmation before submission

- Applicant name, position, affiliation, institutional email, and Google Drive Gmail.
- PI/supervisor name and email.
- Institutional ethics or sensitive-content handling requirements.
- Whether all collaborators will apply independently.
- Written clarification, if necessary, that internal feature extraction and publication of aggregate localization metrics are permitted under the prohibition on sharing derived data.
- Final wording approved by the applicant and PI.

