"""Frozen gates for the typed REBA pilot."""

TEST_SOTA = {
    "hatemm": {
        "pr_auc": 0.5938315566328208,
        "roc_auc": 0.8161837922270064,
        "within_video_roc": 0.631531717970362,
    },
    "mhclip_en": {
        "pr_auc": 0.4689062876331487,
        "roc_auc": 0.7478119495067798,
        "within_video_roc": 0.6003502776382159,
    },
    "mhclip_zh": {
        "pr_auc": 0.5060323955923413,
        "roc_auc": 0.7662739540727505,
        "within_video_roc": 0.530027895612544,
    },
    "hateclipseg": {
        "pr_auc": 0.6193710949898349,
        "roc_auc": 0.6050224699167533,
        "within_video_roc": 0.5619078936355938,
    },
}
