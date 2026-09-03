"""Re-export of src/hier_evidence_common.py (promoted there verbatim on
2026-09-04 when a second experiment needed the same datasets; CLAUDE.md: no
second copy). Column layout and classes documented in that module."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from hier_evidence_common import (  # noqa: E402,F401
    TEXT_ROOT, TEXT_DIM, SCAF_DIM, COL_ELL, COL_PS, COL_BF, COL_BC, COL_PH,
    COL_BLOCK, N_INPUT_SCAF, A_EXT_DIM, SCAF_OFFSET, text_path, load_text_rows,
    scaffold_rows, ScaffoldCache, TrainDataset, EvalDataset)
