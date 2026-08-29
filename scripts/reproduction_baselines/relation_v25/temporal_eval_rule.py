#!/usr/bin/env python3
"""Frozen V22-compatible 1 Hz any-overlap temporal target rule."""
RULE='second_bin_[j,j+1)_is_positive_iff_any_target_span_has_start<bin_end_and_end>bin_start;target_overrides_other_harm_ignore'
def overlaps(start,end,j,duration):return start<min(j+1,duration) and end>j
