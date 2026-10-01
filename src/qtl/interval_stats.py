"""Additive content statistics on a forest, including interval complements.

The two additive passes are promoted from the associative IO prototype. Its
active searches retain their frozen implementation until completion.
"""
import torch


def sum_inside_outside(flat, levels, include_outside=True):
    """Sum arbitrary leaf statistics; levels must already be on flat.device.

    flat has padded batch rows as its first dimension. Only leaf_row entries
    participate, so padding contributes neither values nor counts.
    """
    inside = [None] * len(levels)
    for depth in range(len(levels) - 1, -1, -1):
        lev, pieces = levels[depth], []
        if lev["n_leaf"]:
            pieces.append(flat[lev["leaf_row"]])
        if len(lev["nodes"]) > lev["n_leaf"]:
            pieces.append(inside[depth + 1][lev["L"]] + inside[depth + 1][lev["R"]])
        inside[depth] = torch.cat(pieces, 0)
    outside = [torch.zeros_like(inside[0])]
    for depth, lev in enumerate(levels[:-1]):
        state = torch.zeros_like(inside[depth + 1])
        if include_outside:
            parent = outside[depth][lev["n_leaf"]:]
            state = (state.index_copy(0, lev["L"], parent + inside[depth + 1][lev["R"]])
                     .index_copy(0, lev["R"], parent + inside[depth + 1][lev["L"]]))
        outside.append(state)
    return inside, outside
