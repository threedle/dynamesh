"""
modules/mcfm.py — MCFM temporal token blending, as a standalone importable module.

TRELLIS.2's own source is untouched; a caller opts in by importing blend_conds()
and passing --mcfm on its own command line.

WHAT IT DOES
  MCFM replaces each frame's image-conditioning tokens with a blend over a short
  temporal window, BEFORE the flow model sees them. Neighbouring frames then
  share conditioning, so their textures cannot drift apart independently.
  Parameter-free: nothing here is trained, and with mode=None it is a no-op.

  v2  per-position : token i of frame t attends over token i of the window only.
                     1029 independent softmaxes over W items.
  v3  joint        : frame t's N tokens query a pooled bank of W*N tokens —
                     one softmax over time AND space together.
  C   window [t, t+1]        D   window [t-1, t, t+1]

VALIDATED IN TRELLIS 1 FIRST
  The same four modes were run on spot_lava and teapot_lava2 through TRELLIS 1
  (experiments/enhancement/step06b_asset.py, 8 runs, 150 frames each) and the
  renders confirmed the blend produces coherent moving texture. The tokens do NOT
  transfer — TRELLIS 1 is DINOv2 (1374, 1024), TRELLIS.2 is DINOv3 (1029, 1024) —
  but MCFM has no learned parameters, so only the operation carries over.

N IS NEVER HARDCODED
  TRELLIS 1's step4_mcfm/mcfm.py pins N_TOKENS = 1374, which is a DINOv2-at-518
  number. Here N and D are read from the tensor, so 512 (1029 tokens) and 1024
  (4101 tokens) both work with no change.

MEMORY
  v2 stacks (N, W, D); v3 forms an (N, W*N) attention matrix — at resolution 1024
  with window D that is 4101 x 12303, about 202 MB in fp32. Both are negligible
  against the flow model's own footprint (measured 10.6 GiB at 512, 39.2 GiB at
  1024 for frozen inference).
"""

from typing import Dict, Optional

import torch

MODES = ('v2_C', 'v2_D', 'v2_E', 'v2_F', 'v2_G', 'v2_H', 'v2_I',
         'v3_C', 'v3_D', 'v3_E',
         'ts_C', 'ts_D', 'ts_E',      # temporal -> explicit spatial
         'st_C', 'st_D', 'st_E')      # explicit spatial -> temporal
# C..I are WINDOW WIDTHS, 2 / 3 / 5 / 7 / 11 / 13 / 15 frames. E was added for the window ablation
# (W = 1, 3, 5): W=1 is no blend at all, i.e. mcfm=None, so it needs no offsets
# entry. blend_window() reads W from the stacked tensor and never assumes 2 or 3,
# and blend_conds() clamps out-of-range neighbours to the nearest existing frame,
# so t-2 at frame 1 resolves to frame 1 exactly as t-1 already did.
_OFFSETS = {'C': (0, 1), 'D': (-1, 0, 1), 'E': (-2, -1, 0, 1, 2),
            # F/G added for the window ablation W = 7 / 11. Symmetric like D and E,
            # so the middle entry is always frame t and blend_conds' clamping at the
            # sequence ends behaves identically -- only the width changes.
            'F': (-3, -2, -1, 0, 1, 2, 3),
            'G': (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5),
            # H/I added for the W = 13 / 15 extension of the same ablation. Symmetric
            # like D..G, so offsets.index(0) is still the middle and blend_conds'
            # end-clamping is unchanged. At W=15 a 150-frame sequence still has 143
            # interior frames whose window needs no clamping at all, so the curve is
            # not dominated by edge effects; the 121-frame object (pumpkin_rot) has
            # 114. Only the width changes.
            'H': (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6),
            'I': (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7)}
# WIDTH IS READ FROM THE TENSOR, NOT FROM THIS TABLE. blend_window stacks the list it
# is given and never assumes 2/3/5, and blend_conds clamps out-of-range neighbours to
# the nearest existing frame, so a 7- or 11-frame window at frame 1 resolves exactly
# the way a 3-frame one already did. Nothing else needed changing to support them.

# ---------------------------------------------------------------- readable names
# "v2_D" says nothing about what the operator does. These spell it out:
#
#   temporal_only            token j attends over token j in neighbouring frames
#                            ONLY. No spatial term -- spatial mixing is left to the
#                            model's own cross-attention downstream. (= v2)
#   joint_spatiotemporal     all W*N tokens pooled into one softmax, so token j can
#                            be replaced by token i != j from another frame. (= v3)
#   spatial_then_temporal    spatial self-attention first, then temporal. (= v2b)
#   temporal_then_spatial    the other ordering. Both are implemented here; an
#                            earlier version of this file rejected them and that
#                            comment outlived the implementation (NOT_IMPLEMENTED
#                            is empty below).
#
#   _w2  window [t, t+1]            (= C)
#   _w3  window [t-1, t, t+1]       (= D)
#   _w11 window [t-5 .. t+5]        (= G)  <- what the paper reports
#
# ALIASES, not a rename. The short code is what reaches the config and the run
# directory name, so a run started as "temporal_only_w3" hashes identically to one
# started as "v2_D" and can resume from its checkpoints. Renaming outright would
# have changed every hash and orphaned every existing run.
ALIASES = {
    'temporal_only_w2':         'v2_C',
    'temporal_only_w3':         'v2_D',
    'temporal_only_w5':         'v2_E',
    'temporal_only_w7':         'v2_F',
    'temporal_only_w11':        'v2_G',
    'temporal_only_w13':        'v2_H',
    'temporal_only_w15':        'v2_I',
    'joint_spatiotemporal_w2':  'v3_C',
    'joint_spatiotemporal_w3':  'v3_D',
    'joint_spatiotemporal_w5':  'v3_E',
}
ALIASES.update({
    'temporal_then_spatial_w2': 'ts_C',
    'temporal_then_spatial_w3': 'ts_D',
    'temporal_then_spatial_w5': 'ts_E',
    'spatial_then_temporal_w2': 'st_C',
    'spatial_then_temporal_w3': 'st_D',
    'spatial_then_temporal_w5': 'st_E',
})
NOT_IMPLEMENTED = {}      # all four orderings are implemented here now
PRETTY = {v: k for k, v in ALIASES.items()}


def canonical(mode):
    """Map any accepted spelling to the short code used everywhere downstream.

    Call this ONCE at argument-parse time so the config, the run-directory hash and
    the checkpoint all see the same string no matter which name was typed.
    """
    if mode is None:
        return None
    if mode in NOT_IMPLEMENTED:
        raise ValueError(
            f'{mode!r} (= {NOT_IMPLEMENTED[mode]}) is not implemented here. '
            f'Spatial-then-temporal exists only in TRELLIS 1 '
            f'(experiments/dynamic_texture_trellis_pipeline/step4_mcfm/mcfm.py, '
            f'mcfm_v2b) and has never been run on DINO conditioning tokens. '
            f'Port it before selecting it, and check its stage-1 self-weight: with '
            f'large token norms that softmax saturates and v2b degenerates to v2.')
    return ALIASES.get(mode, mode)


def parse_mode(mode: str):
    """'v2_D' or 'temporal_only_w3' -> ('v2', (-1, 0, 1), 1)."""
    mode = canonical(mode)
    if mode not in MODES:
        raise ValueError(
            f'--mcfm must be one of {MODES} or {tuple(ALIASES)}, got {mode!r}')
    variant, window = mode.split('_')
    offsets = _OFFSETS[window]
    return variant, offsets, offsets.index(0)


def blend_window(window_toks, t_pos: int, variant: str) -> torch.Tensor:
    """window_toks: list of (N, D) tensors, current frame at index t_pos."""
    cur = window_toks[t_pos]                                  # (N, D)
    D = cur.shape[-1]
    scale = D ** -0.5
    stack = torch.stack(window_toks, dim=1)                   # (N, W, D)
    if variant == 'v2':
        q = cur.unsqueeze(1)                                  # (N, 1, D)
        attn = torch.softmax(torch.bmm(q, stack.transpose(1, 2)) * scale, dim=-1)
        return torch.bmm(attn, stack).squeeze(1)              # (N, D)
    if variant == 'ts':          # temporal, THEN an explicit spatial stage
        q = cur.unsqueeze(1)
        at = torch.softmax(torch.bmm(q, stack.transpose(1, 2)) * scale, dim=-1)
        kv = torch.bmm(at, stack).squeeze(1)                  # (N, D)
        sp = torch.softmax(torch.mm(cur, kv.T) * scale, dim=-1)
        return torch.mm(sp, kv)                               # (N, D)
    if variant == 'st':          # explicit spatial stage, THEN temporal
        sp = torch.softmax(torch.mm(cur, cur.T) * scale, dim=-1)
        refined = torch.mm(sp, cur)                           # (N, D)
        q = refined.unsqueeze(1)
        at = torch.softmax(torch.bmm(q, stack.transpose(1, 2)) * scale, dim=-1)
        return torch.bmm(at, stack).squeeze(1)                # (N, D)
    pool = stack.reshape(-1, D)                               # (W*N, D)
    attn = torch.softmax(torch.mm(cur, pool.T) * scale, dim=-1)
    return torch.mm(attn, pool)                               # (N, D)


@torch.no_grad()
def blend_conds(conds: Dict[int, torch.Tensor],
                mode: Optional[str],
                verbose: bool = True) -> Dict[int, torch.Tensor]:
    """Blend a {frame_index -> cond} dict in place of the vanilla tokens.

    conds values may be (N, D) or (1, N, D); the batch dim is preserved so the
    result is a drop-in replacement for whatever the caller already had.

    Frames outside the available range are CLAMPED to the nearest existing frame
    (frame 1's t-1 is frame 1), so the first and last frames blend over a
    truncated window rather than wrapping or dropping out.

    mode=None returns conds unchanged, so a caller can leave the call site in
    place and select vanilla behaviour from the command line.
    """
    if mode is None:
        if verbose:
            print('[MCFM] disabled — vanilla tokens unchanged', flush=True)
        return conds

    variant, offsets, t_pos = parse_mode(mode)
    keys = sorted(conds.keys())
    lo, hi = keys[0], keys[-1]
    present = set(keys)

    def _at(i, centre):
        # Walk AWAY from the centre frame, never across it. `i` is an absolute
        # frame index and `centre` is the frame whose window is being built, so
        # the comparison has to be between those two. Comparing i against t_pos
        # compared a frame index (1..150) with a window position (0..W-1), which
        # is True for every frame past the middle of the window: the search then
        # always ran forward and could return a frame on the far side of the
        # centre, putting it in the window twice and dropping the near neighbour.
        j = min(max(i, lo), hi)
        while j not in present and lo < j < hi:      # tolerate gaps in the index
            j += 1 if i > centre else -1
        return conds[j if j in present else keys[0]]

    out = {}
    for fi in keys:
        win = []
        for o in offsets:
            t = _at(fi + o, fi)
            win.append(t[0] if t.dim() == 3 else t)
        blended = blend_window(win, t_pos, variant)
        ref = conds[fi]
        out[fi] = blended.unsqueeze(0) if ref.dim() == 3 else blended

    if verbose:
        ref, new = conds[keys[0]], out[keys[0]]
        delta = (new.float() - ref.float()).abs().max().item()
        print(f'[MCFM] mode={mode} variant={variant} window={offsets} '
              f'frames={len(keys)} shape={tuple(new.shape)}', flush=True)
        print(f'[MCFM] GATE-blend max|blended-vanilla| at frame {keys[0]} = '
              f'{delta:.5f}  (must be > 0, else the blend is a no-op)', flush=True)
        assert delta > 0, 'GATE-blend FAILED: blended tokens identical to vanilla'
    return out
