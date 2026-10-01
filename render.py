"""
render.py — turntable render of a fitted adapter against the frozen model

The adapter registry is rebuilt by importing train.py, so the registry's shape
always matches the checkpoint being loaded. Loading a checkpoint into a registry
built with different --targets raises "Unexpected key(s) in state_dict", which is
the good case; the bad case would be a silent partial load.

Views, matching every earlier run so the videos are directly comparable:
    side  --turns 0   camera FIXED at the training view, frames 1..150
    360   --turns 1   one revolution across the sequence
    720   --turns 2   two revolutions, so every surface point is seen TWICE at
                      different points in the texture evolution

--- original header follows ---

the v1 turntable renderer — the 360 turntable for the TRELLIS.2 cross-attention arm

THE POINT. Frame is PINNED and the camera orbits, so shape and texture are both
constant and anything that changes across the video is viewpoint alone. That is
the only way to see whether the edit reached surface the training camera never
saw. the decoder-side arm measured the failure mode on the decoder arm: one camera supervises
16.8% of vertices, the adapter edits 100% of them at equal strength (ratio 1.02),
and R^2 of the edit against the camera's image axis is 0.16-0.23 versus ~0.00 for
the frozen colour. It learned a projection, so it falls apart under rotation.
v1's cross-attention arm moves the adapter upstream of 30 frozen self-attention blocks and bets
that TRELLIS.2's own propagation carries the edit around the object.

EFFICIENCY. The frame is pinned, so the texture latent does not change with the
camera: the ODE runs ONCE per arm and the resulting PBR field is re-sampled at
every angle. 120 angles therefore cost 120 rasterisations, not 120 flow
integrations.

GATE-cam asserts that orbit(yaw=0, elev=0, r=2) reproduces the confirmed
front-view EXTRINSICS, so "the texture is a sticker" can never be an artefact of
a wrong camera.
"""

import argparse, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True,
                help='a run directory written by train.py')
ap.add_argument('--ckpt', default='lora_best.pt')
ap.add_argument('--mesh', default=None,
                help="which mesh to render. Defaults to the mesh recorded in the "
                     "run's config.json, and errors if neither is available. "
                     "Rendering a checkpoint against the WRONG mesh produces a "
                     "plausible video of the wrong object with every gate passing, "
                     "which is why there is no fallback here.")
ap.add_argument('--sweep', default='angle', choices=['angle', 'both'],
                help="angle = frame pinned, camera orbits (propagation at one "
                     "instant); both = frame advances 1..N AND camera turns 360 "
                     "together (propagation across the whole sequence)")
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--pin-frame', type=int, default=75)
ap.add_argument('--n-angles', type=int, default=120)
ap.add_argument('--elev', type=float, default=15.0)
ap.add_argument('--radius', type=float, default=2.0)
ap.add_argument('--res', type=int, default=518)
ap.add_argument('--mcfm', default=None,
                help="Override the blend mode instead of taking it from the run's "
                     "config.json. Needed for the frozen three-mode comparison, "
                     "where there is no trained checkpoint to read it from: the "
                     "run dir supplies only mesh/gt_dir/resolution. Accepts the "
                     "readable names (temporal_only_w3, spatial_then_temporal_w3, "
                     "joint_spatiotemporal_w3) or the legacy codes.")
ap.add_argument('--frozen-only', action='store_true',
                help='render PURE TRELLIS.2 only — no adapter, single panel')
ap.add_argument('--turns', type=float, default=1.0,
                help='how many full revolutions the camera makes across the '
                     'sequence. 1 = one 360 over the N frames; 2 = 720, i.e. '
                     'every surface point is seen TWICE at different points in '
                     'the texture evolution, which is what separates a texture '
                     'that lives on the surface from one painted on at a fixed '
                     'angle.')
ap.add_argument('--yaw0', type=float, default=0.0,
                help='CONSTANT yaw added to every frame, in degrees. With '
                     '--turns 0 this pins the camera at a single azimuth for the '
                     'whole sequence: --yaw0 0 is the training view, 90/180/270 '
                     'are views the adapter was never supervised from. That is '
                     'the comparison a rotating video cannot give you, because in '
                     'an orbit you cannot tell a texture change from a viewpoint '
                     'change.')
ap.add_argument('--texel-metrics', default=None,
                help="Write Temporal Flickering measured on the TEXELS -- the PBR "
                     "voxel field the decoder emits -- instead of on rendered "
                     "pixels. A render puts rasterisation, shading, camera "
                     "sampling and background compositing between the model and "
                     "the number, and all of those vary frame to frame for "
                     "reasons unrelated to the texture. The voxel field IS the "
                     "texture; UV baking is a downstream export that adds its own "
                     "interpolation. Voxel COORDINATES are identical every frame "
                     "because encode_shape_slat is a deterministic function of a "
                     "fixed mesh -- asserted per frame below, not assumed -- so "
                     "consecutive fields are differenceable with zero resampling. "
                     "Only meaningful with --sweep both.")
ap.add_argument('--skip-render', action='store_true',
                help='with --texel-metrics, skip rasterisation and the video '
                     'entirely: the ODE still runs per frame but nothing is drawn.')
ap.add_argument('--tag', default='orbit')
ap.add_argument('--fps', type=int, default=20)
ap.add_argument('--cond-mask', default=None,
                help="Path to a boolean .npy silhouette used as the CONDITIONING "
                     "alpha instead of the r.min(axis=2)<245 brightness threshold. "
                     "That threshold assumes a near-white backdrop; a clip shot on "
                     "a grey backdrop (penguin_circuits measures 237) puts 99.7%% of "
                     "the frame above it, the union crop degenerates to the whole "
                     "image and DINOv3 receives an out-of-distribution input, which "
                     "renders a BLACK texture. It also picks up contact shadows, "
                     "which widen the crop even at a corrected threshold. The mesh "
                     "render mask is exact, shadow-free and identical every frame.")
ARGS = ap.parse_args()

# --texel-metrics measures the ADAPTED field, and --frozen-only never runs the
# adapter: decode_both returns (frozen, None) there. Asking for both produced an
# AttributeError on None several minutes into a render. Refuse it at startup,
# where the message can say why, instead of failing after the work is done.
if ARGS.texel_metrics and ARGS.frozen_only:
    raise SystemExit('--texel-metrics needs the adapted arm, which --frozen-only '
                     'does not run. Drop one of the two.')

RUN_DIR = Path(ARGS.run)
CFG = json.load(open(RUN_DIR / 'config.json'))
MESH = ARGS.mesh or CFG.get('mesh')
if not MESH:
    raise SystemExit("no mesh: pass --mesh, or use a run whose config.json records one")
# The CONDITIONING images must come from this run's own data. The trainer used to
# carry a hard-coded --gt-dir default, so a run without one was rendered against
# another object's video: this mesh wearing that object's texture, plausible and
# completely wrong. It cost six renders. That default is gone and both sides now
# refuse rather than guess, which is what the SystemExit below preserves.
GT_DIR = CFG.get('gt_dir')
if not GT_DIR:
    raise SystemExit("run's config.json records no gt_dir; refusing to guess the "
                     "conditioning frames, which would texture this mesh from "
                     "another object's video")
print(f'[MESH]   {MESH}', flush=True)
print(f'[GT-DIR] {GT_DIR}   <- conditioning images', flush=True)

# IMPORTING THE TRAINER CREATES A RUN DIRECTORY. The trainer runs its argparse
# and mkdirs OUT at MODULE SCOPE, before main(), so a bare import lands an empty
# runs/<label>/ with a zero-byte train.log. That is where the 21 empty
# the mislabelled kv-arm shells came from — orbit renders of the neighbour-consistency arm
# rank sweep, whose fake argv omitted --lambda-con so the label fell back to 17.
# Pointing --out-dir at a scratch path stops this file adding to that pile.
_SCRATCH = Path('/tmp') / f'orbit_import_{os.getpid()}'
_so, _se = sys.stdout, sys.stderr
sys.argv = ['train.py',
            '--mesh', MESH,
            '--gt-dir', GT_DIR,
            '--out-dir', str(_SCRATCH),
            '--rank', str(CFG['rank']), '--targets', CFG['targets'],
            '--seed', str(CFG['seed']), '--epochs', str(CFG['epochs']),
            '--n-frames', str(CFG['n_frames']), '--resolution', str(CFG['resolution'])]
sys.path.insert(0, str(_HERE))
# the self-attention arm, NOT the kv arm. the kv arm's trainer.CrossAttnLoRA has no sa_qkv/sa_out, so
# load_state_dict on the self-attention arm checkpoint raises "Unexpected key(s)". Importing the
# matching trainer is what makes the registry shape line up with the checkpoint.
import train as R                          # noqa: E402
sys.stdout, sys.stderr = _so, _se

import numpy as np                          # noqa: E402
import torch                                # noqa: E402
import trimesh                              # noqa: E402
import nvdiffrast.torch as dr               # noqa: E402
from PIL import Image, ImageDraw            # noqa: E402
from flex_gemm.ops.grid_sample import grid_sample_3d   # noqa: E402

DEVICE = R.DEVICE
OUT = (_HERE / 'out' / ARGS.tag).resolve()
NEAR, FAR = 0.5, 3.0


def log(*a):
    print(*a, flush=True)


def orbit_extrinsics(yaw_deg, elev_deg, radius):
    """Camera on a sphere looking at the origin, in v1's MeshRenderer convention."""
    y, e = math.radians(yaw_deg), math.radians(elev_deg)
    eye = np.array([radius * math.cos(e) * math.sin(y),
                    -radius * math.cos(e) * math.cos(y),
                    radius * math.sin(e)], dtype=np.float64)
    fwd = -eye / np.linalg.norm(eye)
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    ext = np.eye(4)
    ext[0, :3], ext[1, :3], ext[2, :3] = right, -up, fwd
    ext[:3, 3] = -ext[:3, :3] @ eye
    return torch.tensor(ext, dtype=torch.float32)


def main():
    # Created here, not at module scope: importing this file to reuse a helper
    # must not leave an empty output directory behind.
    (OUT / 'frames').mkdir(parents=True, exist_ok=True)
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules.sparse.conv import config as conv_config
    conv_config.FLEX_GEMM_ALGO = 'implicit_gemm_splitk'

    log('=' * 88)
    log('the self-attention arm (TRELLIS.2) TURNTABLE — frozen | cross-attn + SELF-ATTN LoRA')
    log(f'  frame {ARGS.pin_frame} PINNED, {ARGS.n_angles} angles, elev {ARGS.elev}')
    log('=' * 88)

    # GATE-cam before anything expensive
    d = float((orbit_extrinsics(0.0, 0.0, 2.0) - R.EXTRINSICS).abs().max())
    log(f'[GATE-cam] |orbit(0,0,2) - confirmed EXTRINSICS| = {d:.3e}')
    assert d < 1e-5, f'GATE-cam FAILED ({d:.3e}): orbit camera != confirmed front view'
    log('[GATE-cam] PASSED')

    pipe = Trellis2TexturingPipeline.from_pretrained(
        'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
    pipe.low_vram = False
    pipe.cuda()
    flow = pipe.models[f'tex_slat_flow_model_{CFG["resolution"]}']
    dec = pipe.models['tex_slat_decoder']
    for m in pipe.models.values():
        if isinstance(m, torch.nn.Module):
            for p in m.parameters():
                p.requires_grad_(False)

    mesh_in = trimesh.load(MESH, process=False, force='mesh')
    mesh_pp = pipe.preprocess_mesh(mesh_in)
    v_raw = torch.from_numpy(np.asarray(mesh_in.vertices)).float().to(DEVICE)
    v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().to(DEVICE)
    faces = torch.from_numpy(np.asarray(mesh_in.faces)).int().to(DEVICE).contiguous()

    # alpha MUST come from the run's own config, not the default: dW is scaled by
    # alpha/rank, so rendering a rank-32 checkpoint with the wrong alpha silently
    # rescales every edit. Runs made before the rank sweep have no 'lora_alpha'
    # key; 4.0 is their implied value and at their rank 4 it gives scaling 1.0,
    # which is exactly how they were trained.
    _alpha = CFG.get('lora_alpha', 4.0)
    reg = R.LoRARegistry(len(flow.blocks), flow.model_channels, flow.cond_channels,
                         CFG['rank'], with_mlp=False,
                         mlp_hidden=int(flow.model_channels * flow.mlp_ratio),
                         targets=tuple(CFG['target_set']),
                         active=CFG.get('active'), alpha=_alpha).to(DEVICE)
    print(f'[LORA-SCALE] rank {CFG["rank"]}  alpha {_alpha}  '
          f'-> scaling {_alpha / CFG["rank"]:.4f}', flush=True)
    print(f'[REGISTRY] targets={tuple(CFG["target_set"])}  '
          f'blocks={len(CFG.get("active", range(30)))}  '
          f'{sum(p.numel() for p in reg.parameters()):,} params', flush=True)
    st = torch.load(RUN_DIR / 'ckpts' / ARGS.ckpt, map_location=DEVICE, weights_only=False)
    reg.load_state_dict(st['reg'] if 'reg' in st else st['registry_state'])
    reg.eval()
    log(f'[LORA] epoch={st.get("epoch")}  psnr={st.get("psnr", float("nan")):.3f}  '
        f'{sum(p.numel() for p in reg.parameters()):,} params on '
        f'{"+".join(CFG["target_set"])}')

    shape_slat = pipe.encode_shape_slat(mesh_pp, CFG['resolution'])
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(DEVICE)
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(DEVICE)
    ss_n = (shape_slat - ss_mean) / ss_std
    tex_std = torch.tensor(pipe.tex_slat_normalization['std'])[None].to(DEVICE)
    tex_mean = torch.tensor(pipe.tex_slat_normalization['mean'])[None].to(DEVICE)

    # Conditioning, preprocessed exactly as training did. The crop box is the
    # UNION over every frame used, so framing is identical frame to frame and
    # contributes nothing to what the video shows.
    gt_dir = Path(R.args.gt_dir)
    _fr_list = ([ARGS.pin_frame] if ARGS.sweep == 'angle'
                else list(range(1, ARGS.n_frames + 1)))
    _raws = [np.array(Image.open(gt_dir / f'frame_{f:04d}.png').convert('RGB'))
             for f in _fr_list]
    if ARGS.cond_mask:
        _m = np.load(ARGS.cond_mask).astype(bool)
        assert _m.shape == _raws[0].shape[:2], (_m.shape, _raws[0].shape)
        _als = [_m for _ in _raws]
        print(f'[COND-MASK] {ARGS.cond_mask}  fg={100*_m.mean():.1f}%', flush=True)
    else:
        _als = [R.largest_component(r.min(axis=2) < 245) for r in _raws]
    _u = np.zeros_like(_als[0])
    for a in _als:
        _u |= a
    ys, xs = np.where(_u)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    bbox = (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))

    def cond_image(i):
        rgba = np.concatenate([_raws[i], (_als[i] * 255).astype(np.uint8)[..., None]], -1)
        f = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
        return Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255).astype(np.uint8))

    cond_img = cond_image(0)

    g = torch.Generator(device='cpu').manual_seed(CFG['seed'])
    noise = ss_n.replace(feats=torch.randn(
        ss_n.coords.shape[0], flow.in_channels - ss_n.feats.shape[1],
        generator=g).to(DEVICE))

    # ── MCFM, read from the run's own config ────────────────────────────────
    # A checkpoint trained with --mcfm learned against BLENDED DINOv3 tokens. If
    # the render feeds vanilla per-frame tokens, the adapter is evaluated on an
    # input distribution it never saw and the video does not show what was
    # trained. Nothing here applied the blend, so every mcfm run rendered through
    # this file was that mismatch.
    #
    # The blend needs a WINDOW (v2_D is [t-1, t, t+1]), not one frame, so conds
    # cannot be built lazily inside decode_both as they were. They are
    # precomputed for every frame, blended once, then indexed. 150 DINOv3 encodes
    # is small against 150 ODE integrations, and ~2 MB per cond is a few hundred
    # MB held.
    #
    # CFG.get('mcfm') is absent for every non-mcfm run, so those take the lazy
    # path below exactly as before and are byte-identical to previous renders.
    # ARGS.mcfm wins over the config so a frozen run can select a mode with no
    # trained checkpoint behind it. Without the override this render would take
    # CFG's mode -- or None -- and silently produce the wrong arm under the right
    # label, which is the one failure this comparison cannot survive.
    _MCFM = ARGS.mcfm or CFG.get('mcfm')
    if ARGS.mcfm:
        log(f'[MCFM] mode OVERRIDDEN from the command line: {ARGS.mcfm} '
            f'(config said {CFG.get("mcfm")})')
    _CONDS = None
    if _MCFM:
        log(f'[MCFM] run trained with --mcfm {_MCFM}; blending conds before render')
        _n = ARGS.n_frames if ARGS.sweep != 'angle' else CFG['n_frames']
        with torch.no_grad():
            _CONDS = {i: pipe.get_cond([cond_image(i - 1)], CFG['resolution'])['cond']
                      for i in range(1, _n + 1)}
        sys.path.insert(0, str(_HERE))
        from modules.mcfm import blend_conds
        _pre = {k: v.clone() for k, v in _CONDS.items()}
        _CONDS = blend_conds(_CONDS, _MCFM)
        _d = max(float((_CONDS[k] - _pre[k]).abs().max()) for k in _CONDS)
        log(f'[MCFM] GATE-blend max|blended-vanilla| = {_d:.5f}  (must be > 0)')
        assert _d > 0, ('GATE-blend FAILED: the blend returned the vanilla tokens '
                        'unchanged, so this render would silently be the no-mcfm '
                        'arm wearing an mcfm label.')
        del _pre

    # ── the wide-context arm: WIDE CONTEXT WINDOW ──────────────────────────────────────────
    # A --context-window 3 run trained with cross-attention seeing [f-1, f, f+1]
    # concatenated (1029 -> 3087 tokens). Its checkpoint has the SAME 300 keys as
    # a plain the self-attention arm one -- the LoRA A/B matrices act on the feature dim, not the
    # token count -- so it loads clean and renders plausible frames while being
    # fed a third of the context it was trained on. No error, no warning, just a
    # silently wrong arm under the right label. Exactly the MCFM renderer gap
    # again, so the fix mirrors the wide-context arm verbatim.
    _CW = CFG.get('context_window') or 1
    if _CW > 1:
        assert _CONDS is None, (
            'context_window and mcfm are mutually exclusive -- MCFM pre-collapses '
            'the window this flag exists to hand over intact. A config carrying '
            'both is not a state the trainer can produce, so refuse it here too.')
        _n = ARGS.n_frames if ARGS.sweep != 'angle' else CFG['n_frames']
        with torch.no_grad():
            _base = {i: pipe.get_cond([cond_image(i - 1)], CFG['resolution'])['cond']
                     for i in range(1, _n + 1)}
        _keys = sorted(_base)
        _lo, _hi = _keys[0], _keys[-1]
        _half = _CW // 2
        _off = list(range(-_half, _half + 1))
        assert len(_off) == _CW, (_off, _CW)
        _n_img = int(next(iter(_base.values())).shape[-2])

        def _stack(f):
            ts = [_base[min(max(f + o, _lo), _hi)] for o in _off]
            return torch.cat(ts, dim=(1 if ts[0].dim() == 3 else 0))

        _CONDS = {f: _stack(f) for f in _keys}
        _shape = tuple(next(iter(_CONDS.values())).shape)
        assert _shape[-2] == _n_img * _CW, _shape
        log(f'[R32] context_window={_CW} offsets={_off}  '
            f'tokens {_n_img} -> {_shape[-2]}')

        # GATE-window. The CENTRE slice must be frame f byte-for-byte. If the
        # ordering were off, every frame would render conditioned on a neighbour
        # and the output would still look like a plausible dynamic texture --
        # invisible until someone diffs it against training. Checked on an
        # interior frame so the edge clamp is not what is being tested.
        _f = _keys[len(_keys) // 2]
        _c = _CONDS[_f].narrow(-2, _half * _n_img, _n_img)
        assert torch.equal(_c, _base[_f]), (
            f'GATE-window FAILED at frame {_f}: the centre slice of the stacked '
            f'cond is not frame {_f}. Frame ordering is wrong and every render '
            f'would be conditioned on the wrong neighbour.')
        log(f'[R32] GATE-window PASSED — centre slice of frame {_f} is frame {_f}')
        del _base

    def decode_both(ci, fidx=None):
        """Run the ODE for the frozen and adapted arms from the SAME noise and
        the SAME conditioning, so the only difference is the adapter.

        fidx is the 1-based frame index, used only to look up a blended cond.
        Both arms get the SAME cond — the blend is a property of the input, not
        of the adapter, so the frozen arm must see it too or the comparison
        would confound the adapter with the conditioning.
        """
        with torch.no_grad():
            if _CONDS is not None and fidx is not None:
                c = _CONDS[fidx]
            else:
                c = pipe.get_cond([ci], CFG['resolution'])['cond']
            a = dec(R.run_ode(flow, noise, c, ss_n, R.T_PAIRS) * tex_std + tex_mean) * 0.5 + 0.5
            if ARGS.frozen_only:
                return a, None          # no adapter run at all
            with R.lora_ctx(flow, reg):
                xl = R.run_ode(flow, noise, c, ss_n, R.T_PAIRS)
            b = dec(xl * tex_std + tex_mean) * 0.5 + 0.5
        return a, b


    pbr_fz = pbr_lo = None
    if ARGS.sweep == 'angle':
        # frame pinned -> the field never changes -> integrate ONCE, then the
        # angles cost rasterisation only.
        t0 = time.time()
        pbr_fz, pbr_lo = decode_both(cond_img, 1)   # cond_img is cond_image(0) -> frame 1
        log(f'[FIELD] both arms decoded once in {time.time()-t0:.1f}s — '
            f'{ARGS.n_angles} angles now cost rasterisation only')

    ctx = dr.RasterizeCudaContext()
    intr = R.INTRINSICS.to(DEVICE)
    proj = R.intrinsics_to_projection(intr, NEAR, FAR)
    res = ARGS.res
    grid_res = CFG['resolution']

    def render_at(ext, pbr):
        full = (proj @ ext.to(DEVICE)).unsqueeze(0)
        vh = torch.cat([v_raw, torch.ones_like(v_raw[:, :1])], -1).unsqueeze(0)
        clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
        rast, _ = dr.rasterize(ctx, clip, faces, (res, res))
        m = rast[0, ..., 3] > 0
        idx = torch.nonzero(m.reshape(-1)).squeeze(1)
        pos = dr.interpolate(v_pp.unsqueeze(0).contiguous(), rast, faces)[0]
        pos = pos[0].reshape(-1, 3)[idx].contiguous()
        attrs = grid_sample_3d(pbr.feats, pbr.coords,
                               shape=torch.Size([*pbr.shape, *pbr.spatial_shape]),
                               grid=((pos + 0.5) * grid_res).reshape(1, -1, 3),
                               mode='trilinear')
        rgb = attrs[..., :3].reshape(-1, 3).float().clamp(0, 1)
        base = torch.ones(res * res, 3, device=DEVICE)
        img = base.index_put((idx,), rgb).view(res, res, 3)
        return (img.cpu().numpy() * 255).astype(np.uint8)

    _rg = CFG.get('arm', '?')
    _tg = '+'.join(t.replace('to_', '') for t in CFG.get('target_set', []))
    _LORA_LABEL = f'arm{_rg} LoRA  ({_tg})'
    LAB = 28
    if ARGS.sweep == 'angle':
        items = [(0, ARGS.pin_frame, i * 360.0 / ARGS.n_angles)
                 for i in range(ARGS.n_angles)]
    else:
        n = ARGS.n_frames
        items = [(i, i + 1, i * 360.0 * ARGS.turns / n) for i in range(n)]
    # TEXEL accumulators. The trace is [T, Nvox, 3] and would be ~1.6 GB at 150
    # frames, so flicker and jerk are accumulated online against a two-frame
    # window instead of storing it.
    _tx = {'prev': None, 'prev2': None, 'first': None, 'coords': None,
           'flick': [], 'jerk': [], 'n': 0}
    # The FROZEN arm, measured in the same pass. decode_both already produces it for
    # every frame, so this costs one tensor diff and no extra ODE work -- and it is
    # the paper's baseline: pure TRELLIS.2, same mesh, same per-frame conditioning,
    # no adapter and no temporal term. Measuring it in a separate run would mean
    # re-solving every frame just to diff a field we already had in hand.
    _fz = {'prev': None, 'prev2': None, 'first': None,
           'flick': [], 'jerk': [], 'n': 0}
    for k, (ci, fr, yaw) in enumerate(items, 1):
        yaw = yaw + ARGS.yaw0            # constant azimuth offset (see --yaw0)
        ext = orbit_extrinsics(yaw, ARGS.elev, ARGS.radius)
        if ARGS.sweep == 'both':
            # items are (ci, fr, yaw) with ci 0-based and fr = ci+1, and _CONDS is
            # keyed 1..n from cond_image(i-1) — so fr indexes the blended cond that
            # corresponds to exactly this cond_image(ci).
            pbr_fz, pbr_lo = decode_both(cond_image(ci), fr)
            if ARGS.texel_metrics:
                with torch.no_grad():
                    c_now = pbr_lo.feats[:, :3].float()
                    if _tx['coords'] is None:
                        _tx['coords'] = pbr_lo.coords.clone()
                        _tx['first'] = c_now.clone()
                    else:
                        # the whole comparison rests on this: same voxels, every
                        # frame. A mismatch means the fields are not aligned and
                        # any difference between them is meaningless.
                        assert torch.equal(_tx['coords'], pbr_lo.coords), (
                            f'voxel coords changed at frame {fr} — texel differences '
                            f'would be comparing different voxels')
                    if _tx['prev'] is not None:
                        _tx['flick'].append(float((c_now - _tx['prev']).abs().mean()))
                    if _tx['prev2'] is not None:
                        _tx['jerk'].append(float(
                            (c_now - 2 * _tx['prev'] + _tx['prev2']).abs().mean()))
                    _tx['prev2'] = _tx['prev']
                    _tx['prev'] = c_now
                    _tx['n'] += 1
                    # identical arithmetic on the frozen field. Coords are shared
                    # with the adapted arm (same mesh, same voxelisation), so the
                    # coords assert above covers both.
                    f_now = pbr_fz.feats[:, :3].float()
                    if _fz['first'] is None:
                        _fz['first'] = f_now.clone()
                    if _fz['prev'] is not None:
                        _fz['flick'].append(float((f_now - _fz['prev']).abs().mean()))
                    if _fz['prev2'] is not None:
                        _fz['jerk'].append(float(
                            (f_now - 2 * _fz['prev'] + _fz['prev2']).abs().mean()))
                    _fz['prev2'] = _fz['prev']
                    _fz['prev'] = f_now
                    _fz['n'] += 1
        if ARGS.skip_render and ARGS.texel_metrics:
            continue
        with torch.no_grad():
            a_fz = render_at(ext, pbr_fz)
            a_lo = None if ARGS.frozen_only else render_at(ext, pbr_lo)
        panels = ([(a_fz, 'TRELLIS.2 (pure, no adapter)')] if ARGS.frozen_only
                  else [(a_fz, 'TRELLIS.2 frozen'), (a_lo, _LORA_LABEL)])
        canv = Image.fromarray(np.full((res + LAB, res * len(panels), 3), 18, np.uint8))
        d_ = ImageDraw.Draw(canv)
        for j, (im, lb) in enumerate(panels):
            canv.paste(Image.fromarray(im), (j * res, LAB))
            d_.rectangle([j * res, 0, (j + 1) * res - 1, LAB - 1], fill=(38, 38, 58))
            _t = (f'{lb}   yaw {yaw:5.1f}' if ARGS.sweep == 'angle'
                  else f'{lb}   frame {fr:3d}   yaw {yaw:5.1f}')
            d_.text((j * res + 8, 8), _t, fill=(240, 240, 240))
        canv.save(OUT / 'frames' / f'{k:04d}.png')
        if ARGS.sweep == 'both':
            del pbr_fz, pbr_lo
            torch.cuda.empty_cache()
        if k % 20 == 0 or k == 1:
            log(f'  {k:3d}/{len(items)}  frame={fr}  yaw={yaw:5.1f}')

    # Name the file after WHAT IT IS. This used to be a constant, so every arm
    # produced identically-named mp4s, and scp-ing three of them into
    # one directory silently left you with only the last.
    # turns=0 is the FIXED training view, not '0x360' -- name it for what it is
    # TEXEL METRICS. Written BEFORE any encoding: with --skip-render there are no
    # frames on disk, ffmpeg exits non-zero, and anything after it never runs. The
    # measurement is the point of that mode, so it must not sit downstream of a
    # video step it deliberately skipped.
    if ARGS.texel_metrics and _tx['n'] > 2:
        F = float(np.mean(_tx['flick']))
        J = float(np.mean(_tx['jerk']))
        D = float((_tx['prev'] - _tx['first']).abs().mean())
        rec = {'run': RUN_DIR.name, 'mcfm': CFG.get('mcfm'),
               'n_frames': _tx['n'], 'n_voxels': int(_tx['coords'].shape[0]),
               'texel_flicker': F, 'texel_jerk': J, 'texel_drift': D,
               'flicker_per_frame': _tx['flick'], 'jerk_per_frame': _tx['jerk']}
        if _fz['n'] and _fz['flick']:
            _fF = sum(_fz['flick']) / len(_fz['flick'])
            _fJ = sum(_fz['jerk']) / len(_fz['jerk']) if _fz['jerk'] else float('nan')
            _fD = float((_fz['prev'] - _fz['first']).abs().mean())
            rec.update({'frozen_flicker': _fF, 'frozen_jerk': _fJ, 'frozen_drift': _fD,
                        'frozen_flicker_per_frame': _fz['flick'],
                        'frozen_jerk_per_frame': _fz['jerk']})
            log(f'[TEXEL] frozen  flicker {_fF:.6f}   jerk {_fJ:.6f}   drift {_fD:.6f}')
        Path(ARGS.texel_metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(ARGS.texel_metrics).write_text(json.dumps(rec, indent=1))
        log(f"\n[TEXEL] voxels {rec['n_voxels']:,}   frames {rec['n_frames']}")
        log(f"[TEXEL] flicker {F:.6f}   jerk {J:.6f}   drift {D:.6f}")
        log(f"[TEXEL] -> {ARGS.texel_metrics}")
    if ARGS.skip_render:
        log('[DONE] --skip-render: no frames drawn, no video encoded')
        return

    _turns = ('sideview' if ARGS.turns == 0
              else f'{ARGS.turns:g}x360'.replace('1x360', '360'))
    # tau belongs in the name too: two the confidence-weighted arm arms differ ONLY by tau, and
    # without it both write the same filename and clobber each other on scp.
    _cw = CFG.get('conf_weight'); _ct = CFG.get('conf_tau')
    _tau = f'_tau{_ct:g}' if _cw else ''
    vid = OUT / f'arm{_rg}_{_tg}{_tau}_{_turns}_{ARGS.sweep}_vs_frozen.mp4'
    # ffmpeg is wherever this machine keeps it: PATH first, then the binary that
    # ships with imageio-ffmpeg, which install_environment.sh already pulls in.
    # Hardcoding /usr/bin/ffmpeg only worked on the cluster this came from.
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg is None:
        # The frames are the result; the mp4 is a convenience. Losing ffmpeg must
        # not throw away a render that has already finished.
        log(f'\n[VIDEO] skipped: no ffmpeg on PATH and imageio-ffmpeg is not installed.'
            f'\n[VIDEO] the frames are in {OUT / "frames"}\n[DONE]')
        return
    pr = subprocess.run([ffmpeg, '-encoders'], capture_output=True, text=True)
    fl = (['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
          if 'libx264' in pr.stdout else
          ['-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p'])
    subprocess.run([ffmpeg, '-y', '-framerate', str(ARGS.fps),
                    '-i', str(OUT / 'frames' / '%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', *fl, str(vid)], check=True)
    log(f'\n[VIDEO] {vid}  ({vid.stat().st_size/1e6:.1f} MB)\n[DONE]')


if __name__ == '__main__':
    main()
