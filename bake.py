"""bake.py — one mesh.obj plus one texture PNG per frame.

WHY THIS INSTEAD OF A GLB PER FRAME
  export.py writes a vertex-coloured mesh per frame. The geometry is
  identical in all of them, so 150 frames of one object cost 150 copies of the same
  215k vertices: 8.2 MB a frame, 2.5 GB an object, ~100 GB for the 42. Only the
  colour changes, so only the colour needs to be written per frame.

  The bake is lossless here, which is not true in general: the rendered colour is
  attrs[..., :3] sampled at the surface point, with no normal, view vector or
  specular term anywhere in SurfaceRenderer.sample or render_at. A texel therefore
  carries everything a pixel could, and any renderer can consume the result.

HOW
  1. Unwrap ONCE with xatlas, so every frame
     bakes into the same layout and the maps stay comparable texel to texel.
  2. Per frame, rasterise in UV space: the "camera" is the atlas itself, so each
     texel gets the 3D surface position that lands on it, and grid_sample_3d reads
     the field there. This is render_at with UV coordinates standing in for a view.
  3. Write mesh.obj once, then tex_f0001.png ... tex_f0150.png.

ATLAS SIZE. A 2048^2 atlas over this 430k-face mesh is 9.7 texels
a face, and it aliased and speckled; that is why export.py went to vertex
colours in the first place. Default here is 4096 and --atlas is exposed, and
--check renders one frame both ways and prints the difference so the choice is
measured rather than assumed.

    python bake.py --run runs/<label> --frames 1-150
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Paths resolve through modules/paths.py, never to one machine's filesystem.
from modules.paths import add_trellis2_to_path, hf_home, set_offline_from_env

hf_home()
set_offline_from_env()
add_trellis2_to_path()
_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--ckpt', default='lora_best.pt')
ap.add_argument('--frames', default='1',
                help="frame indices: '1-150' for a range, '1,5,10' for specific "
                     "frames, or mixed, e.g. '1-10,50,140-150'")
ap.add_argument('--atlas', type=int, default=4096)
ap.add_argument('--dilate', type=int, default=8,
                help='texels of island-border padding; 0 disables')
ap.add_argument('--tag', default=None)
# TRANSFER BAKES. Figure 5 applies an adapter fitted on one object to meshes it never saw,
# so the geometry and the conditioning frames are NOT the ones in the run's config.json.
# The transfer renders recorded both in their logs (MESH=, COND=), and those are passed
# back in here. Without them a transfer bake would silently re-bake the training object.
ap.add_argument('--mesh', default=None, help='override CFG["mesh"] (figure 5 transfers)')
ap.add_argument('--gt-dir', dest='gt_dir', default=None, help='override CFG["gt_dir"]')
ap.add_argument('--cond-cache', dest='cond_cache', default=None,
                help='read the per-frame conditioning from this file instead of computing it. '
                     'With it, neither gated model (DINOv3, RMBG) is loaded at all, so the demo '
                     'runs without HuggingFace access to them.')
ap.add_argument('--write-cond-cache', dest='write_cond_cache', default=None,
                help='compute the conditioning for every frame and write it here, for others to '
                     'reuse through --cond-cache. Needs the gated models, so we run this, not the reader.')
ap.add_argument('--arm', default='adapted', choices=['adapted', 'frozen', 'both'])
ap.add_argument('--check', action='store_true',
                help='bake frame 1 and compare against the vertex-colour path')
ARGS = ap.parse_args()

# --run may be absolute, or relative to this repository. Accept either.
from modules.paths import REPO_ROOT as _DYNA
RUN_DIR = Path(ARGS.run)
if not (RUN_DIR / 'config.json').exists() and (_DYNA / ARGS.run / 'config.json').exists():
    RUN_DIR = _DYNA / ARGS.run
# Every check below runs before the pipeline, the unwrap and the atlas, so a typo costs a
# second rather than the minutes of GPU work it used to.
if not (RUN_DIR / 'config.json').exists():
    ap.error(f"no config.json under --run {ARGS.run!r}. Point --run at a run directory, "
             f"for example ./demo/spot_lava/run or a directory train.py wrote.")
CFG = json.load(open(RUN_DIR / 'config.json'))

def _parse_frames(spec):
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            if '-' in tok:
                a, b = tok.split('-')
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(tok))
        except ValueError:
            ap.error(f'--frames got {tok!r}, which is not a frame number or a range. '
                     f'Use forms like 1, 1-150, or 1,5,9-12.')
    return out

FRAMES = _parse_frames(ARGS.frames)
_HAVE = int(CFG.get('n_frames') or 0)
_OUT_OF_RANGE = [f for f in FRAMES if f < 1 or (_HAVE and f > _HAVE)]
if _OUT_OF_RANGE:
    ap.error(f'--frames asks for {_OUT_OF_RANGE[:5]}{"..." if len(_OUT_OF_RANGE) > 5 else ""}, '
             f'but this run has frames 1-{_HAVE}. Baking would stop partway through.')

# Required for every arm, --arm frozen included: the registry is built and loaded before the
# arm is chosen, so a missing file stops the bake either way. Saying so here beats failing late.
if not (RUN_DIR / 'ckpts' / ARGS.ckpt).exists():
    _have = sorted(p.name for p in (RUN_DIR / 'ckpts').glob('*.pt')) if (RUN_DIR / 'ckpts').is_dir() else []
    ap.error(f"no checkpoint {ARGS.ckpt!r} under {RUN_DIR / 'ckpts'}. "
             + (f"This run has: {', '.join(_have)}" if _have else 'That directory holds no .pt file.'))
MESH = ARGS.mesh or CFG['mesh']
GT_DIR = ARGS.gt_dir or CFG['gt_dir']
if ARGS.mesh:
    print(f'[TRANSFER] mesh overridden: {MESH}', flush=True)
    assert MESH != CFG['mesh'], (
        'the --mesh override equals the run\'s own mesh, so this is not a transfer; '
        'drop --mesh rather than label a training-object bake as one')
if ARGS.gt_dir:
    print(f'[TRANSFER] conditioning overridden: {GT_DIR}', flush=True)
MCFM = CFG.get('mcfm')
if isinstance(MCFM, str) and MCFM.lower() in ('none', ''):
    MCFM = None

# The trainer module is imported for its LoRARegistry, lora_ctx, run_ode, T_PAIRS and
# largest_component. It MUST be the one that trained the run: the kv arm's trainer, which
# the upstream exporter imports, rejects '--targets qkvo+sa' outright, because the
# self-attention target set only exists from the self-attention arm on. Import the matching trainer so the self-attention arm
# checkpoint is rebuilt by the self-attention arm's own registry.
_so, _se = sys.stdout, sys.stderr
_TRAINER = 'train'   # this release ships the self-attention arm trainer only
sys.argv = [f'{_TRAINER}.py',
            '--mesh', MESH, '--gt-dir', GT_DIR,
            '--rank', str(CFG['rank']), '--targets', CFG['targets'],
            '--seed', str(CFG['seed']), '--epochs', str(CFG['epochs']),
            '--n-frames', str(CFG['n_frames']), '--resolution', str(CFG['resolution'])]
if CFG.get('gt_render_dir'):
    sys.argv += ['--gt-render-dir', CFG['gt_render_dir']]
sys.path.insert(0, str(_HERE))
R = __import__(_TRAINER)
sys.stdout, sys.stderr = _so, _se
print(f'[TRAINER] {_TRAINER} (arm {CFG.get("arm")})', flush=True)

import numpy as np
import torch
import trimesh
from PIL import Image

DEVICE = torch.device('cuda')
ARM = CFG.get('arm', '?')
# The default tag must identify the OBJECT, not just the arm: the arm number is a property
# of the arm and is identical for every object, so two objects baked without --tag
# both landed in one shared directory and the second silently overwrote the first.
# mesh stem keeps it readable, run_id keeps it unique across arms on one object,
# and in bake.py MESH is the possibly-overridden mesh so a transfer gets its own.
_DEFAULT_TAG = f"{Path(MESH).stem}_{CFG.get('run_id') or f'arm{ARM}'}"
OUT = (_HERE / 'out' / (ARGS.tag or _DEFAULT_TAG)).resolve()


def log(*a):
    print(*a, flush=True)


def unwrap(v, f):
    """xatlas unwrap. Returns (vmap, faces, uv).

    xatlas SPLITS vertices at seams, so the returned vmap is longer than the input
    vertex list and idx indexes into it. v[vmap] is therefore the unwrapped mesh's
    positions, and every seam copy keeps the position of the vertex it came from,
    which is what lets one 3D position serve several texels.
    """
    import xatlas
    vmap, idx, uv = xatlas.parametrize(np.asarray(v, dtype=np.float32),
                                       np.asarray(f, dtype=np.uint32))
    log(f'[UV] {len(v):,} verts -> {len(vmap):,} after seam splits, '
        f'{len(idx):,} faces, uv range [{uv.min():.3f}, {uv.max():.3f}]')
    return vmap, idx.astype(np.int64), uv.astype(np.float32)


def main():
    # Created here, not at module scope: importing this file to reuse a helper
    # must not leave an empty output directory behind.
    OUT.mkdir(parents=True, exist_ok=True)
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules.sparse.conv import config as conv_config
    from flex_gemm.ops.grid_sample import grid_sample_3d
    import nvdiffrast.torch as dr
    conv_config.FLEX_GEMM_ALGO = 'implicit_gemm_splitk'

    log('=' * 88)
    log(f'BAKE UV — arm{ARM}  mcfm={MCFM}  atlas {ARGS.atlas}^2  frames {FRAMES}')
    log('=' * 88)

    COND_CACHE = None
    if ARGS.cond_cache:
        _blob = torch.load(ARGS.cond_cache, map_location='cpu')
        assert _blob['resolution'] == CFG['resolution'], (
            f"cache was built at resolution {_blob['resolution']}, this run is "
            f"{CFG['resolution']}")
        COND_CACHE = _blob['conds']
        log(f'[COND] cache {ARGS.cond_cache}: {len(COND_CACHE)} frames, '
            f"{_blob.get('dtype', '?')}, from {_blob.get('model', '?')}")
        # The conditioning is the only thing those two models produce here, so with a cache
        # they are dead weight -- and they are the gated downloads. Replacing the classes the
        # pipeline config names keeps trellis2/ untouched while never fetching them.
        from trellis2.modules import image_feature_extractor as _ife
        from trellis2.pipelines import rembg as _rembg

        class _NotLoaded:
            """Stands in for a model this run never calls, so it is never downloaded."""

            def __init__(self, *a, **k):
                pass

            def __call__(self, *a, **k):
                raise RuntimeError('this run reads conditioning from --cond-cache; '
                                   'the image model was never loaded')

            def to(self, *a, **k):
                return self

            def cpu(self, *a, **k):
                return self

        for _mod in (_ife, _rembg):
            for _name in dir(_mod):
                _obj = getattr(_mod, _name)
                if isinstance(_obj, type) and not _name.startswith('_'):
                    setattr(_mod, _name, _NotLoaded)

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
    assert len(mesh_pp.vertices) == len(mesh_in.vertices), 'vertex order broken'
    v_raw = np.asarray(mesh_in.vertices)
    v_pp = np.asarray(mesh_pp.vertices)
    faces = np.asarray(mesh_in.faces)
    log(f'[MESH] {len(v_raw):,} verts  {len(faces):,} faces  <- {MESH}')

    vmap, uv_faces, uv = unwrap(v_raw, faces)
    # v_pp follows the SAME split, because vmap indexes the original vertex list and
    # preprocess_mesh preserves vertex order. The atlas therefore carries the frame the
    # PBR field lives in, which is the frame grid_sample_3d needs.
    pp_split = torch.from_numpy(v_pp[vmap]).float().to(DEVICE)

    # rasterise the atlas ONCE: UV is the camera. Clip space wants [-1,1], and y is
    # flipped so texel row 0 is uv v=1, matching how PNGs are written and read.
    ctx = dr.RasterizeCudaContext()
    uvc = torch.from_numpy(uv).float().to(DEVICE)
    clip = torch.cat([uvc * 2.0 - 1.0,
                      torch.zeros_like(uvc[:, :1]),
                      torch.ones_like(uvc[:, :1])], dim=-1).unsqueeze(0)
    clip[..., 1] *= -1.0
    tri = torch.from_numpy(uv_faces).int().to(DEVICE).contiguous()
    A = ARGS.atlas
    rast, _ = dr.rasterize(ctx, clip.contiguous(), tri, (A, A))
    covered = (rast[0, ..., 3] > 0)
    idx_tex = torch.nonzero(covered.reshape(-1)).squeeze(1)
    pos = dr.interpolate(pp_split.unsqueeze(0).contiguous(), rast, tri)[0]
    pos = pos[0].reshape(-1, 3)[idx_tex].contiguous()
    pct = 100.0 * idx_tex.numel() / (A * A)
    log(f'[ATLAS] {A}x{A}, {idx_tex.numel():,} texels covered ({pct:.1f}% of the map), '
        f'{idx_tex.numel() / max(len(faces), 1):.1f} texels per face')

    # ── the trained adapter ────────────────────────────────────────────────────
    # alpha MUST come from the run's own config, not the default: dW is scaled by
    # alpha/rank, so loading a rank-32 checkpoint with the wrong alpha silently
    # rescales every edit. Runs made before the rank sweep have no 'lora_alpha'
    # key; 4.0 is their implied value and at their rank 4 it gives scaling 1.0,
    # which is exactly how they were trained.
    # active MUST come from it too: omitted, LoRARegistry registers every block,
    # so a --blocks early/mid/late checkpoint no longer matches the registry.
    reg = R.LoRARegistry(len(flow.blocks), flow.model_channels, flow.cond_channels,
                         CFG['rank'], with_mlp=False,
                         mlp_hidden=int(flow.model_channels * flow.mlp_ratio),
                         targets=tuple(CFG['target_set']),
                         active=CFG.get('active'),
                         alpha=CFG.get('lora_alpha', 4.0)).to(DEVICE)
    st = torch.load(RUN_DIR / 'ckpts' / ARGS.ckpt, map_location=DEVICE, weights_only=False)
    reg.load_state_dict(st['reg'] if 'reg' in st else st['registry_state'])
    log(f"[CKPT] {ARGS.ckpt}  epoch {st.get('epoch', '?')}")

    # Copied verbatim from export.py:124-127, which is the working path.
    # There is no pipe.get_shape_slat; the encode is explicit and the normalisation
    # constants come out of pipe.shape_slat_normalization.
    shape_slat = pipe.encode_shape_slat(mesh_pp, CFG['resolution'])
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(DEVICE)
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(DEVICE)
    ss_n = (shape_slat - ss_mean) / ss_std
    tex_std = torch.tensor(pipe.tex_slat_normalization['std'])[None].to(DEVICE)
    tex_mean = torch.tensor(pipe.tex_slat_normalization['mean'])[None].to(DEVICE)

    # CONDITIONING IS THE VIDEO FRAME, NOT THE TARGET. The trainer reads
    # args.gt_dir / f'frame_{fi:04d}.png' (train.py:1454) and uses
    # gt_render_dir / f'gt_{fi:04d}.png' only as the loss target. export.py
    # reads gt_*.png out of gt_dir, which conditions the flow on the supervision
    # target instead of the driving video; that is not what the run was trained on.
    gt_dir = Path(R.args.gt_dir)
    raws = [np.asarray(Image.open(gt_dir / f'frame_{i:04d}.png').convert('RGB'))
            for i in range(1, int(CFG['n_frames']) + 1)]
    als = [R.largest_component(r.min(axis=2) < 245) for r in raws]
    u = np.zeros_like(als[0]); [np.logical_or(u, a, out=u) for a in als]
    ys, xs = np.where(u)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    bbox = (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))

    def cond_image(fi):
        i = fi - 1
        rgba = np.concatenate([raws[i], (als[i] * 255).astype(np.uint8)[..., None]], -1)
        f = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
        return Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255).astype(np.uint8))

    def _cond(fi):
        """Conditioning for frame fi, from the cache when one was given."""
        if COND_CACHE is not None:
            return COND_CACHE[fi].to(DEVICE, torch.float32)
        return pipe.get_cond([cond_image(fi)], CFG['resolution'])['cond']

    g = torch.Generator(device='cpu').manual_seed(CFG['seed'])
    noise = ss_n.replace(feats=torch.randn(
        ss_n.coords.shape[0], flow.in_channels - ss_n.feats.shape[1],
        generator=g).to(DEVICE))
    grid_res = CFG['resolution']

    # ── MCFM, over the whole clip, as the trainer does ────────────────────────
    conds = {}
    if MCFM or ARGS.write_cond_cache:
        allf = list(range(1, int(CFG['n_frames']) + 1))
        log(f'[COND] encoding {len(allf)} frames'
            + (f', then blending with {MCFM}' if MCFM else ''))
        with torch.no_grad():
            for fj in allf:
                conds[fj] = _cond(fj)
        if ARGS.write_cond_cache:
            # raw, before any blend: the blend is cheap and belongs to the config, not the cache
            _dst = Path(ARGS.write_cond_cache)
            _dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'resolution': CFG['resolution'],
                        'model': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
                        'dtype': 'float16',
                        'gt_dir': str(GT_DIR),
                        'conds': {k: v.detach().half().cpu() for k, v in conds.items()}}, _dst)
            log(f'[COND] wrote {len(conds)} frames -> {_dst} '
                f'({_dst.stat().st_size / 1e6:.0f} MB)')
        if MCFM:
            sys.path.insert(0, str(_HERE))
            from modules.mcfm import blend_conds
            conds = blend_conds(conds, MCFM)
    else:
        log('[MCFM] none in config.json, raw per-frame conditioning')

    def field(fi, adapted):
        with torch.no_grad():
            c = conds[fi] if MCFM else _cond(fi)
            if adapted:
                with R.lora_ctx(flow, reg):
                    x = R.run_ode(flow, noise, c, ss_n, R.T_PAIRS)
            else:
                x = R.run_ode(flow, noise, c, ss_n, R.T_PAIRS)
            return dec(x * tex_std + tex_mean) * 0.5 + 0.5

    def bake(pbr):
        """field -> RGB atlas, with the island borders padded.

        DILATION IS NOT COSMETIC. Rasterising the atlas covers triangle INTERIORS, so
        every island edge is one texel of colour against uncovered white. Any renderer
        filtering that texture bilinearly, Blender included, mixes the white in and
        draws a bright seam along every UV cut. Measured here: 19% of vertices sit on
        a texel the rasteriser never covered. Each pass grows the covered region by
        one texel into the white, which is the standard fix and is why bakers emit
        padded atlases.
        """
        attrs = grid_sample_3d(pbr.feats, pbr.coords,
                               shape=torch.Size([*pbr.shape, *pbr.spatial_shape]),
                               grid=((pos + 0.5) * grid_res).reshape(1, -1, 3),
                               mode='trilinear')
        rgb = attrs[..., :3].reshape(-1, 3).float().clamp(0, 1)
        tex = torch.zeros(A * A, 3, device=DEVICE).index_put((idx_tex,), rgb)
        tex = tex.reshape(A, A, 3)
        filled = torch.zeros(A * A, device=DEVICE).index_put(
            (idx_tex,), torch.ones_like(idx_tex, dtype=torch.float32)).reshape(A, A)
        # max-pool dilation: a 3x3 max over both colour and coverage carries real
        # colour outward and never pulls the zeros back in.
        t = tex.permute(2, 0, 1).unsqueeze(0)
        f = filled.unsqueeze(0).unsqueeze(0)
        for _ in range(ARGS.dilate):
            t_d = torch.nn.functional.max_pool2d(t, 3, stride=1, padding=1)
            f_d = torch.nn.functional.max_pool2d(f, 3, stride=1, padding=1)
            grow = (f_d > 0) & (f == 0)
            t = torch.where(grow, t_d, t)
            f = torch.where(grow, f_d, f)
        tex = t.squeeze(0).permute(1, 2, 0)
        # whatever is still unreached becomes white, matching the render background
        tex = torch.where((f.squeeze(0).squeeze(0) > 0).unsqueeze(-1),
                          tex, torch.ones_like(tex))
        return (tex.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    def vert_colours(pbr):
        attrs = grid_sample_3d(pbr.feats, pbr.coords,
                               shape=torch.Size([*pbr.shape, *pbr.spatial_shape]),
                               grid=((torch.from_numpy(v_pp).float().to(DEVICE) + 0.5)
                                     * grid_res).reshape(1, -1, 3),
                               mode='trilinear')
        return attrs[..., :3].reshape(-1, 3).float().clamp(0, 1).cpu().numpy()

    # ── the mesh, written ONCE ────────────────────────────────────────────────
    obj_path = OUT / 'mesh.obj'
    m = trimesh.Trimesh(vertices=v_raw[vmap], faces=uv_faces, process=False)
    m.visual = trimesh.visual.TextureVisuals(
        uv=uv.astype(np.float64), material=trimesh.visual.material.SimpleMaterial())
    m.export(obj_path)
    log(f'[MESH] {obj_path}  ({obj_path.stat().st_size / 1048576:.1f} MB, written once)')

    arms = ['adapted', 'frozen'] if ARGS.arm == 'both' else [ARGS.arm]
    for fi in FRAMES:
        for arm in arms:
            pbr = field(fi, adapted=(arm == 'adapted'))
            tex = bake(pbr)
            name = f'tex_f{fi:04d}.png' if arm == 'adapted' else f'tex_frozen_f{fi:04d}.png'
            Image.fromarray(tex).save(OUT / name)
            log(f'  [{arm:7s}] frame {fi:4d} -> {name}  '
                f'{(OUT / name).stat().st_size / 1024:.0f} KB  mean {tex.mean():.1f}')
            if ARGS.check and arm == 'adapted':
                # GATE-bake: the atlas must agree with the vertex-colour path it
                # replaces. Compared at the VERTICES, where both are defined: read the
                # baked texel each vertex's uv lands on, against the field sampled at
                # that vertex directly.
                # Compare at FACE CENTROIDS, not at vertices. A vertex sits exactly on an
                # island boundary, so its uv rounds to a texel the rasteriser may never
                # have covered; measured, 19% of them do, and those read as background
                # and swamped the mean at 0.150 even though the interior was fine.
                # A centroid is interior by construction.
                fc_uv = uv[uv_faces].mean(axis=1)
                fc_pp = v_pp[vmap][uv_faces].mean(axis=1)
                a_ = grid_sample_3d(
                    pbr.feats, pbr.coords,
                    shape=torch.Size([*pbr.shape, *pbr.spatial_shape]),
                    grid=((torch.from_numpy(fc_pp).float().to(DEVICE) + 0.5)
                          * grid_res).reshape(1, -1, 3), mode='trilinear')
                ref = a_[..., :3].reshape(-1, 3).float().clamp(0, 1).cpu().numpy()
                u_ = np.clip((fc_uv[:, 0] * (A - 1)).astype(int), 0, A - 1)
                v_ = np.clip(((1 - fc_uv[:, 1]) * (A - 1)).astype(int), 0, A - 1)
                baked = tex[v_, u_].astype(np.float32) / 255.0
                d = np.abs(baked - ref).mean()
                log(f'  [GATE-bake] mean |atlas - field| at {len(ref):,} face centroids '
                    f'= {d:.5f} ({"OK" if d < 0.02 else "TOO HIGH"})')
    log(f'\n[SAVE] {OUT}\n[DONE]')



if __name__ == '__main__':
    main()
