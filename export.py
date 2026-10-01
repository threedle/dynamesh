"""
export.py — write a trained arm's textured mesh out as GLB/PLY

WHY VERTEX COLOURS AND NOT A UV BAKE
  Baking into a 2048^2 atlas after an xatlas unwrap is the obvious route. This
  mesh has
  430,910 faces, which is 9.7 texels per face — the bake aliases and speckles,
  and it is what made an earlier TRELLIS.2 export look worse than TRELLIS v1
  despite the field being fine. The PBR field is sampled DIRECTLY at the mesh's
  215,462 vertices instead. That is 7x the 29,349 pixels the training camera
  actually resolves, so vertex colour is not the limiting factor here — the
  atlas was.

TWO FRAMES, ONE VERTEX ORDER (same convention as the trainer)
  v_raw  the registered mesh as exported; the GLB carries THESE positions.
  v_pp   preprocess_mesh output, normalised to [-0.5, 0.5]; the PBR voxel field
         lives in this frame, so sampling happens here.
  preprocess_mesh preserves vertex order, which is what lets the two coexist.

Writes both the frozen and the adapted mesh from the SAME noise and the SAME
conditioning, so anything that differs between them is the adapter.
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
ap.add_argument('--run', required=True, help='a run directory written by train.py')
ap.add_argument('--ckpt', default='lora_best.pt')
ap.add_argument('--frames', default='1',
                help="frame indices: '1-150' for a range, '1,5,10' for specific "
                     "frames, or mixed, e.g. '1-10,50,140-150'")
ap.add_argument('--tag', default=None)
ARGS = ap.parse_args()

RUN_DIR = Path(ARGS.run)
CFG = json.load(open(RUN_DIR / 'config.json'))

def _parse_frames(spec):
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if '-' in tok:
            a, b = tok.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out

FRAMES = _parse_frames(ARGS.frames)

# The upstream copy referenced a bare MESH that was never assigned, and then loaded a
# hardcoded TRELLIS v1 ply regardless of which run was passed. Both come from the run's
# own config.json here, so the exported mesh is the one that run was actually fit to.
MESH = CFG['mesh']
GT_DIR = CFG['gt_dir']
MCFM = CFG.get('mcfm')
if isinstance(MCFM, str) and MCFM.lower() in ('none', ''):
    MCFM = None

_so, _se = sys.stdout, sys.stderr
sys.argv = ['train.py',
            '--mesh', MESH, '--gt-dir', GT_DIR,
            '--rank', str(CFG['rank']), '--targets', CFG['targets'],
            '--seed', str(CFG['seed']), '--epochs', str(CFG['epochs']),
            '--n-frames', str(CFG['n_frames']), '--resolution', str(CFG['resolution'])]
sys.path.insert(0, str(_HERE))
import train as R                         # noqa: E402
sys.stdout, sys.stderr = _so, _se

import numpy as np
import torch
import trimesh
from PIL import Image

DEVICE = torch.device('cuda')
ARM = CFG.get('arm', '?')
TGT = '+'.join(t.replace('to_', '') for t in CFG.get('target_set', []))
# The default tag must identify the OBJECT, not just the arm: the arm number is a property
# of the arm and is identical for every object, so two objects baked without --tag
# both landed in one shared directory and the second silently overwrote the first.
# mesh stem keeps it readable, run_id keeps it unique across arms on one object,
# and in bake.py MESH is the possibly-overridden mesh so a transfer gets its own.
_DEFAULT_TAG = f"{Path(MESH).stem}_{CFG.get('run_id') or f'arm{ARM}'}_glb"
OUT = (_HERE / 'out' / (ARGS.tag or _DEFAULT_TAG)).resolve()


def log(*a):
    print(*a, flush=True)


def main():
    # Created here, not at module scope: importing this file to reuse a helper
    # must not leave an empty output directory behind.
    OUT.mkdir(parents=True, exist_ok=True)
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules.sparse.conv import config as conv_config
    from flex_gemm.ops.grid_sample import grid_sample_3d
    conv_config.FLEX_GEMM_ALGO = 'implicit_gemm_splitk'

    log('=' * 88)
    log(f'EXPORT GLB — arm{ARM} ({TGT})   frames {FRAMES}')
    log('=' * 88)

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
    v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().to(DEVICE)
    V = len(mesh_in.vertices)
    log(f'[MESH] {V:,} verts  {len(mesh_in.faces):,} faces')

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
    reg.eval()
    log(f'[LORA] {sum(p.numel() for p in reg.parameters()):,} params  '
        f'epoch={st.get("epoch")}')

    shape_slat = pipe.encode_shape_slat(mesh_pp, CFG['resolution'])
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(DEVICE)
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(DEVICE)
    ss_n = (shape_slat - ss_mean) / ss_std
    tex_std = torch.tensor(pipe.tex_slat_normalization['std'])[None].to(DEVICE)
    tex_mean = torch.tensor(pipe.tex_slat_normalization['mean'])[None].to(DEVICE)

    gt_dir = Path(R.args.gt_dir)
    allf = list(range(1, CFG['n_frames'] + 1))
    raws = [np.array(Image.open(gt_dir / f'frame_{f:04d}.png').convert('RGB')) for f in allf]
    als = [R.largest_component(r.min(axis=2) < 245) for r in raws]
    u = np.zeros_like(als[0])
    for a in als:
        u |= a
    ys, xs = np.where(u)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    bbox = (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))

    def cond_image(fi):
        i = fi - 1
        rgba = np.concatenate([raws[i], (als[i] * 255).astype(np.uint8)[..., None]], -1)
        f = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
        return Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255).astype(np.uint8))

    g = torch.Generator(device='cpu').manual_seed(CFG['seed'])
    noise = ss_n.replace(feats=torch.randn(
        ss_n.coords.shape[0], flow.in_channels - ss_n.feats.shape[1],
        generator=g).to(DEVICE))
    grid_res = CFG['resolution']

    def vert_colours(pbr):
        attrs = grid_sample_3d(pbr.feats, pbr.coords,
                               shape=torch.Size([*pbr.shape, *pbr.spatial_shape]),
                               grid=((v_pp + 0.5) * grid_res).reshape(1, -1, 3),
                               mode='trilinear')
        return attrs[..., :3].reshape(-1, 3).float().clamp(0, 1).cpu().numpy()

    def write(cols, path):
        m = trimesh.Trimesh(vertices=np.asarray(mesh_in.vertices),
                            faces=np.asarray(mesh_in.faces),
                            vertex_colors=(cols * 255).astype(np.uint8),
                            process=False)
        m.export(path)
        return path

    # MCFM. The upstream copy conditioned each frame on its own raw tokens, which is
    # the self-attention arm WITHOUT the temporal blend, so the exported mesh would not be the arm the
    # run was trained as. The trainer blends the whole cond dict before the flow sees it
    # (train.py:1498), and the blend needs the neighbouring frames, so it
    # has to be built over every frame of the clip, not over FRAMES alone.
    conds = {}
    if MCFM:
        sys.path.insert(0, str(_HERE))
        from modules.mcfm import blend_conds
        all_frames = list(range(1, int(CFG['n_frames']) + 1))
        log(f'[MCFM] {MCFM}: encoding {len(all_frames)} frames, then blending')
        with torch.no_grad():
            for fj in all_frames:
                conds[fj] = pipe.get_cond([cond_image(fj)], CFG['resolution'])['cond']
        conds = blend_conds(conds, MCFM)
        log(f'[MCFM] blended, shape {tuple(conds[FRAMES[0]].shape)}')
    else:
        log('[MCFM] none in config.json, using raw per-frame conditioning')

    for fi in FRAMES:
        with torch.no_grad():
            c = conds[fi] if MCFM else pipe.get_cond(
                [cond_image(fi)], CFG['resolution'])['cond']
            pbr_fz = dec(R.run_ode(flow, noise, c, ss_n, R.T_PAIRS)
                         * tex_std + tex_mean) * 0.5 + 0.5
            with R.lora_ctx(flow, reg):
                xl = R.run_ode(flow, noise, c, ss_n, R.T_PAIRS)
            pbr_lo = dec(xl * tex_std + tex_mean) * 0.5 + 0.5
        cf, cl = vert_colours(pbr_fz), vert_colours(pbr_lo)
        log(f'\n[FRAME {fi:04d}]  frozen mean {cf.mean():.4f}  '
            f'arm{ARM} mean {cl.mean():.4f}')
        log(f'  vertices >0.95 on all 3 channels:  frozen '
            f'{int((cf.min(1) > 0.95).sum()):,}   arm{ARM} '
            f'{int((cl.min(1) > 0.95).sum()):,}  of {V:,}')
        for ext in ('glb', 'ply'):
            log(f'  {write(cf, OUT / f"frozen_f{fi:04d}.{ext}")}')
            log(f'  {write(cl, OUT / f"arm{ARM}_{TGT}_f{fi:04d}.{ext}")}')
    log(f'\n[SAVE] {OUT}\n[DONE]')


if __name__ == '__main__':
    main()
