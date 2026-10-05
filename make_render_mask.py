"""
make_render_mask.py — rasterise a mesh's silhouette through the pipeline camera.

make_targets.py needs `render_mask.npy`: our own silhouette at --res, produced
by the SAME mesh + camera the training loop rasterises. For earlier objects this came
out of an earlier render step; for any new object it has to be generated.

Writes out/<tag>/render_mask.npy plus a PNG to eyeball.
"""
import argparse, math
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image

_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--mesh', required=True)
ap.add_argument('--res', type=int, default=518)
ap.add_argument('--tag', required=True)
ap.add_argument('--raw', action='store_true',
                help='rasterise the mesh as-is instead of applying '
                     'preprocess_mesh normalisation (centre, uniform scale into '
                     '[-0.5,0.5], axis swap). The training loop rasterises v_raw, '
                     'so use this only if the mesh is already in that frame.')
A = ap.parse_args()

DEVICE = 'cuda'
_FX_N = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 1., 0., 2.],
                           [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX_N, 0., 0.5],
                           [0., _FX_N, 0.5],
                           [0., 0., 1.]], dtype=torch.float32)
NEAR, FAR = 0.5, 3.0


def intrinsics_to_projection(intr, near, far):
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    r = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    r[0, 0] = 2 * fx; r[1, 1] = 2 * fy
    r[0, 2] = 2 * cx - 1; r[1, 2] = -2 * cy + 1
    r[2, 2] = far / (far - near); r[2, 3] = near * far / (near - far); r[3, 2] = 1.0
    return r


def preprocess(v):
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    v = (v - (vmin + vmax) / 2) * (0.99999 / (vmax - vmin).max())
    tmp = v[:, 1].copy(); v[:, 1] = -v[:, 2]; v[:, 2] = tmp
    return v


def main():
    import nvdiffrast.torch as dr
    m = trimesh.load(A.mesh, process=False, force='mesh')
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces)
    if not A.raw:
        V = preprocess(V)
    print(f'[MESH] {Path(A.mesh).name}  verts {len(V):,}  faces {len(F):,}  '
          f'{"raw" if A.raw else "normalised"}')
    print(f'  bbox min {np.round(V.min(0),4)}  max {np.round(V.max(0),4)}')

    ctx = dr.RasterizeCudaContext()
    v = torch.from_numpy(V).float().to(DEVICE)
    f = torch.from_numpy(F).int().to(DEVICE).contiguous()
    full = (intrinsics_to_projection(INTRINSICS.to(DEVICE), NEAR, FAR)
            @ EXTRINSICS.to(DEVICE)).unsqueeze(0)
    vh = torch.cat([v, torch.ones_like(v[:, :1])], -1).unsqueeze(0)
    clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
    rast, _ = dr.rasterize(ctx, clip, f, (A.res, A.res))
    mask = (rast[0, ..., 3] > 0).cpu().numpy()

    out = _HERE / 'out' / A.tag
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'render_mask.npy', mask)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out / 'render_mask.png')
    ys, xs = np.nonzero(mask)
    print(f'[MASK] {int(mask.sum()):,} px at {A.res}^2 ({100*mask.mean():.2f}%)  '
          f'bbox ({xs.min()},{ys.min()},{xs.max()},{ys.max()})')
    print(f'  -> {out}/render_mask.npy')


if __name__ == '__main__':
    main()
