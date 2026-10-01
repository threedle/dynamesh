"""
make_targets.py — 150 2D targets that match our render's silhouette EXACTLY

WHY THE OLD PATH WAS WRONG
  The earlier path went image -> mesh vertices -> image.
  That round trip is lossy by construction:

      mesh            430,910 faces
      covered pixels   98,083   -> only 23% of faces ever win a pixel
      vertices supervised every frame  39,708/215,462  (18.4%)
      vertices NEVER supervised       173,951

  The mesh is 4.4x denser than the raster can resolve, so most faces never
  receive colour, and re-rendering from those vertices leaves holes. Normalising
  the barycentric weights recovered half the fringe (739 -> 343 px) but cannot
  fix the cause: the vertices simply do not carry the signal.

WHY THE MIDDLEMAN IS UNNECESSARY HERE
  The GT video and our render share ONE fixed camera. Pixel p in the video and
  pixel p in our render are already the same screen location, so the target for
  our pixel p is just the video's pixel p. Going out to the surface and back
  buys nothing at a fixed viewpoint — it only loses.

  The surface round trip is the right tool when you must re-render from a NEW
  view. For a fixed-camera 2D loss it is pure loss.

WHAT LIMITS COVERAGE
  Only registration. Our mesh's silhouette is not exactly the video object's:
  IoU can fall over the sequence because the object in the video can grow
  (102,549 -> 109,583 px) while our mesh is fixed. That leaves a thin rim of our
  silhouette lying outside the video object — 0.6% to 2.2% of our pixels.

  Measured: a centroid+isotropic-scale alignment makes this WORSE (94.6% vs
  97.8% at f1), so the video is already correctly positioned and must not be
  warped. The rim is instead filled from its nearest valid neighbour, which is
  a few pixels away, giving 100% coverage with no white anywhere.

OUTPUT (per frame, at --render-res, our silhouette by construction)
  gt_{f}.png      target image, EVERY pixel of our silhouette filled
  valid_{f}.npy   our full silhouette (100%) — the filled target
  strict_{f}.npy  only pixels whose colour came straight from the video, no fill
"""

import argparse, json, os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--gt-dir', required=True,
                help='reference video frames, frame_0001.png ...')
ap.add_argument('--render-mask', required=True,
                help='our render silhouette at --render-res, from the SAME '
                     'mesh+camera the training loop rasterises')
ap.add_argument('--render-res', type=int, default=518)
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--bg-thresh', type=int, default=245)
ap.add_argument('--tag', default='gt_targets')
args = ap.parse_args()

import numpy as np
from PIL import Image
from scipy import ndimage

OUT = (_HERE / 'out' / args.tag).resolve()
(OUT / 'frames').mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def largest_component(mask):
    lb, n = ndimage.label(mask)
    if n == 0:
        return mask
    return lb == (1 + np.argmax(np.bincount(lb.ravel())[1:]))


def main():
    log('=' * 88)
    log('GT TARGETS — direct, fixed camera, no surface round trip')
    log('=' * 88)

    ours = np.load(args.render_mask)
    R = args.render_res
    assert ours.shape == (R, R), f'render mask {ours.shape} != {R}^2'
    A = int(ours.sum())
    log(f'[RENDER] our silhouette {A:,} px at {R}^2 — fixed, mesh and camera are fixed')

    strict_px = np.zeros(args.n_frames, np.int64)
    filled_px = np.zeros(args.n_frames, np.int64)
    ious = np.zeros(args.n_frames)
    src_max = 0.0            # brightest min-channel value anywhere in the source object

    for i in range(args.n_frames):
        f = i + 1
        raw = Image.open(Path(args.gt_dir) / f'frame_{f:04d}.png').convert('RGB')
        arr = np.array(raw).astype(np.float32) / 255.0
        # object mask at FULL resolution, then nearest-resize: thresholding after
        # a smooth resize would bleed background into the silhouette
        vid_full = largest_component(arr.min(axis=2) < args.bg_thresh / 255.0)

        # BLEED the object colour outward over the background BEFORE resizing.
        # Measured: the video has ZERO pixels above 0.98 inside its own object
        # (max 0.957), yet LANCZOS on the raw frame produced values above 1.0 by
        # ringing against the white background, and the rim fill then copied that
        # overshoot inward. With the background replaced by its nearest object
        # colour there is no white left to ring against or to copy.
        _, (by, bx) = ndimage.distance_transform_edt(~vid_full, return_indices=True)
        bled = arr[by, bx]

        vid = np.array(Image.fromarray((vid_full * 255).astype(np.uint8))
                       .resize((R, R), Image.NEAREST)) > 127
        # BOX = area average, the correct filter for 960->518 downsampling and
        # the one that cannot overshoot its inputs
        img = np.array(Image.fromarray((bled * 255).astype(np.uint8))
                       .resize((R, R), Image.BOX)).astype(np.float32) / 255.0

        strict = ours & vid                      # colour straight from the video
        ious[i] = (ours & vid).sum() / (ours | vid).sum()
        src_max = max(src_max, float(arr[vid_full].min(axis=1).max()))

        out = np.ones((R, R, 3), np.float32)     # white OUTSIDE our silhouette only
        out[ours] = img[ours].clip(0, 1)
        Image.fromarray((out * 255).astype(np.uint8)).save(
            OUT / 'frames' / f'gt_{f:04d}.png')
        np.save(OUT / 'frames' / f'valid_{f:04d}.npy', ours)
        np.save(OUT / 'frames' / f'strict_{f:04d}.npy', strict)

        strict_px[i], filled_px[i] = int(strict.sum()), A
        if f % 25 == 0 or f == 1:
            log(f'  {f:3d}/{args.n_frames}  from video {strict_px[i]:,}/{A:,} '
                f'({100*strict_px[i]/A:5.2f}%)  rim filled {A-strict_px[i]:,}  '
                f'IoU {ious[i]:.3f}')

    # GATE on WHITE, meaning all three channels at background level. Note the
    # per-pixel min-channel is NOT conserved by area averaging — since
    # min(avg R, avg G, avg B) >= avg(min R, G, B), a legitimate blend of two
    # object pixels can sit above the source's max min-channel (0.957). So the
    # gate is the white threshold itself, which no blend of object colour can
    # reach once the background has been bled away.
    WHITE = 0.98
    white, brightest = 0, 0.0
    for f in range(1, args.n_frames + 1):
        g = np.array(Image.open(OUT / 'frames' / f'gt_{f:04d}.png')
                     .convert('RGB')).astype(np.float32) / 255.0
        v = g.min(axis=2)[ours]
        brightest = max(brightest, float(v.max()))
        white += int((v > WHITE).sum())
    log(f'\n[COVERAGE] every frame: {A:,}/{A:,} = 100.00% of our silhouette has a colour')
    log(f'[SOURCE]   straight from the video: min {100*strict_px.min()/A:.2f}%  '
        f'max {100*strict_px.max()/A:.2f}%  mean {100*strict_px.mean()/A:.2f}%')
    log(f'[FILLED]   nearest-neighbour rim:   min {A-strict_px.max():,}  '
        f'max {A-strict_px.min():,} px')
    log(f'[IoU]      ours vs video silhouette: {ious.min():.3f} .. {ious.max():.3f} '
        f'(can fall over time — the object in the video may grow, our mesh cannot)')
    log(f'[GATE-white] source max min-channel {src_max:.3f}; our targets peak at '
        f'{brightest:.4f}; px above white({WHITE}) inside our silhouette: {white}')
    assert white == 0, f'GATE-white FAILED: {white} white px'
    log('[GATE-white] PASS — no white anywhere inside the silhouette')

    # Ship the silhouette the targets were built on ALONGSIDE them, so the
    # trainer's GATE-align can compare against this exact array rather than
    # guessing at a path elsewhere in the tree.
    np.save(OUT / 'render_mask.npy', ours)
    json.dump(dict(render_res=R, silhouette_px=A, n_frames=args.n_frames,
                   strict_px_min=int(strict_px.min()),
                   strict_px_max=int(strict_px.max()),
                   iou_min=float(ious.min()), iou_max=float(ious.max()),
                   white_px_inside=int(white), brightest=float(brightest)),
              open(OUT / 'gt_targets.json', 'w'), indent=2)
    log(f'[SAVE] {OUT}\n[DONE]')


if __name__ == '__main__':
    main()
