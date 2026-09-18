# DynaMesh real-shadow render pipeline (runbook)

The single reference for re-rendering ANY DynaMesh clip or figure panel
with real Blender shadows. Written so a fresh Claude session (or human)
can operate it end to end. Cameras live in [CAMERA_ANGLES.md](CAMERA_ANGLES.md);
figure-XML surgery in [figtool.py](figtool.py); this file is the how-to.

## What it is

Every mesh render on the project page, the supp page, and the paper's
draw.io figures goes through one Blender scene (Itai's `blank.blend`:
floor + emissive side panel + overhead sun, Cycles GPU, 50 samples,
OPTIX denoise) so all methods share identical lighting and a real cast
shadow. Reference videos and non-mesh baselines (SV4D 2.0, L4GM, DG4D)
are NEVER re-rendered — a shadow is evidence of geometry.

The approved look: `NO_SIDE_SHADOW=1` (side panel lights but does not
cast; baked into `render_clip.sbatch`) and, since 2026-09-16,
`CAM_FILL=3.0` on figure panels and the duck novel views (a
camera-aligned non-casting sun that brightens exactly what the camera
sees; user picked 3.0 over 1.0/2.0).

## Where things live

- Cluster root: `/net/projects/ranalab/guanc/blender/` (`$B` below)
  - `blender-3.2.1-linux-x64/`, `itai_render/` (scene + scripts)
  - `itai_render/render_anim_colors.py` — the driver (mirrored here in
    `tools/blender/`; edit locally, scp up, keep both in sync)
  - `render_clip.sbatch`, `render_sweep.sbatch` (mirrored in
    `tools/blender/`)
  - outputs: `$B/clips/<tag>/0001..0150.png` + `$B/clips/<tag>.mp4`
- Frame sources (150-frame vertex-colored PLY exports):
  `/net/projects/ranalab/guanc/geometry-def/dynamesh_render/out/<export>/`
  - `adapted_f%04d.ply` = ours, `frozen_f%04d.ply` = frozen TRELLIS.2,
    `meshnca_f%04d.ply` (meshnca_export*/) = MeshNCA
  - per-object export dirs and their cameras: see CAMERA_ANGLES.md
- Missing an export? `dynamesh_render/export_verts.sbatch`
  (`RUN=<run dir>,FRAMES=1:2:...:150,TAG=<name>,MESH=<obj>` — transfer
  runs need MESH) and `export_mnca.sbatch`
  (`OBJ=,FRAMES=16:53:98:150,EXP=<results name>,OUT=`).

## Render a clip

```bash
ssh guanc@fe.ai.cs.uchicago.edu
B=/net/projects/ranalab/guanc/blender
O=/net/projects/ranalab/guanc/geometry-def/dynamesh_render/out
sbatch --export=ALL,CAM_FILL=3.0,PLYDIR=$O/sx_duck,OUTDIR=$B/clips/mytag,\
FRAMES=1:150,AZ=230,EL=-4,SCALE=1.0,RES="768 768",EXTRA="--roll 43.5" \
  $B/render_clip.sbatch
```

- `FRAMES` — `1:150` range, or a DOTTED list `10.57.104.150` (the sbatch
  translates dots to commas; NEVER put commas in `--export` values, sbatch
  splits on them — this silently drops frames).
- Env knobs (add to `--export`): `NO_FLOOR=1` (floorless/no shadow — use
  when the camera is below the floor plane or rolled), `CAM_FILL=<x>`
  (3.0 standard), `SUN_ANGLE`. `NO_SIDE_SHADOW=1` is already in the sbatch.
  `SUN_TILT="tilt rotz"` (degrees) leans the sun off vertical — required for
  top-down cameras (el ≳ 60) where the object hides its own straight-down
  shadow; the shadow extends toward world `(-sin rotz, cos rotz)`. For a
  camera at az 0, `rotz 0` throws it toward the image bottom, `rotz -45`
  bottom-left, `rotz 45` bottom-right.
- `EXTRA` flags: `--pattern frozen_f{:04d}.ply` / `meshnca_f{:04d}.ply`,
  `--roll <deg>` (positive = image clockwise), `--rotx/--rotz` (object),
  `--spin 360` (turntable), `--dist` (default 3.2).
- The driver normalizes to the unit sphere, drops to the floor, and uses
  the pipeline camera convention: yaw 0 looks from +Y; per-object
  blender_az = pipeline_yaw + offset (offsets in CAMERA_ANGLES.md).
- Sweeps (view hunting, 320px/1 frame): `render_sweep.sbatch` with
  `EL=,FRAME=,AZLIST="0 30 60",EXTRA=`.

## Encode + ship a webpage clip

```bash
ffmpeg -y -framerate 25 -i $B/clips/mytag/%04d.png -vf scale=384:384 \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart $B/clips/mytag.mp4
```
(121-frame pumpkin: `-framerate 20.1667`.) scp down, copy over the page
file (name map: `tools/` has `slot_cameras.json`; supp names are
`supp/videos/NNN_*.mp4`), commit + push. Validate against the OLD clip
frame-by-frame first (contact sheet), never by a single frame.

## Figure XML surgery

`tools/figtool.py` (needs pillow+numpy):
- `extract <fig.xml> <dir>` → panels + `index.json` (geometry, payload ids)
- `align <old_panel> <new_render> <out.jpg> [bbox|width|height]` —
  silhouette-bbox aligns the render onto the old panel canvas
- mark entries in index.json with `"new": <path>`, then
  `swap <fig.xml> <index.json>` rewrites the base64 payloads in place
  (handles duplicated payloads occurrence-by-occurrence).

Rules learned the hard way: match views by SILHOUETTE IoU in the same
render domain (color features fail across lighting); frame indices via
effect-intensity curves (lava/amber = warm-pixel fraction) or same-domain
pixel match; white objects (vase) need threshold 240+, not 185; check for
in-figure 2D rotations (duck A = 43.5° clockwise); film-strip reference
frames and baseline panels are untouched.

## Gotchas

- sbatch `--export` + commas: see FRAMES note above. Same for EXTRA.
- `scancel -n <name>` may take out more than you think — check `squeue`
  after, and re-check export jobs actually finished (`[DONE]` in logs).
- Exclude node r003 (bad GPU) — already in the sbatch headers.
- Below-horizon = camera z < 0 in world, not el < 0.
- WORLD_FILL is a no-op (blank.blend world has no Background node); use
  CAM_FILL.
- The live WebGL viewers (comparison.js) have their own lighting — packs
  need `computeVertexNormals()`, r160 physical lights need ~π-scaled
  intensities.

## draw.io PDF export ignores clipPath (2026-09-17)

The PDF exporter draws every image cell as its FULL opaque rectangle in
document order — `clipPath=inset(...)` is honored by the web viewer but NOT
by the PDF export. Any panel whose (unclipped) white padding overlaps the
panel above therefore covers that panel's shadow with a hard edge in the PDF
while the viewer looks fine ("it's not a crop thing").

Fix applied to gallery / supp_gallery / supp_comparison / extended_texture /
flicker / generalization: bake the crop into the payload — crop the bitmap to
its clip window + content bbox (thr 252, 2 px margin), remap mxGeometry (the
cell rect equals the clip WINDOW, and inset percentages are relative to the
full image extent, not the cell), and delete the clipPath. Where content
genuinely overlaps a neighbor's shadow (extended_texture lower rows, flicker
round insets) the occluder is converted to PNG with a 240→252 luminance alpha
ramp so the shadow behind composites through. Film-strip artwork is left
clipped (its overlaps are the intended frame design). Audit + bake code lives
in the session scratchpad and reruns from figtool-style extraction; verify
with a full-rect document-order composite, not the web viewer.

Also: `figtool.align` now smoothsteps both the distance fade and the border
fade and blur-extends the penumbra (`soften_shadow`) so Cycles' compact
umbra edge never reads as a straight cutoff line at figure scale.

## Video baselines carry NO shadow (2026-09-17)

A shadow is evidence of geometry: L4GM, SV4D 2.0 and DG4D output video, so
their panels must sit on plain white. The original figures had gray floor
shadows baked into those panels; they were scrubbed (component analysis:
low-saturation gray regions outside/below the object silhouette whitened,
object pixels untouched) in comparison.xml and supp_comparison.xml — all 36
video-baseline panels. Mesh methods (ours, Frozen TRELLIS.2, MeshNCA) keep
their real Cycles shadows. Reference film strips untouched.

## Render archive split (2026-09-18)

`$B/clips/` now holds ONLY the canonical figure sources (see
`clips_archive_NOT_IN_FINAL_FIGURES/README.md` for the keep list) plus the
webpage clip dirs. All 266 superseded generations, view sweeps and failed
candidates were verified against the panels embedded in the current figure
XMLs and moved to `$B/clips_archive_NOT_IN_FINAL_FIGURES/`, each family
explained in that folder's README.
