# DynaMesh webpage & figure camera reference

Single source of truth for every render camera. Two coordinate systems:

- **Pipeline (yaw, elev)** — the TRELLIS render convention used by
  `render_view_guanc.sbatch` (`YAW0=`, `ELEV=`) and all of Raj's
  `view_*`/`panel_*` tags. The supervised/training camera is `y0 e0`.
- **Blender az** — the azimuth handed to `render_anim_colors.py`
  (`--az`, `--el`). Relation per object: `blender_az = (pipeline_yaw +
  offset) % 360`, same elevation.

The authoritative slot map recovered from Raj's tree
(`after_submission/shadow/slot_cameras.json`, copied next to this file)
lists, for every supp clip slot, the exact source render tag; the tags'
yaw/elev live in `out/job_manifest.json` on the cluster.

## Per-object Blender offsets (pipeline y0 -> blender az)

**The offset is 180 for almost every trimesh export** (verified 2026-09-16
by silhouette-matching against the original clips: unicorn, blub, goat
~184, lady2, ivysaur ~185, meshnca-ivysaur, duck (d01 y50 -> az230), spot).
Earlier per-object offsets (150/170/190/210...) were mirror-twin sweep
errors — do NOT reuse them. Deviations that remain real:

| object (export dir) | offset |
|---|---|
| hand (sx_hand) | 0 |
| chair gallery (sx_chair, wooden crack) | ~175 (user-approved views) |
| pumpkin (sx_pumpkin) | ~190 (user-approved views) |
| airplane red cracks (sx_airplane) | ~195 (user-approved views) |
| chair moss (sx_chairmoss) | nv1 az195 e0, nv2 az240 e25 (measured directly) |
| cont unicorn (sx_cont1/2) | figure views measured directly (az205/az25) |
| meshnca spot | final views measured directly (novel az215 e25, nv2 az45 e30) |
| transfers (gx_ror_* / gx_lava_*) | per-shape blender az below |

Raj's train renders also carry 0-10 deg extra downward tilt (el 5-10)
for some objects; always confirm el against the original clip.

## Webpage clip cameras (blender az/el unless noted)

Gallery (main): hand sup az0 e0, nv1 az50 e15, nv2 az230 e15.
Chair: sup az175 e0, nv1 az220 e25, nv2 az310 e-20 **NO_FLOOR CAM_FILL=3.0**.
Pumpkin: sup az190 e0 (121f), nv1 az190 e25, nv2 az134 e25.
Airplane: sup az185 e0, nv1 az240 e25, nv2 az145 e25.
Duck (empirical, silhouette-locked against the original clips — the
  manifest params for this object do NOT map through the az offset; Raj's
  duck cameras also carry extra elevation):
  sup = **az190 el5** (refined 2026-09-16, IoU .87; the earlier az210 el10
  shipped briefly and was superseded);
  nv1 = **az230 el-4 `--roll 43.5` NO_FLOOR=1** (az230 el-4 with no roll
  reproduces the original 057/d01 render at IoU 0.971; the figure's duck A
  = that view rotated 43.5 deg clockwise in-image, IoU 0.958 — the user
  wants the figure look: head right, facing the bottom-right corner);
  nv2 (webpage) = **az30 el55 CAM_FILL=3.0** (final per user 2026-09-17;
  figure duck B stays az60 e0); both webpage novels use CAM_FILL=3.0.
Spot comparison: ALL THREE geometry methods (ours, frozen, meshnca) share
  novel az225 e25 and nv2 az45 e30 (meshnca measured empirically: az240/az60
  minus the user's final -15 tweak lands exactly on the ours/frozen yaws).
Duck row / flicker chairmoss: nv1 az195 e0 & az195 e0 (frozen same),
  nv2 az240 e25 (both methods).
Additional gallery + ivysaur — CORRECTED 2026-09-16 by silhouette-matching
  against the original clips in git history (the old per-object offsets
  were mirror-twin errors; the true offset is a UNIVERSAL 180 for all of
  these objects, with Raj's train views carrying 0-5 deg extra tilt):
  unicorn train az180 e0 / diagA az225 e25 / diagC az40 e30;
  blub train az180 e5 / az225 e25 / az35 e30;
  goat train az184 e5 / az225 e25 / az40 e30;
  lady2 train az180 e0 / az225 e25 / az45 e30;
  ivysaur (ours+frozen) train az185 e0 / az225 e25 / az45 e30;
  meshnca-ivysaur train az180 e5 / az225 e25 / az45 e30;
  duck sup az190 e5.
  (IoU vs originals: bulky objects hit .96-.97; thin-featured ones cap at
  .73-.87 with the view still visually exact — see 00_view_match_check.png.)
Cont: both sup az150 e0, encode with crop=564:564:94:107 (zoom to match refs).
Transfers (el0 except noted): ror train (spot) az180; lava train (skull)
  az210; ufo 270, blub 240, bob 210, octopus 240, dragon 210, chair 230,
  mike 270, napoleon 170, sheep 240, tie_fighter 240, octocat az330 e30,
  teapot az150 e10 **--rotx 90**.
Teaser (live viewer only): training view, pack `plane`.

## Figure cameras (empirically locked per figure, blender az/el)

Gallery: webpage cameras, canonical frames 10/57/104/150.
Supp gallery (frame-matched in the original domain, err <=0.017):
  unicorn diagA+diagC rows frames 45/80/115/150; blub diagA+diagC
  45/80/115/150; goat diagA+diagC 30/60/90/120; lady train 24/44/63/83,
  lady row2 = diagB-like back view (az~315 e-20 NO_FLOOR, being refined).
Supp comparison (ivysaur, Page-6): row1 = diagC (az45 e30), row2 = diagA
  (az225 e25) — corrected with the universal-180 offset; frames
  30/70/110/150; frozen row2 is the strip at x>0.
Flicker: both blocks = chairmoss az250 e25 (rotated +10 per review); left = frozen (054)
  frames 21/61/101/150, right = ours (052) frames 21/62/102/150; zoom
  insets share one crop box per row (template-matched, kept old ring).
Main comparison (spot): ours/frozen ~az210-230 e25-35 (refining),
  meshnca ~az240 e15; frames ours 19/55/105/150, frozen 76/95/125/150 (pushed later per review).
Spot second view: ours/frozen az40 e25, meshnca ~az30-60 e25;
  frames ours 1/50/105/150, frozen 1/85/125/150.
System figure: spot at ~az140 e25 (its own mirror-side view); panels gray f1 / f65 / f85 / f99.
Extended-texture supp (cont unicorn): view1 az205 e15, view2 az25 e20;
  clip1(blue,sx_cont2) frames 30/70/110/150, clip2(amber,sx_cont1)
  frames 30/65/91/132.
Existing texture (duck): duck A = nv1 camera (az230 e-4 roll43.5,
  NO_FLOOR), duck B = az60 e0; frames 102/118/134/150; strips are
  pre-composited 4-slot images (bbox slots at fixed segment gaps).
Teaser (plane vx_planeF): row1 ~az220 e20, row2 ~az130 e25, film strip
  az180 e0 (frame-matching pending; plane deforms so IoU needs
  frame-aligned comparison). Black-background panels stay untouched.
Failure (vase sx_vase, white object: threshold 240+, NO_FLOOR sweeps):
  viewA az180 e15, viewB az330 e25 (raised per review), frames 30/70/110/150.
Generalization figure (all silhouette-locked, frames 10/55/100/145,
  fill 3.0): ror block = spot(gx_ror_train) az85 e10, alien(gx_ror_alien,
  exported from run 72244c6e + data/alien_glow mesh) az150 e15,
  monster(gx_ror_monster, data/monster_lava_2 mesh) az180 e15,
  teapot(gx_ror_teapot) az180 e12 with `--rotx 90` (upright fix, match
  .86); lava block = skull(gx_lava_train) az325 e30, sheep az210
  e15, chair az90 e15, dragon az0 e15. Row1 of the figure is SPOT (the
  training shape), not a horse.
MeshNCA-ablation supp figure: 4 rows = MeshNCA results
  abl_spot_lava_f016 / f053 / f098 / ours_spot_lava (target-frame
  ablation; exports meshnca_export_abl_* + meshnca_export/spot_lava),
  all at az225 e25 (match .90), columns = frames 16/53/98/150 (the
  export default). Left target panels stay untouched.
Supp gallery corrections: unicorn+blub rows are diagA+diagC (frames
  45/80/115/150); lady row2 = back view az330 e-15 NO_FLOOR (match caps
  at .82), frames 24/44/63/83.

## Pipeline camera families (Raj's conventions)

train = y0 e0 · diagA = y45 e25 · diagB = y135 e-20 · diagC = y225 e30.
Gallery paper cams: hand y50 e15 & y230 e15, pumpkin y74 e25 (figure) /
y304 e25 (webpage), chair figure rows y120 e25 & y135 e15, plane y45 e25 &
y310 e25.

## Render driver knobs (`itai_render/render_anim_colors.py`)

`--az --el --dist(3.2) --scale(1.0 unit-sphere) --rotx --rotz --spin
--frames A:B` · env: `NO_SIDE_SHADOW=1` (always, the approved look),
`NO_FLOOR=1` (below-horizon cameras; camera z < 0), `CAM_FILL=<x>`
(camera-aligned non-casting fill; FIGURE PANELS use 4.0, duck webpage novels 3.0, chair nv2 3.0), `SUN_ANGLE`,
`BAKE_SHADOW=1` (floor-shadow bake mode). Blender scene = Itai's
blank.blend; sbatch wrappers in `/net/projects/ranalab/guanc/blender/`.

## Gotchas

- Silhouette matching mirror-confuses near-symmetric objects; verify with
  the color pattern in the SAME render domain, and any two views that are
  180 apart in pipeline yaw must stay 180 apart in blender az.
- The figure XMLs sometimes rotate panels in 2D (duck d01: -43.5 deg) —
  check each figure dir's README under Raj's `out/FIG_*`.
- Below-horizon means camera z < 0 in world (approx look_z + dist*sin(el)),
  not merely el < 0: duck d01 at e-8 keeps the floor; pumpkin e-25 did not.
