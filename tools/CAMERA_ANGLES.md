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

| object (export dir) | offset |
|---|---|
| hand (sx_hand) | 0 |
| chair gallery (sx_chair, wooden crack) | 175 |
| pumpkin (sx_pumpkin) | 190 |
| airplane red cracks (sx_airplane) | 195 |
| spot lava (sx_spot) | 180 |
| duck / bob_spots (sx_duck) | 210 |
| ivysaur (sx_ivysaur) | 210 |
| blub (sx_blub) | 190 |
| unicorn rainbow (sx_unicorn) | 150 |
| goat burnt (sx_goat) | 170 |
| nefertiti effect_2 (sx_lady2) | 150 |
| cont unicorn 1 = sx_cont2 / 2 = sx_cont1 | 150 |
| chair moss (sx_chairmoss) | 195 (nv1 az195 e0, nv2 az240 e25 measured directly) |
| meshnca spot (meshnca_export/spot_lava) | see measured clip cameras below |
| meshnca ivysaur | 210 |
| transfers (gx_ror_* / gx_lava_*) | per-shape blender az below |

## Webpage clip cameras (blender az/el unless noted)

Gallery (main): hand sup az0 e0, nv1 az50 e15, nv2 az230 e15.
Chair: sup az175 e0, nv1 az220 e25, nv2 az310 e-20 **NO_FLOOR CAM_FILL=3.0**.
Pumpkin: sup az190 e0 (121f), nv1 az190 e25, nv2 az134 e25.
Airplane: sup az185 e0, nv1 az240 e25, nv2 az145 e25.
Duck (provenance `slot_cameras.json` + FIG_EXISTING_TEXTURE_V2/README):
  sup = train **pipeline y0 e0 -> az210 e0**;
  nv1 = `view_bob_spots_27m_d01` **pipeline y50 e-8 -> az260 e-8** plus the
  figure's **image roll -43.5 deg** (README: rotation applied so the duck's
  base is level; use the driver's `--roll`);
  nv2 = `view_bob_spots_27m_sideH` **pipeline y210 e0 -> az60 e0**.
Spot comparison: ours/frozen novel az225 e25, nv2 az45 e30; meshnca pair
  tuned by review — novel az225 e25, nv2 az75 e30 (started az240/az60,
  user asked +-15 toward/away from camera).
Duck row / flicker chairmoss: nv1 az195 e0 & az195 e0 (frozen same),
  nv2 az240 e25 (both methods).
Additional gallery (train/diagA/diagC = pipeline y0e0 / y45e25 / y225e30):
  unicorn az150/az195 e25/az15 e30; goat az170/az215 e25/az35 e30;
  blub az190/az235 e25/az55 e30; lady2 az150/az195 e25/az15 e30.
Ivysaur (all three methods): train az210 e0, diagA az255 e25, diagC az75 e30.
Cont: both sup az150 e0, encode with crop=564:564:94:107 (zoom to match refs).
Transfers (el0 except noted): ror train (spot) az180; lava train (skull)
  az210; ufo 270, blub 240, bob 210, octopus 240, dragon 210, chair 230,
  mike 270, napoleon 170, sheep 240, tie_fighter 240, octocat az330 e30,
  teapot az150 e10 **--rotx 90**.
Teaser (live viewer only): training view, pack `plane`.

## Pipeline camera families (Raj's conventions)

train = y0 e0 · diagA = y45 e25 · diagB = y135 e-20 · diagC = y225 e30.
Gallery paper cams: hand y50 e15 & y230 e15, pumpkin y74 e25 (figure) /
y304 e25 (webpage), chair figure rows y120 e25 & y135 e15, plane y45 e25 &
y310 e25.

## Render driver knobs (`itai_render/render_anim_colors.py`)

`--az --el --dist(3.2) --scale(1.0 unit-sphere) --rotx --rotz --spin
--frames A:B` · env: `NO_SIDE_SHADOW=1` (always, the approved look),
`NO_FLOOR=1` (below-horizon cameras; camera z < 0), `CAM_FILL=<x>`
(camera-aligned non-casting fill; chair nv2 uses 3.0), `SUN_ANGLE`,
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
