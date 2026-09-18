# DynaMesh render handoff — where every camera and render lives

Written 2026-09-18 as a colleague handoff. Everything below is committed,
pushed, or archived; nothing lives only in a chat session.

## Camera angles (the authoritative records)

All in this repo, `dynamesh_page/tools/` (github: threedle/dynamesh page repo):

| file | what it holds |
|---|---|
| `figure_cameras.csv` | THE per-row sheet: figure, row, PLY export dir, az, el, extra flags (`--obj-roll`, `--rotx`, …), env knobs (`CAM_FILL`, `NO_FLOOR`, `SUN_TILT`), exact frame picks, status. Includes the supp-page NV2 clips and NEVER-RE-RENDER baseline rows. |
| `CAMERA_ANGLES.md` | Per-object azimuth conventions (universal az offset 180 for trimesh exports, silhouette-IoU matching method, per-object notes). |
| `SHADOW_PIPELINE.md` | The full runbook: how to render (sbatch invocations, env knobs), figtool XML surgery, and the hard-won gotchas — draw.io PDF export ignores clipPath (crops must be baked into payloads), video baselines carry no shadow, the render-archive split. |
| `TEASER_GREYII.md` | Teaser (Grey-II plane): run id, `blender_az = pipeline_yaw + 160`, approved frames, row cameras, floor/mirror constraints. |
| `blender/render_anim_colors.py`, `blender/render_clip.sbatch`, `blender/render_sweep.sbatch` | Local mirrors of the cluster driver + sbatch wrappers (edit here, scp up). |
| `figtool.py` | Figure-XML extract / align / swap tooling (payload-level, md5-keyed). |

## Renders (cluster, `/net/projects/ranalab/guanc/`)

| path | contents |
|---|---|
| `blender/clips/` | CANONICAL frame PNGs only: the 59 figure-source dirs (listed per figure in the archive README below) plus the webpage clip dirs (`shad_*`, `shad2_*`, `ag2_lady_diagC_floor`, …). |
| `blender/clips_archive_NOT_IN_FINAL_FIGURES/` | All 266 superseded generations, view sweeps and failed candidates, verified frame-by-frame against the panels embedded in the current figure XMLs. `README.md` inside explains every family and names its replacement, and lists the canonical keep set per figure. |
| `blender/teaser_row1/`, `blender/teaser_row2_final/` | Grey-II teaser row renders (row 2 currently being re-picked; candidates in `blender/tsr2_cands/`, decision grid with Guan). |
| `geometry-def/dynamesh_render/out/` | The vertex-colored PLY/GLB exports every render reads from: `sx_<object>/` (figure objects, adapted + frozen), `gx_<effect>_<target>/` (generalization transfers), `meshnca_export*/`, `px_greyII/` (teaser frames). |

## Shipped deliverables

| path | contents |
|---|---|
| `~/Downloads/draw.io_xml_files/` (Guan's laptop) | The 14 final figure XMLs, panels embedded (gallery, supp_gallery, supp_comparison, comparison, spot_second_view, system, failure, flicker, extended_texture, existing_texture, generalization, meshnca_ablation, technique, teaser). |
| `dynamesh_page` repo (`index.html`, `supp/`, `assets/`) | The live project + supp pages (threedle.github.io/dynamesh), all videos cache-busted. |
| `~/Downloads/dynamesh_figures_before_after/` (Guan's laptop) | Review evidence: before/after sheets, PDF-semantics confirm renders, candidate grids. |

## Open item

Teaser row 2 view: nose-up re-pick in progress; numbered candidate grid at
`~/Downloads/dynamesh_figures_before_after/00_teaser_row2_fine_grid.png`,
renders in `blender/tsr2_cands/`. Once picked: render frames
70/90/105/130/150/230/300/450 at that camera, rebuild the teaser row-2
panels via `figtool.py`, update `figure_cameras.csv`.
