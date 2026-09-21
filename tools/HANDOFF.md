# DynaMesh render handoff — where every camera and render lives

Written 2026-09-18 as a handoff for a colleague or a fresh Claude session.
Everything below is committed, pushed, or archived; nothing lives only in a
chat session. Read this file, then `SHADOW_PIPELINE.md` (the how-to) and
`figure_cameras.csv` (the what), before touching any figure or render.

## Working protocol (user-mandated, learned the hard way)

1. Cluster access: `ssh uchicago` (alias for guanc@fe.ai.cs.uchicago.edu).
   Write ONLY under `/net/projects/ranalab/guanc/`; rajhansini's tree is
   read-only. Exclude node r003 (bad GPU; already in the sbatch headers).
2. Never re-render reference videos or the video baselines (SV4D 2.0, L4GM,
   DG4D) — and their panels carry NO shadow (shadow = evidence of geometry).
   Mesh methods (ours, Frozen TRELLIS.2, MeshNCA) get real Cycles shadows.
3. Before changing any figure view: silhouette-match against the current
   panel, render at the matched camera, and put a labeled before/after PNG in
   `~/Downloads/dynamesh_figures_before_after/` for Guan to approve BEFORE
   overwriting the XML. When any visual choice is uncertain, render labeled,
   NUMBERED candidates into that folder and let Guan pick.
4. Record every camera/frame change in `figure_cameras.csv` in the same
   commit as the change.
5. Webpage changes must reach mobile: it is the same responsive build, so
   bump the `?v=` cache-buster on every changed asset URL, and grep the repo
   for other pages referencing the file.
6. Judge renders only from `render_clip.sbatch` output (render_sweep lacks
   NO_SIDE_SHADOW) and only visually — full sheets, never metrics or single
   frames.

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

## Open items

None as of 2026-09-21. Teaser row 2 was picked and applied (az240 e20
`--rotx 180 --obj-roll -120`, dir `teaser_row2_v2`); the supp-gallery
blub section is final (three rows, all ending at f128 with the eye
visible). Both colleague feedback batches (frame re-picks, brightness
dims, ivysaur NV2, vase-B shadows, duck layout) are implemented in the
XMLs and on the webpage. Current state per row: `figure_cameras.csv`.
