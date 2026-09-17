# Teaser rebuild — airplane "Waves, from Grey II" (handoff)

State as of 2026-09-17. The teaser figure has been rebuilt end to end and the
before/after is approved-pending-review. Everything below is verified, not
assumed; where a number was measured the method is given so you can re-check it
rather than trust it. Companion docs: [SHADOW_PIPELINE.md](SHADOW_PIPELINE.md)
(how to render), [CAMERA_ANGLES.md](CAMERA_ANGLES.md) (all other figures).

## What the figure is

8 columns x 3 bands = 24 image panels:
reference film strip (ground-truth video frames), row 1 and row 2 (our result
from two cameras). The OLD teaser stitched two clips together, `plane_waves`
in columns 1-4 and `plane_waves_from_frame_150` in columns 5-8, and the right
half was never shadow-rendered so it sat on black. The NEW one uses ONE
450-frame clip across all 8 columns, every panel through the shadow pipeline.

## Source

| thing | path |
|---|---|
| screen recording the user supplied | `~/Desktop/new_teaser_airplane.mov` (stays on the Training-view tab the whole time) |
| dataset, 450 frames, starts grey | `/net/projects/ranalab/rajhansini/TRELLIS.2/data/plane_extended_final/` |
| ground-truth frames | `<dataset>/frames_from_video/frame_%04d.png` |
| mesh | `<dataset>/mesh/plane_extended_final_render_frame.obj` |
| run (ours) | `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/after_submission/rung27_mcfm_temporal_only/runs/rung27_l1_lp_mcfmv2_G_all_qkvo+sa_r4_s42_dc8e1c97` |

The run is under `after_submission/`, NOT the main `runs/` root — grepping the
main root for it finds nothing. `mcfm = v2_G` means an 11-frame temporal window
(C..I are widths 2/3/5/7/11/13/15). Dataset identified by pixel-matching the
recording's ground-truth panel against all three 450-frame plane candidates
(RGB err 43.7 for this one vs 68.9 and 75.6).

Kling was fed a plain grey render at a tilted camera (yaw 30 / elev 25) and that
rotation is BAKED INTO THE MESH. Texture arrives over roughly frames 45-130;
after that it only churns.

## Frames (user-approved, "option A")

```
70  90  105  130  150  230  300  450
```

Chosen so the figure does NOT open on a grey mesh (explicit user instruction)
while still showing the effect arrive. f70 has the wave on one wing with grey
still on the other; grey is gone by f150; 230/300/450 show it still moving.

## Cameras

**`blender_az = pipeline_yaw + 160`** for this object. Measured by silhouette-IoU
matching ground-truth frame 130 against a 36-step el0 sweep (az160 wins, 0.521 vs
0.469 for the runner-up). Cross-checked: Diagonal A maps to az205 el25 and does
render as the nose-down "diving" look the user independently described.

| tab | pipeline | blender |
|---|---|---|
| Training | y0 e0 | az160 el0 |
| Diagonal A | y45 e25 | az205 el25 |
| Diagonal B | y135 e-20 | az295 el-20 |
| Diagonal C | y225 e30 | az25 el30 |

**FINAL choices:**
- **row 1 — `AZ=160 EL=10 SCALE=1.55`**, floor on.
- **row 2 — `AZ=295 EL=-10 SCALE=1.40`**, floor on. This is Diagonal B's azimuth
  lifted 10 degrees. See the floor note below for why it is not el-20.

### Floor / shadow constraint (do not re-derive this)

The look point ends up at **z = 0.777** (object origin after normalize-to-unit-
sphere at scale 1.55, then drop-to-floor). Camera height is `0.777 + 3.2*sin(el)`,
so the camera drops BELOW the floor at about **el < -13**. Below the floor the
floor occludes the object completely — az295 el-20 with the floor on renders a
**pure white 7 KB image**, verified. `NO_FLOOR=1` is what makes it visible, and it
removes the surface the shadow falls on. Hence el-10 for row 2.

### Do not chase an exact mirror of a view

The airframe's bilateral symmetry plane has normal **[0.797, 0.367, 0.480]**,
**61 degrees off vertical** (fitted to 45,811 vertices, reflect-and-match,
RMS 0.013 on a unit-normalised cloud). Because it is tilted, no fixed-elevation
azimuth change can mirror a view. Reflecting the camera across the real plane
needs a compensating roll, the roll tilts the floor, and `NO_FLOOR` then kills
the shadow. `--rotx=-52.6` stands the symmetry plane vertical and makes mirroring
a plain `az -> 74.3 - az` with no roll — but it also un-tilts the aircraft, so the
pose stops looking like the one being mirrored. **You can have any two of: the
banked attitude, an exact mirror, a level floor with its shadow.** This was
explored at length and abandoned; pick views from a sweep instead.

## Render paths on the cluster

```
B=/net/projects/ranalab/guanc/blender
O=/net/projects/ranalab/guanc/geometry-def/dynamesh_render/out
```

| what | path |
|---|---|
| PLY export (frames 1/45/70/90/105/130/150/230/300/380/450, adapted + frozen) | `$O/px_greyII/` |
| **row 1 final** | `$B/teaser_row1/` |
| **row 2 final (el-10 scale 1.40)** | `$B/teaser_row2_final/` |
| row 2 alternates | `$B/teaser_row2/` (el-20 NO_FLOOR), `$B/teaser_row2_e10/` (el-10 scale 1.15, too small), `$B/teaser_row2_e0/` (el0) |
| proof that el-20 + floor is blank | `$B/proof_e20floor/0130.png` |
| view sweeps | `$B/sweep_gII_{e0,e20,e35,em20,r1a,r1b}/`, `$B/roll_{e25,e10,em10,em25}/` |
| mirror experiments (abandoned) | `$B/upright_e{0,10,20}/`, `$B/mirror_true*/`, `$B/mrl_*/` |

Re-render either row:

```bash
sbatch -J tsr1 --export=ALL,CAM_FILL=3.0,PLYDIR=$O/px_greyII,OUTDIR=$B/teaser_row1,\
FRAMES=70.90.105.130.150.230.300.450,AZ=160,EL=10,SCALE=1.55,RES="768 768" $B/render_clip.sbatch
```

Re-export PLYs if needed (**must** pass `--exclude=r003`, see gotchas):

```bash
sbatch -J g_pxexp --exclude=r003 --export=ALL,RUN=<run above>,\
FRAMES=1:45:70:90:105:130:150:230:300:380:450,TAG=px_greyII \
  /net/projects/ranalab/guanc/geometry-def/dynamesh_render/export_verts.sbatch
```

## Local deliverables

| what | path |
|---|---|
| rebuilt figure XML | `~/Downloads/dynamesh_teaser_greyII/dynamesh_teaser_greyII.drawio.xml` |
| before / after for review | `~/Downloads/dynamesh_figures_before_after/11_teaser.png` |
| the 24 aligned panels actually embedded | `~/Downloads/dynamesh_teaser_greyII/panels/` |
| raw renders + ground-truth strip frames | `~/Downloads/dynamesh_teaser_greyII/renders/` |
| flat renders of both layouts | `~/Downloads/dynamesh_teaser_greyII/ba_{before,after}.png` |
| ORIGINAL (untouched) | `~/Downloads/draw.io_xml_files/dynamesh_teaser-continuous_waves-final.drawio.xml` |

## How the XML was rebuilt

`~/Projects/dynamesh_page/tools/figtool.py`: `extract` the original to a dir,
sort each band's slots by x, `align` each new render onto the matching old panel
canvas, write `"new": <path>` into `index.json`, then `swap`. Slot order, left to
right, by original panel index:

- strip `13 12 11 10 25 26 27 24`
- row 1 `17 16 15 14 00 01 04 06`
- row 2 `21 20 19 18 02 03 05 07`

Bands are identified by y: strip 90-180, row 1 500-520, row 2 695-710. The four
film-strip graphics have `count > 1` and are left alone.

## Gotchas that cost real time

- **`render_sweep.sbatch` does NOT set `NO_SIDE_SHADOW`; `render_clip.sbatch`
  does.** Every candidate sweep has the side panel casting a second shadow. Only
  judge final look from `render_clip.sbatch` output.
- **`mcfm_blend.py` staleness.** guanc's copy was Aug 21 and rejects `v2_G`.
  Synced from Raj's `after_submission/rung27_mcfm_temporal_only/mcfm_blend.py`
  (Aug 29, adds v2_E..v2_I); backup at `mcfm_blend.py.bak.2026-09-16`.
- **`export_verts.sbatch` does not exclude r003** the way the Blender sbatch
  headers do. r003 has a bad GPU and fails with an uncorrectable ECC error.
- **Silhouette IoU must exclude the shadow.** A naive brightness threshold merges
  object and shadow, and shadows do not mirror, so the score measures the wrong
  thing. Mask on saturation plus darkness instead.
- Queue: the account regularly sits at 90+ jobs with fairshare throttled and many
  held. Check `squeue -h -u guanc -o %r | sort | uniq -c` before blaming a job.

## Open items

- Strip frames sit small in their slots; `figtool align` size-matches to the old
  object bbox and the old strip images were framed tighter. Crop the ground-truth
  frames harder before aligning if the user wants them to fill more.
- Row 2 at az295 is fairly side-on at any elevation, so it reads flatter than
  row 1's spread X. Inherent to Diagonal B's azimuth.
- The rebuilt XML has not been copied into `draw.io_xml_files/`; that folder holds
  the user's originals and they have not said where the new one should live.
