#!/bin/bash
#SBATCH --job-name=rsweep
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --exclude=r003
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/net/projects/ranalab/guanc/blender/logs/sweep_%j.log
#SBATCH --error=/net/projects/ranalab/guanc/blender/logs/sweep_%j.log
set -eo pipefail
B=/net/projects/ranalab/guanc/blender
for az in ${AZLIST:-0 30 60 90 120 150 180 210 240 270 300 330}; do
  $B/blender-3.2.1-linux-x64/blender $B/itai_render/blend_files/blank.blend --background \
    --python $B/itai_render/render_anim_colors.py -- \
    "$PLYDIR" "$OUTDIR/az${az}el${EL}" --frames "${FRAME:-75}:${FRAME:-75}" --az $az --el ${EL:-0} --scale 1.0 \
    --resolution 320 320 --samples 20 ${EXTRA:-} 2>&1 | tail -1
done
