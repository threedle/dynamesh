#!/bin/bash
#SBATCH --job-name=rclip
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --exclude=r003
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/net/projects/ranalab/guanc/blender/logs/clip_%j.log
#SBATCH --error=/net/projects/ranalab/guanc/blender/logs/clip_%j.log
set -eo pipefail
export NO_SIDE_SHADOW=1   # chosen look: side panel lights but does not cast
B=/net/projects/ranalab/guanc/blender
# PLYDIR OUTDIR FRAMES AZ EL DIST SCALE EXTRA passed via --export
$B/blender-3.2.1-linux-x64/blender $B/itai_render/blend_files/blank.blend --background \
  --python $B/itai_render/render_anim_colors.py -- \
  "$PLYDIR" "$OUTDIR" --frames "${FRAMES//./,}" --az "$AZ" --el "$EL" \
  --dist "${DIST:-3.2}" --scale "${SCALE:-1.55}" \
  --resolution ${RES:-768 768} ${EXTRA:-} 2>&1 | grep -vE "^Fra:|Saved:"
