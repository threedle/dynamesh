#!/bin/bash

# Define a download function
release_download()
{
  wget --no-check-certificate -O $2 "$1"
  if ! unzip -tq $2 > /dev/null 2>&1; then
    echo "ERROR: $2 is not a valid zip. Download it manually from:" >&2
    echo "       https://github.com/threedle/dynamesh/releases/tag/demo-v1" >&2
    rm -f $2
    exit 1
  fi
}

# Demo assets
release_download https://github.com/threedle/dynamesh/releases/download/demo-v1/dynamesh_demo_spot_lava.zip dynamesh_demo_spot_lava.zip
unzip -q -o dynamesh_demo_spot_lava.zip -d . && rm dynamesh_demo_spot_lava.zip

# The archive unpacks into the layout the commands below expect:
#   ./meshes/spot.obj                              the untextured shape
#   ./demo/spot_lava/frames/frame_0001.png ...     the reference video, 150 frames
#   ./demo/spot_lava/targets/frames/gt_*.png       the loss targets
#   ./demo/spot_lava/targets/frames/valid_*.npy    which target pixels are real
#   ./demo/spot_lava/targets/render_mask.npy       the silhouette GATE-align checks
#   ./demo/spot_lava/run/config.json               the run configuration
#   ./demo/spot_lava/run/ckpts/lora_best.pt        the trained adapter
