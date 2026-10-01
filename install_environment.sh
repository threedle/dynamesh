#!/usr/bin/env bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 # install PyTorch
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html # install Kaolin
pip install numpy==2.2.6 scipy==1.15.3 einops==0.8.2 tqdm==4.68.2 # install additional packages
pip install scikit-image==0.25.2 # install scikit-image, used for the masked SSIM in train.py
pip install imageio==2.37.3 imageio-ffmpeg opencv-python-headless==4.13.0.92 pillow==12.2.0 # install additional packages
pip install trimesh==4.12.2 plyfile==1.1.4 pymeshfix==0.18.1 igraph==1.0.0 # install additional packages
pip install open3d==0.19.0 # install additional packages
pip install xatlas==0.0.11 # install the UV unwrapper that bake.py rasterises into
pip install transformers==5.11.0 safetensors==0.8.0 huggingface_hub==1.19.0 # install the TRELLIS.2 backbone dependencies; huggingface_hub also provides the hf login command
pip install easydict==1.13 kornia==0.8.2 timm==1.0.27 # install modules the TRELLIS.2 package itself imports
pip install spconv-cu118==2.3.8 # install the sparse convolution used by the structured latents
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 # install utils3d, pinned to the commit TRELLIS.2 expects
pip install lpips==0.1.4 # install the perceptual loss, used by train.py when --lpips is passed
pip install ninja==1.13.0 zstandard==0.25.0 # install the build tool the CUDA extensions below compile with, and zstandard, which o_voxel imports but does not declare as a dependency
# --no-deps on all three below is required, not optional. Their declared
# dependencies are wrong for this stack: FlexGEMM's pyproject.toml demands
# triton>=3.2.0, but the environment this repository is tested against runs
# triton 3.1.0, the version torch==2.5.1 itself bundles. Without --no-deps, pip
# "satisfies" that demand by installing a newer triton, which in turn pulls a
# newer, incompatible torch (observed: torch replaced by a cu13 build, dragging
# in nvidia-cusparselt-cu13 and a fresh cuda-toolkit package) and silently
# breaks the C++ ABI every extension installed above was just compiled against.
pip install --no-build-isolation --no-deps git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae # install nvdiffrast v0.4.0, built from source: there is no wheel on PyPI
pip install --no-build-isolation --no-deps git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e filelock==3.29.0 # install FlexGEMM (its own torch/triton pins are wrong for this stack) plus the one runtime dependency it actually needs
pip install --no-build-isolation --no-deps "git+https://github.com/rajhansini/TRELLIS.2.git@e5ef863da768991e474e510e4370ec8308f61afc#subdirectory=o-voxel" # install O-Voxel from the TRELLIS.2 commit this repository is tested against

# FlexGEMM's own autotuner.py, at the commit pinned above, calls
# triton.runtime.Autotuner.__init__() with a do_bench argument that
# triton==3.1.0 (the version torch==2.5.1 bundles, and the only one --no-deps
# above keeps installed) does not accept: "Autotuner.__init__() takes from 7
# to 13 positional arguments but 14 were given". FlexGEMM's newer commits need
# a newer triton than this stack can use without breaking torch (see the
# --no-deps note above), so the fix is at the installed file, not the pin.
#
# The file cannot be located by `import flex_gemm`: its own __init__.py
# imports kernel modules that construct the broken Autotuner call at IMPORT
# TIME, so the broken import is exactly what stops that lookup from working.
# Found by searching site-packages directly instead.
python - <<'PYEOF'
import pathlib, site, sys

candidates = []
for base in site.getsitepackages() + [site.getusersitepackages()]:
    candidates += list(pathlib.Path(base).glob('flex_gemm/utils/autotuner.py'))
assert len(candidates) == 1, f'expected exactly one flex_gemm install, found {candidates}'
p = candidates[0]
t = p.read_text()
old = ("        super().__init__(\n"
       "            fn,\n"
       "            arg_names,\n"
       "            configs,\n"
       "            key,\n"
       "            reset_to_zero,\n"
       "            restore_value,\n"
       "            pre_hook,\n"
       "            post_hook,\n"
       "            prune_configs_by,\n"
       "            warmup,\n"
       "            rep,\n"
       "            use_cuda_graph,\n"
       "            do_bench,\n"
       "        )")
new = ("        super().__init__(\n"
       "            fn,\n"
       "            arg_names,\n"
       "            configs,\n"
       "            key,\n"
       "            reset_to_zero,\n"
       "            restore_value,\n"
       "            pre_hook,\n"
       "            post_hook,\n"
       "            prune_configs_by,\n"
       "            warmup,\n"
       "            rep,\n"
       "            use_cuda_graph,\n"
       "        )\n"
       "        if not hasattr(self, 'keys'):\n"
       "            self.keys = key\n"
       "        if hasattr(self, 'num_warmups') and self.num_warmups is None:\n"
       "            self.num_warmups = 25\n"
       "        if hasattr(self, 'num_reps') and self.num_reps is None:\n"
       "            self.num_reps = 100")
assert t.count(old) == 1, (
    f"flex_gemm autotuner.py call site not found verbatim (matched {t.count(old)}x); "
    f"FlexGEMM upstream may have changed again")
p.write_text(t.replace(old, new))
print('patched', p)
PYEOF
# Prove the fix actually took, in a FRESH process (the broken import can only
# be observed in a process that has not already partially imported the
# package), and from a directory with nothing on disk that could shadow a
# top-level module.
FLEX_GEMM_ERR=$( (cd / && python -c "import flex_gemm") 2>&1 )
FLEX_GEMM_RC=$?
if [ $FLEX_GEMM_RC -ne 0 ] && ! echo "$FLEX_GEMM_ERR" | grep -q "Found no NVIDIA driver"; then
  # Any failure counts, not only the one this patch targets: a driver-less
  # build/login node is expected to fail here with "no NVIDIA driver" (the
  # import gets past construction and only then touches a GPU), which is not
  # a verification failure; anything else, including flex_gemm never having
  # installed at all, must not be reported as success.
  echo "ERROR: import flex_gemm failed after the autotuner patch:" >&2
  echo "$FLEX_GEMM_ERR" >&2
  exit 1
fi
echo 'flex_gemm autotuner patch verified (import succeeds, or fails only for lack of a GPU on this machine)'

# CuMesh IS required, despite an earlier version of this file claiming otherwise.
# That claim was based on an incomplete check: Trellis2TexturingPipeline itself
# only reaches cumesh through postprocess_mesh(), true, but the pipeline module
# unconditionally does `import o_voxel` at its own top level, and
# o_voxel/postprocess.py unconditionally does `import cumesh` at ITS top level.
# So importing Trellis2TexturingPipeline at all -- which train.py, bake.py,
# render.py and export.py all do -- fails immediately without it. Confirmed by
# running the full 30-epoch training end to end: it fails at that import with
# ModuleNotFoundError, not deep inside some code path this repository avoids.
#
# CuMesh's own build fails against CUDA 12.0/12.1 with an unresolved upstream
# bug (JeffreyXiang/CuMesh#37): nvcc rejects a cuda::std::tuple usage as an
# incomplete type, and separately this CUDA version's bundled CUB predates the
# decomposer-based cub::DeviceRadixSort::SortPairs overload CuMesh calls. Two
# independent fixes, both applied here: including <cuda/std/tuple> directly
# supplies the missing type definition, and NVCC_PREPEND_FLAGS (which nvcc
# always honours ahead of its own bundled headers, unlike CPATH, which pip's
# generated -I list outranks regardless of environment) puts a newer CCCL on
# the include path so the newer SortPairs overload resolves.
pip install nvidia-cuda-cccl-cu12==12.6.77 --no-deps # newer CUB/libcu++ headers only; no runtime libs, so torch's own CUDA runtime pin is untouched
CCCL_INC=$(python -c "import nvidia.cuda_cccl, os; print(os.path.dirname(nvidia.cuda_cccl.__file__) + '/include')")
export NVCC_PREPEND_FLAGS="-I$CCCL_INC"
rm -rf /tmp/dynamesh_cumesh_build
# The nested submodule clone (cubvh -> eigen) occasionally fails with a
# transient index-pack error under load; retry a few times before giving up.
CUMESH_CLONED=0
for _try in 1 2 3; do
  git clone --recursive -q https://github.com/JeffreyXiang/CuMesh.git /tmp/dynamesh_cumesh_build \
    && (cd /tmp/dynamesh_cumesh_build && git checkout -q 12289e1062f0603f2f0d0771b02e1395d247f26f \
        && git submodule update --init --recursive -q) \
    && { CUMESH_CLONED=1; break; }
  echo "CuMesh clone attempt $_try failed, retrying..." >&2
  rm -rf /tmp/dynamesh_cumesh_build
done
[ "$CUMESH_CLONED" = "1" ] || { echo "ERROR: could not clone CuMesh after 3 attempts" >&2; exit 1; }
cd /tmp/dynamesh_cumesh_build
python - <<'PYEOF'
import pathlib
p = pathlib.Path('src/clean_up.cu')
t = p.read_text()
old = "#include <cub/cub.cuh>\n"
new = "#include <cub/cub.cuh>\n#include <cuda/std/tuple>\n"
assert t.count(old) == 1, (
    f"clean_up.cu's #include block not found verbatim (matched {t.count(old)}x); "
    f"CuMesh upstream may have changed again")
p.write_text(t.replace(old, new))
print('patched src/clean_up.cu: added #include <cuda/std/tuple>')
PYEOF
pip install --no-build-isolation --no-deps .
cd - > /dev/null
rm -rf /tmp/dynamesh_cumesh_build
python -c "import cumesh; cumesh.CuMesh(); print('cumesh imports and constructs OK')"

# flash_attn is REQUIRED, not optional: trellis2/modules/sparse/config.py
# defaults ATTN='flash_attn', and full_attn.py's flash_attn branch does
# `import flash_attn` directly with no fallback -- training reaches this on
# the very first self-attention forward pass. Confirmed by a full training run
# without it: ModuleNotFoundError, well after every other import succeeded.
#
# This exact prebuilt wheel is what the tested environment uses; it is built
# for torch 2.4 but works against torch==2.5.1 here (the two are ABI-compatible
# for this wheel). Building from source instead is possible but slow and more
# likely to fail on an unfamiliar host; use the wheel unless it does not fit
# your platform.
#
# It will not import on a machine with no NVIDIA GPU: flash_attn_2_cuda.so
# needs a newer GLIBC than a typical login/build node ships and only resolves
# on the actual GPU node. That is expected and not a sign the install failed.
#
# To use TRELLIS.2's alternate backend instead (untested against this
# repository's shipped checkpoints, and not guaranteed to reproduce them
# bit-for-bit): export ATTN_BACKEND=xformers and pip install xformers instead
# of the two lines below.
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-deps
