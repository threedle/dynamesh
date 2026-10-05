"""
train.py — fit a per-shape LoRA adapter on cross-attention and self-attention
================================================================================
TRELLIS.2's own source is imported, never edited.

WHERE THE ADAPTER GOES, AND WHY BOTH MODULES
  Adapting only the projections inside cross-attention (kv / qkv / kvo / qkvo)
  compares choices within one module. The stronger claim, that cross-attention is
  the right PLACE, needs a comparison across modules, and self-attention is the
  only one that matters: it is the propagation mechanism the whole method leans
  on, which is also why earlier versions of this method deliberately froze it.
  --targets selects either or both, so that comparison can be made directly.

THE MECHANISM, AND WHAT IT PREDICTS
  Block order is self_attn -> cross_attn -> mlp (modulated.py:149-157). So:

    cross_attn delta reads c_f, the frozen DINOv3 tokens of frame f. Constant
      input, IDENTICAL for all 30 blocks, varying only with the frame index. The
      edit changes WHAT THE IMAGE SAYS; frozen self-attention then distributes it.
    self_attn delta reads h_i, block i's own voxel features. It changes HOW
      VOXELS TALK TO EACH OTHER. At block 0, h_0 is noise + shape latent only --
      ZERO frame information -- so a self-attn adapter there cannot be frame
      conditioned at all; it can only reshape propagation.

  The loss sees ONE camera. Gradient reaches most latent voxels (measured
  1,659/1,999 = 83% by the entropy diagnostic, job 2177639) but every one of
  them carries FRONT-VIEW ERROR ONLY. Training the distributor on that signal
  lets gradient descent reroute texture toward the front.

  PREDICTION, recorded BEFORE the run so it cannot be retrofitted:
      front-view PSNR   the self-attention arm HIGHER than the qkvo arm  (more parameters, and the
                        ladder is monotone in adapted-parameter count, 8/8, no
                        plateau yet)
      off-axis          the self-attention arm WORSE -- the turntable is the verdict, not PSNR
  If BOTH improve, the design premise is wrong and that is a real finding. Say so.

WHERE EXACTLY
  models['tex_slat_flow_model_512'].blocks[0..29]
      .cross_attn.to_q    Linear(1536, 1536)   <- LoRA   (as before)
      .cross_attn.to_kv   Linear(1024, 3072)   <- LoRA   image -> 3D keys/values
      .cross_attn.to_out  Linear(1536, 1536)   <- LoRA
      .self_attn.to_qkv   Linear(1536, 4608)   <- LoRA   NEW IN THIS ARM
      .self_attn.to_out   Linear(1536, 1536)   <- LoRA   NEW IN THIS ARM
      .mlp, .adaLN_modulation                      FROZEN

  --targets qkvo+sa  = the qkvo arm's full cross-attn set PLUS self-attention.
                       77,824/block x 30 = 2,334,720 params. This is the arm to run.
  --targets sa       = self-attention ALONE, no cross-attn adapter at all. The
                       pure "put it somewhere else" control. 36,864/block x 30 =
                       1,105,920 params.
  --targets kv+sa    = the kv arm's set plus self-attention, for a matched pair against
                       the kv arm at the low end of the ladder.

  Both new layers are hooked with the SAME zero-init B as every other LoRA here, so
  GATE-plain still proves the adapter is the exact identity at step 0.

HOW THIS STAYS COMPARABLE TO THE LADDER
  Every arm in the reported grid ran the same settings: --recon l1 --lpips
  --w-lpips 0.1, NO --conf-weight, --render-res 960, rank 4, 30 epochs, seed 42,
  lr 1e-4, changing --targets alone. A run with a different loss is not a matched
  cell and cannot be compared against the others.

  NOTE the trust map is OFF, deliberately. --conf-weight was measured to LOSE in
  5/5 matched pairs (margins 0.492-1.146 dB), so every arm after the confidence-weighted arm is
  unweighted. Passing --conf-weight here would also make _ARM fall through to 20
  and silently mislabel the run.

ARM NUMBERING
  _TARGET_ARM below maps each target set to an integer that goes into the run
  label and the config hash. The numbers are historical and are never reused or
  reassigned: an existing entry means an existing set of runs on disk, so the
  table is append-only and self-attention took the next free integer.

WHY HERE AND NOT IN THE DECODER
  the decoder-side arm put the adapter in the mesh DECODER. Measured consequence: the training
  camera sees 16.8% of the vertices, the adapter edits 100% of them at equal
  strength (ratio 1.02), and R^2 of the edit against the training camera's image
  axis is 0.16-0.23 versus ~0.00 for the frozen colour. It learned a projection,
  because the decoder has NO cross-attention -- the image never reaches it, and
  all of TRELLIS's 3D reasoning already finished upstream.

  Here the adapter sits INSIDE the texture flow, on cross-attention, which is
  where image evidence enters the 3D. Downstream of it are 30 blocks of
  self-attention (frozen) that propagate across voxels. The bet: an edit
  admitted at cross-attention gets distributed by TRELLIS's own machinery to
  voxels the camera never saw. That bet is what this experiment tests, and the
  turntable decides it -- not PSNR from the training view.

WHERE EXACTLY
  models['tex_slat_flow_model_1024'].blocks[0..29]
      .cross_attn.to_q    Linear(1536, 1536)   <- LoRA
      .cross_attn.to_kv   Linear(1024, 3072)   <- LoRA   image -> 3D keys/values
      .cross_attn.to_out  Linear(1536, 1536)   <- LoRA
      .self_attn.*                                 FROZEN  (3D coherence)
      .mlp, .adaLN_modulation                      FROZEN
  rank 4 -> 40,960/block x 30 = 1,228,800 parameters. B is zero-init, so step 0
  is the unmodified model.

  --targets cross+mlp adds mlp.mlp[0]/mlp.mlp[2] for the ablation. Excluding the
  MLP is a SCOPING decision, not a principled one: the MLP would also propagate
  (29 blocks of self-attention follow it). It is left out first so the claim has
  one variable. an early arm that adapted the MLP alongside attention adapted the MLP alongside attention and reached
  21.3 dB, so MLP adaptation is demonstrably not harmful -- just unisolated.

THE ONE-STEP TRAINING DESIGN, AND WHY
  Backpropagating through all 12 ODE steps means 12 sequential 1.3B forward
  passes with activations retained. Instead, per step:

      run the ODE to a RANDOM timestep k        under no_grad   (~6 evals avg)
      ONE model call at k                       WITH grad
      x_0_hat = sampler._pred_to_xstart(x_k, t_k, v)
      decode -> PBR voxels -> sample at mesh vertices -> render -> loss

  Sampling k fresh each step means the LoRA still sees every timestep. The
  x_0 formula is the sampler's OWN helper, not a reimplementation, so the
  rectified-flow parameterisation cannot drift from TRELLIS.2's.

GRADIENT HEALTH IS THE POINT OF THIS RUN
  The gradient now travels back through the texture decoder AND a flow
  evaluation, both in bf16, to a LoRA sitting behind cross-attention's softmax
  where gradients are already small. v1 hit exactly this (fp16 flush-to-zero in
  dec_mesh, fixed with LOSS_SCALE=4096). So this script logs, every epoch:

      per-target   |dB| for to_q / to_kv / to_out separately
      per-block    |dB| for all 30 blocks, so a dead early half is visible
      zero count   how many of the 90 B matrices received EXACTLY zero
      dynamic range min/max/median across matrices
      grad/param   |dB| / |B|, i.e. is the step size meaningful

  and GATE-grad refuses to start training if ANY B matrix gets zero gradient.
  Failing in 60 seconds beats discovering it after four hours.

Usage:
  python train.py --mesh mesh.ply --epochs 30
  python train.py --mesh mesh.ply --smoke      # 2 epochs, 8 frames
"""

import argparse, json, math, os, sys, time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Paths resolve through modules/paths.py, never to one machine's filesystem.
from modules.paths import add_trellis2_to_path, hf_home, set_offline_from_env

hf_home()
set_offline_from_env()
TRELLIS2 = add_trellis2_to_path()
_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--mesh', required=True)
ap.add_argument('--gt-render-dir', default=None,
                help='rendered backprojected GT — same mesh, same camera, so the '
                     'target shares our silhouette by construction')
ap.add_argument('--gt-dir', default=None,
                help='reference-video frames this adapter is fitted to')
ap.add_argument('--rank',       type=int, default=4)
ap.add_argument('--lora-alpha', type=float, default=4.0,
                help="LoRA scaling numerator: delta W is multiplied by "
                     "alpha/rank (Hu et al. 2021, sec 4.1). Keep FIXED across a "
                     "rank sweep — that is the whole point. Without it dW = B@A "
                     "sums `rank` outer products, so a larger rank takes larger "
                     "effective steps and the sweep measures rank and step size "
                     "together. Default 4 makes the factor exactly 1.0 at rank 4, "
                     "reproducing every run made before the sweep bit for bit.")
ap.add_argument('--targets',    default='qkv',
                choices=['v', 'kv', 'kvo', 'qkv', 'qkvo', 'cross', 'cross+mlp',
                         'sa', 'kv+sa', 'kvo+sa', 'qkv+sa', 'qkvo+sa'],
                help="v    = the VALUE half of to_kv only, with the KEY half "
                     "locked (the qkvo arm). Cross-attention splits into a 'where' "
                     "pathway (K, which sets the attention map) and a 'what' "
                     "pathway (V, which sets appearance). Perfusion "
                     "(arXiv:2305.01644) shows training K makes attention "
                     "'spread across the entire image rather than focusing on "
                     "the object'; we measured exactly that here — CLS mass "
                     "0.546 -> 0.567 and flicker -28%% after adapting. to_kv is "
                     "a FUSED Linear(1024, 3072) whose output reshapes to "
                     "(2, heads, dim) before k, v = unbind(dim=2), so rows "
                     "0:1536 are K and 1536:3072 are V. This option adapts the "
                     "V rows and adds exact zeros to the K rows; "
                     "kv   = to_kv ONLY. With the mesh and the noise both fixed, "
                     "q is near-static across frames while k,v carry the only "
                     "per-frame signal, so this is the sole place a frame-dependent "
                     "edit can live; "
                     "qkv  = + to_q (an early v1 arm's set on v1); "
                     "qkvo = + to_out (what v1's cross-attention arm used, 1,228,800 params); "
                     "'cross' is an alias for qkvo kept so v1's cross-attention arm configs still load")
ap.add_argument('--blocks', default='all', choices=['all', 'early', 'mid', 'late'],
                help='which contiguous third of the 30 blocks carries the adapter. '
                     'An edit at block 0 has 29 self-attention blocks downstream to '
                     'spread it; one at block 29 has none and is effectively a '
                     'decoder-side edit. That contrast is the point of this arm.')
ap.add_argument('--epochs',     type=int, default=30)
ap.add_argument('--n-frames',   type=int, default=150)
ap.add_argument('--lr',         type=float, default=1e-4)
ap.add_argument('--loss-scale', type=float, default=4096.0)
ap.add_argument('--grad-clip',  type=float, default=1.0)
ap.add_argument('--w-lpips',    type=float, default=0.1,
                help="Weight on the perceptual term. INERT unless --lpips is "
                     "passed. The default stays 0.1 rather than 0.0 because this "
                     "value is already in the config hash of every run ever made; "
                     "changing it would silently re-hash all of them.")
ap.add_argument('--lpips',      action='store_true',
                help="Add a perceptual term to the data loss. Rationale: L1 and "
                     "L2 both compare pixel-to-pixel, which is harsh about a "
                     "few-pixel shift and FORGIVING ABOUT BLUR -- backwards from "
                     "what we care about. LPIPS compares activations of a frozen "
                     "pretrained AlexNet, where a blurred crack and a sharp crack "
                     "differ sharply even when their pixel means agree, and a "
                     "slightly displaced crack does not. Fed the silhouette bbox "
                     "crop at NATIVE resolution -- resizing to 256 would discard "
                     "exactly the high frequencies this term exists to recover.")
ap.add_argument('--recon',      default='l2', choices=['l2', 'l1'],
                help="THE PER-PIXEL PENALTY. l2 (the default) is MEAN-seeking: "
                     "its minimiser at a pixel the "
                     "model is unsure about is the average of the plausible "
                     "values, i.e. a blend of crack and rock. Uncertainty is "
                     "highest at texture boundaries, so the hedging concentrates "
                     "exactly on edges -- which is what blur IS. Measured on "
                     "teapot_lava2: interior edge detail 5.44 against the "
                     "target's 9.82 WITH every pixel supervised, so this is not "
                     "a propagation limit and not a resolution limit. l1's "
                     "minimiser is the conditional MEDIAN, which commits to one "
                     "value instead of splitting the difference. 3D Gaussian "
                     "Splatting uses L1 (+0.2*D-SSIM) for this reason; we have "
                     "been using the other one. NOTE PSNR = 10*log10(1/MSE), so "
                     "an l1 run is no longer optimising the reported metric and "
                     "WILL score lower on it -- judge it on edge detail.")
ap.add_argument('--seed',       type=int, default=42)
ap.add_argument('--resolution', type=int, default=512, choices=[512, 1024],
                help="512 by default, and this is MEASURED, not a preference. At 1024 the "
                     "texture decoder's backward graph does not fit on a 44 GiB A40: "
                     "an identical 43.6 GiB peak was measured under "
                     "explicit_gemm/frozen, implicit_gemm/grad-weights and "
                     "implicit_gemm_splitk/grad-weights — i.e. it is the graph size, not "
                     "the conv algorithm. At 512 the same probe reaches backward. "
                     "TRELLIS.2 ships tex_slat_flow_model_512 as a first-class option and "
                     "the frozen baseline scored the same at both (PSNR 8.149 vs 8.148).")
ap.add_argument('--render-res', type=int, default=518,
                help='MUST equal the resolution the targets were rendered at '
                     '(518). The old 512 default crashed the smoke run with a '
                     '(512,512,3) vs (518,518,3) broadcast error; GATE-align '
                     'now catches it before any training happens.')
ap.add_argument('--conf-quantile', type=float, default=-1.0,
                help="the regularised arms. Set tau from THIS object's own incidence "
                     "distribution — discard the least reliable q fraction of "
                     "observations — instead of a hardcoded angle. A flat panel "
                     "and a teapot then get different absolute thresholds "
                     "automatically, which is what lets the method transfer. "
                     "-1 keeps the fixed --conf-tau (the confidence-weighted arm behaviour).")
ap.add_argument('--lambda-con', type=float, default=0.0,
                help="the regularised arms/22. Weight on the MAGNITUDE CEILING as a FRACTION of "
                     "the data loss (a ratio, not an absolute — that is what makes "
                     "it object-independent). The term is relu(||d||^2 - m^2) where "
                     "m is the p99 edit magnitude over the VERIFIED voxels, so it "
                     "is exactly ZERO inside the budget: an unsupervised voxel may "
                     "edit in any direction and any pattern, just not harder than "
                     "the hardest edit seen on measured surface. It is a CEILING, "
                     "not a target — this is NOT 'stay at frozen', which was "
                     "measured to yield a dark band at alpha=0 because frozen is "
                     "globally dark (0.062 vs a truth near 0.31). "
                     "(The earlier help here described a mean-matching design that "
                     "was replaced by this hinge and never ran.)")
ap.add_argument('--lambda-smo', type=float, default=0.0,
                help='the regularised arms ONLY — leave at 0. Pairwise neighbour-consistency on '
                     'the edit. Its minimiser, chained across the voxel graph, is '
                     'a single CONSTANT edit everywhere, which collapses the output '
                     'toward frozen: the regularised arms measured 19.51 -> 12.70 dB and SSIM '
                     '0.728 -> 0.404 against the confidence-weighted arm. the neighbour-consistency arm = lambda_con > 0 with '
                     'this at 0.')
ap.add_argument('--reg-k-voxels', type=float, default=2.0,
                help='confidence falls off over this many VOXELS from the nearest '
                     'resolved surface point — resolution-relative, not a world '
                     'distance, so it is scale free.')
ap.add_argument('--reg-gamma', type=float, default=1.0)
ap.add_argument('--conf-weight', action='store_true',
                help="the confidence-weighted arm. Weight each pixel's loss by how trustworthy that "
                     "observation is, instead of treating every covered pixel as "
                     "equally authoritative. w = cos(n,v)/footprint, zeroed below "
                     "--conf-tau. This is the Unstructured Lumigraph blending "
                     "field (Buehler et al. SIGGRAPH 2001) — angular deviation, "
                     "resolution, and discarding poorly-positioned observations — "
                     "applied to a training loss rather than to blending.")
ap.add_argument('--conf-ramp-qlo', type=float, default=-1.0,
                help='the confidence-ramp arm. Lower end of the feathered cutoff, as a QUANTILE of '
                     "this object's incidence distribution over the covered "
                     'pixels. Below it the weight is exactly 0. -1 disables the '
                     'ramp and restores the confidence-weighted arm hard step at --conf-tau.')
ap.add_argument('--conf-ramp-qhi', type=float, default=-1.0,
                help='the confidence-ramp arm. Upper end of the feathered cutoff, same units. At '
                     'and above it the weight is unchanged from the confidence-weighted arm. Between '
                     'the two the weight follows a smoothstep, so both the weight '
                     'and its slope are continuous. Quantiles rather than absolute '
                     'cosines because the same cosine selects wildly different '
                     'fractions of the surface on different geometry.')
ap.add_argument('--conf-tau', type=float, default=0.25,
                help='discard observations with cos(n,v) below this outright. The '
                     'literature hard-zeros rather than soft-weighting because '
                     'there are MANY grazing pixels and a soft weight still lets '
                     'them sum to a real share of the loss.')
ap.add_argument('--lambda-smooth', type=float, default=0.0,
                help="weight on the neighbour-consistency term (the confidence-weighted arm). 0 "
                     "disables it, reproducing the kv arm/18/19 exactly. The term "
                     "pulls each voxel's base_color toward its 3D neighbours, "
                     "weighted by how UNRELIABLY that voxel is observed, so a "
                     "grazing-angle voxel inherits colour from the well-observed "
                     "surface next to it instead of drifting on a shared LoRA "
                     "push nothing checks. Measured motivation: 78.3%% of "
                     "blown-out vertices sit at grazing angles vs 35.8%% of "
                     "correct ones, while visibility is FLAT across brightness "
                     "bands — so the failure tracks incidence angle, not "
                     "occlusion.")
ap.add_argument('--smooth-radius', type=float, default=0.02,
                help='only voxels within this distance (normalised units, the '
                     'object spans ~1.0) of a pixel the camera actually resolved '
                     'are smoothed. Beyond it the weight is exactly 0, so the '
                     'unobserved back is left entirely to cross-attention.')
ap.add_argument('--smooth-gamma', type=float, default=1.0,
                help='exponent on the unreliability weight (1-cos)^gamma')
ap.add_argument('--held-every', type=int, default=0,
                help='0 (default) = train and evaluate on ALL frames, which is '
                     'what the task asks for. N>0 holds out every Nth frame '
                     'instead; earlier arms hardcoded N=10 and lost 15 frames '
                     'of supervision for a generalisation claim nobody made.')
ap.add_argument('--smoke',      action='store_true')
ap.add_argument('--out-dir',    default=None, type=Path)
ap.add_argument('--alignment',  type=Path, default=None,
                help="the decoder-side arm's solved mesh->video registration, applied before rendering")
ap.add_argument('--align-mesh', action='store_true',
                help="apply the decoder-side arm's transform. OFF by default because the meshes in "
                     "the export step already applies it to the vertices. Pass "
                     "this only for a mesh that has NOT been through that export.")
ap.add_argument('--no-grad-ckpt', action='store_true',
                help='disable gradient checkpointing (faster, needs >44 GB — OOMs on an A40)')
ap.add_argument('--decoder-grad-weights', action='store_true', default=True,
                help='set decoder weights requires_grad=True (NOT trained) so flex_gemm '
                     'does not return grad_weight=None and crash on reshape')
ap.add_argument('--no-decoder-grad-weights', dest='decoder_grad_weights',
                action='store_false')
ap.add_argument('--spconv-algo', default='implicit_gemm_splitk',
                choices=['implicit_gemm', 'implicit_gemm_splitk', 'explicit_gemm',
                         'masked_implicit_gemm', 'masked_implicit_gemm_splitk'],
                help="flex_gemm sparse-conv algorithm. TRELLIS.2 ships "
                     "masked_implicit_gemm_splitk, whose BACKWARD needs neighbour-cache "
                     "fields that are only stored when the first call on those coords "
                     "requires grad — which is never true here, because encode_shape_slat "
                     "runs frozen first. MEASURED on an A40 at res 512: explicit_gemm dies "
                     "with 'flip_cuda not implemented for UInt32'; masked_*_splitk raises a "
                     "Triton CompilationError; implicit_gemm_splitk WORKS (17.07 GiB, "
                     "sub-second per step once Triton has compiled).")
# Choices come from modules/mcfm.py, never a copy. A hardcoded list here silently
# froze at W=2/3 and rejected v2_E at argparse time, after the job had already
# been allocated a GPU -- the operator supported the window, the CLI did not.
from modules.mcfm import MODES as _MCFM_MODES, ALIASES as _MCFM_ALIASES
ap.add_argument('--mcfm', default=None,
                choices=list(_MCFM_MODES) + list(_MCFM_ALIASES),
                help="MCFM temporal token blending, applied to the cached DINOv3 "
                     "conditioning BEFORE training so the flow model never sees "
                     "vanilla per-frame tokens. Readable names, preferred: "
                     "temporal_only_w11 (= v2_G) is what the paper reports; it blends "
                     "token i across the window ONLY, leaving spatial mixing to the "
                     "model's own cross-attention. joint_spatiotemporal_* (= v3) pools "
                     "over time AND space and is 1.5-2.6 dB worse. "
                     "spatial_then_temporal_* and temporal_then_spatial_* are also "
                     "implemented here. Windows: w2=[t,t+1], w3=[t-1,t,t+1], up to "
                     "w11=[t-5..t+5]. Parameter-free. "
                     "UNSET (the default) is byte-identical to every previous run: "
                     "it is absent from the hash, absent from the label, and the "
                     "blend call returns the dict unchanged.")
args = ap.parse_args()

# Normalise the readable name to the short code IMMEDIATELY, before anything reads
# it. args.mcfm feeds both the config hash and the run-directory label, so doing
# this here means 'temporal_only_w3' and 'v2_D' produce the SAME run directory and
# can resume from each other's checkpoints. Doing it later would have silently
# forked every existing run into a new directory.
if args.mcfm is not None:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from modules.mcfm import canonical as _mcfm_canonical
    _typed = args.mcfm
    args.mcfm = _mcfm_canonical(args.mcfm)          # raises on spatial_then_temporal_*
    if _typed != args.mcfm:
        print(f'[MCFM] {_typed} -> {args.mcfm}  (same config hash, same run dir)',
              flush=True)

_TSET = {'v':   ('to_v',),
         'kv':  ('to_kv',),
         'kvo': ('to_kv', 'to_out'),
         'qkv': ('to_q', 'to_kv'),
         'qkvo': ('to_q', 'to_kv', 'to_out'),
         'cross': ('to_q', 'to_kv', 'to_out'),
         'cross+mlp': ('to_q', 'to_kv', 'to_out'),
         # the self-attention arm. 'sa_*' names are DISTINCT from the cross-attention ones on
         # purpose: cross_attn and self_attn both own a submodule literally called
         # to_out, and a shared key would put one LoRALayer on two different
         # Linears -- silently tying two edits together and making the ablation
         # meaningless. Distinct names also make grad_report's per-target block
         # separate them, which is how a dead self-attn half becomes visible.
         'sa':      ('sa_qkv', 'sa_out'),
         'kv+sa':   ('to_kv', 'sa_qkv', 'sa_out'),
         'kvo+sa':  ('to_kv', 'to_out', 'sa_qkv', 'sa_out'),
         'qkv+sa':  ('to_q', 'to_kv', 'sa_qkv', 'sa_out'),
         'qkvo+sa': ('to_q', 'to_kv', 'to_out', 'sa_qkv', 'sa_out')}[args.targets]
# Does this arm touch self-attention at all? Used by the arm number, the banner
# and GATE-selfattn, so it is computed once here rather than re-derived.
_HAS_SA = any(t.startswith('sa_') for t in _TSET)
# WHICH TARGET SET GETS WHICH ARM NUMBER.
# qkv and qkvo once fell through to the kv arm's number, which made them
# indistinguishable in the label. They take 18 and 19.
#   18 = to_q + to_kv           "adapt the question AND the image, not the output"
#   19 = to_q + to_kv + to_out  "adapt all three"
# The old occupants move rather than being overwritten, so no number ever means
# two things: kvo was 18 (run ac8beea4, 23.37 dB) and v was 19 (run 8dc4a430,
# 21.41 dB). Those directories keep their names -- only future runs are affected.
_TARGET_ARM = {'kv': 17, 'qkv': 18, 'qkvo': 19, 'kvo': 25, 'v': 26,
                # the self-attention arm = ANY arm that adapts self-attention. See _ARM below:
                # _HAS_SA short-circuits the whole chain, so a self-attn arm can
                # never be mislabelled 20/21/22/23/24 by an unrelated flag.
                'sa': 27, 'kv+sa': 27, 'kvo+sa': 27, 'qkv+sa': 27, 'qkvo+sa': 27}

_NB = 30                       # tex_slat_flow_model_512/1024 both have 30 blocks
_THIRD = _NB // 3
ACTIVE = {'all':   list(range(_NB)),
          'early': list(range(0, _THIRD)),
          'mid':   list(range(_THIRD, 2 * _THIRD)),
          'late':  list(range(2 * _THIRD, _NB))}[args.blocks]

EPOCHS   = 2 if args.smoke else args.epochs
N_FRAMES = 8 if args.smoke else args.n_frames

# TRAIN AND EVALUATE ON ALL 150. The task is "this mesh, these 150 textures from
# this video" — every frame is part of the deliverable, so withholding some
# spends supervision to answer a generalisation question nobody asked. The old
# every-10th split came from run01_single_path_lora.py and was copied into all 20
# arms unexamined; it cost 15 of 150 frames of signal for nothing.
HELD_OUT = ([] if args.held_every <= 0
            else [f for f in range(5, N_FRAMES + 1, args.held_every)])
TRAIN    = [f for f in range(1, N_FRAMES + 1) if f not in HELD_OUT]
ALL_FRAMES = list(range(1, N_FRAMES + 1))
# per-epoch monitoring only — a full 150-frame eval is 12 ODE steps x 150 and
# would cost more than the epoch itself. The reported numbers use ALL_FRAMES.
MONITOR  = ALL_FRAMES[4::10]

import hashlib
_CFG = dict(variant='trellis2_backproj_lora', loss_region='rendered_gt',
            # _HAS_SA FIRST. Adapting self-attention IS the experiment, so it
            # outranks every other flag: the self-attention arm must never be labelled 20
            # because someone also passed --conf-weight. Kept byte-identical to
            # the _ARM chain below -- if these two ever disagree, config.json
            # and the directory name describe different experiments.
            arm=(27 if _HAS_SA
                  else 24 if (args.conf_ramp_qlo >= 0 and args.conf_ramp_qhi >= 0)
                  else 22 if (args.lambda_con > 0 and args.lambda_smo <= 0)
                  else 21 if (args.lambda_con > 0 or args.lambda_smo > 0)
                  else 20 if args.conf_weight else 23 if args.lambda_smooth > 0
                  else _TARGET_ARM.get(args.targets, 17)),
            conf_ramp_qlo=args.conf_ramp_qlo, conf_ramp_qhi=args.conf_ramp_qhi,
            conf_weight=bool(args.conf_weight), conf_tau=args.conf_tau,
            conf_quantile=args.conf_quantile, lambda_con=args.lambda_con,
            lambda_smo=args.lambda_smo, reg_k_voxels=args.reg_k_voxels,
            reg_gamma=args.reg_gamma,
            lambda_smooth=args.lambda_smooth, smooth_gamma=args.smooth_gamma,
            smooth_radius=args.smooth_radius,
            blocks=args.blocks,
            active=ACTIVE, target_set=list(_TSET), rank=args.rank,
            lora_alpha=args.lora_alpha, targets=args.targets,
            epochs=EPOCHS, n_frames=N_FRAMES, lr=args.lr, seed=args.seed,
            loss_scale=args.loss_scale, resolution=args.resolution,
            w_lpips=args.w_lpips, held_out=HELD_OUT,
            # RECONSTRUCTION PENALTY. Added CONDITIONALLY, and that is deliberate:
            # inserting recon='l2' unconditionally would change the hash of every
            # run ever made, so 230e1d68 / 4ec44438 / bc7a75b0 would stop being
            # reproducible from their own commands. Absent for l2 (the historical
            # behaviour, byte for byte), present for anything else -- which is all
            # that is needed to keep an l1 run out of an l2 run's directory.
            **({} if args.recon == 'l2' else dict(recon=args.recon)),
            # PERCEPTUAL TERM. Conditional for the same reason as recon: absent
            # keeps every historical hash intact, present is enough to keep an
            # lpips run out of a non-lpips run's directory.
            **({} if not args.lpips else dict(lpips=True)),
            # MCFM BLEND. Conditional for the same reason as recon and lpips:
            # absent keeps every historical hash intact. Present is REQUIRED —
            # the four modes differ only here, so without it all four would hash
            # to one directory and resume from each other's checkpoints, the
            # identical failure the mesh/gt_dir comment below records.
            **({} if not args.mcfm else dict(mcfm=args.mcfm)),
            # RENDER RESOLUTION. Same omission as the paths below, same failure
            # mode: the teapot ran at 518 and spot at 960, so a teapot re-run at
            # 960 hashes to the 518 run's directory and resumes from its
            # checkpoints. It also sets how many pixels are supervised and scored
            # (28,559 at 518 vs 191,275 at 960), so it changes what PSNR means and
            # two runs that differ in it are not the same experiment.
            render_res=args.render_res,
            # WHICH OBJECT, AND WHICH DATA. These were absent from the hash until
            # these were added, and it cost two full runs. spot_star and spot_lava differ
            # ONLY in these paths, so they hashed identically, landed in the same
            # run directory, and the job that started second found the first's
            # lora_e001.pt and RESUMED from it — training cow-spot weights on lava
            # data for 29 epochs while both overwrote each other's checkpoints.
            # The symptom was 59 rows in a 30-epoch epoch_metrics.csv.
            # The mesh and the targets ARE the experiment. They belong in its id.
            mesh=str(Path(args.mesh).resolve()),
            # None only on the import-for-registry path (bake/render/export),
            # which never reaches training and never uses RUN_ID. For a real
            # run both are set, so the hash is byte-identical to before.
            gt_dir=str(Path(args.gt_dir).resolve()) if args.gt_dir else None,
            gt_render_dir=(str(Path(args.gt_render_dir).resolve())
                           if args.gt_render_dir else None))
RUN_ID = hashlib.md5(json.dumps(_CFG, sort_keys=True).encode()).hexdigest()[:8]
_ARM = (27 if _HAS_SA          # keep byte-identical to _CFG['arm'] above
         else 24 if (args.conf_ramp_qlo >= 0 and args.conf_ramp_qhi >= 0)
         else 22 if (args.lambda_con > 0 and args.lambda_smo <= 0)
         else 21 if (args.lambda_con > 0 or args.lambda_smo > 0)
         else 20 if args.conf_weight
         else 23 if args.lambda_smooth > 0
         else _TARGET_ARM.get(args.targets, 17))
assert _ARM == _CFG['arm'], (
    f'the two arm-number chains disagree ({_ARM} vs {_CFG["arm"]}) — the directory '
    f'name and config.json would describe different experiments')
# The reconstruction penalty is ORTHOGONAL to the arm: l1 composes with any of them,
# so it is a tag in the label rather than a new arm number -- a
# 1-D counter cannot carry a second axis, and 'the kvo arm' would not tell anyone it
# is the no-trust-map arm. Empty for l2, so existing directory names are
# unchanged; without it two runs would differ only by an opaque hash.
_RTAG  = (('' if args.recon == 'l2' else f'_{args.recon}') + ('_lp' if args.lpips else '')
          + (f'_mcfm{args.mcfm}' if args.mcfm else ''))
LABEL  = f'arm{_ARM}{_RTAG}_{args.blocks}_{args.targets}_r{args.rank}_s{args.seed}_{RUN_ID}'
OUT    = (args.out_dir or (_HERE / 'runs' / LABEL)).resolve()


def _atomic_save(obj, path):
    """torch.save that a SIGKILL cannot leave half-written.

    30 epochs takes ~2.8-4.2h against a 4h partition wall, so these runs are
    EXPECTED to be killed mid-flight and requeued. A kill that lands inside
    torch.save leaves a truncated .pt that torch.load cannot read -- and since
    the resume picks the newest checkpoint, that torn file would be chosen on
    every retry, failing the job identically until --max-retries is spent.

    Writing to a sibling .tmp and os.replace-ing it makes the swap atomic within
    the directory: the reader sees either the previous complete checkpoint or the
    new complete one, never a partial. The glob is 'lora_e*.pt', so a leftover
    .tmp from a kill is invisible to the resume scan rather than being picked up.
    """
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


class _Tee:
    def __init__(self, p):
        self._f = open(p, 'a', buffering=1)
    def write(self, m):
        sys.__stdout__.write(m); self._f.write(m)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()


def _begin_run():
    """Create the run directory and start teeing stdout into it.

    Called from the __main__ guard only. bake.py, render.py and export.py import
    this module with a synthetic argv purely to rebuild the adapter registry that
    matches a checkpoint; they never train. Doing either of these at module scope
    meant every one of those imports created a runs/<label>/ tree it never wrote
    to, and silently replaced the caller's stdout.
    """
    for d in (OUT, OUT / 'logs', OUT / 'ckpts', OUT / 'diag'):
        d.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(OUT / 'train.log')
    sys.stderr = sys.stdout

import numpy as np
import torch
import torch.nn as nn
import trimesh
from PIL import Image

DEVICE = torch.device('cuda')


# ── LoRA ─────────────────────────────────────────────────────────────────────

class LoRALayer(nn.Module):
    """delta = (alpha/rank) * B @ (A @ x). A kaiming-init, B ZERO-init so step 0
    is the identity.

    The alpha/rank factor is LoRA's (Hu et al. 2021, sec 4.1) and exists for
    exactly one reason: dW = B @ A sums `rank` outer products, so without it a
    larger rank produces a LARGER effective update at the same learning rate. A
    rank sweep would then confound rank with effective step size, and "r=32 is
    better" could not be separated from "r=32 took bigger steps".

    alpha defaults to 4, so at rank 4 the factor is EXACTLY 1.0 and every run
    made before this change is reproduced bit for bit -- the neighbour-consistency arm included.
    """
    def __init__(self, in_dim, out_dim, rank, alpha=4.0):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scaling = float(alpha) / float(rank)

    def forward(self, x):
        return (((x.float() @ self.A.T) @ self.B.T) * self.scaling).to(x.dtype)


class CrossAttnLoRA(nn.Module):
    """
    One bundle per block. Only the requested projections are created, so the
    parameter count reflects the target set rather than being masked at runtime —
    an unused-but-present LoRALayer would still be in the optimizer and still be
    reported by GATE 0.
    """
    def __init__(self, ch, ctx, rank, targets=('to_q', 'to_kv', 'to_out'),
                 with_mlp=False, mlp_hidden=None, alpha=4.0):
        super().__init__()
        if 'to_q' in targets:
            self.to_q = LoRALayer(ch, ch, rank, alpha)
        if 'to_kv' in targets:
            self.to_kv = LoRALayer(ctx, 2 * ch, rank, alpha)
        if 'to_v' in targets:
            # half the output width of to_kv: this drives the V rows only, and
            # the K rows receive a literal zero (see the hook in lora_ctx).
            # Note that a to_kv adapter cannot express this — its rank-r A is
            # SHARED by K and V, so their updates are forced through one
            # r-dimensional subspace and cannot be separated after the fact.
            self.to_v = LoRALayer(ctx, ch, rank, alpha)
        if 'to_out' in targets:
            self.to_out = LoRALayer(ch, ch, rank, alpha)
        # ── the self-attention arm: SELF-ATTENTION ───────────────────────────────────────────
        # self_attn.to_qkv is a FUSED Linear(ch, 3*ch) -- q, k and v for the
        # voxel-to-voxel attention in one matrix (modules.py:63). Unlike
        # cross_attn.to_kv there is nothing to key-lock here: all three halves
        # are the same 3D tokens, so splitting them would not separate a 'where'
        # pathway from a 'what' pathway the way it does on the image side.
        # self_attn.to_out is a separate Linear(ch, ch) (modules.py:72) and gets
        # its own adapter, mirroring cross_attn.to_out exactly.
        if 'sa_qkv' in targets:
            self.sa_qkv = LoRALayer(ch, 3 * ch, rank, alpha)
        if 'sa_out' in targets:
            self.sa_out = LoRALayer(ch, ch, rank, alpha)
        if with_mlp:
            self.mlp_fc1 = LoRALayer(ch, mlp_hidden, rank, alpha)
            self.mlp_fc2 = LoRALayer(mlp_hidden, ch, rank, alpha)

    def has(self, name):
        return hasattr(self, name)


class LoRARegistry(nn.Module):
    def __init__(self, n_blocks, ch, ctx, rank, with_mlp, mlp_hidden,
                 targets=('to_q', 'to_kv', 'to_out'), active=None, alpha=4.0):
        super().__init__()
        self.targets = tuple(targets)
        self.active = list(range(n_blocks)) if active is None else list(active)
        self.blocks = nn.ModuleDict({
            str(i): CrossAttnLoRA(ch, ctx, rank, self.targets, with_mlp,
                                  mlp_hidden, alpha)
            for i in self.active})

    def get(self, i):
        k = str(i)
        return self.blocks[k] if k in self.blocks else None



@contextmanager
def lora_ctx(flow_model, registry):
    """
    the self-attention arm: hooks on cross_attn AND self_attn.

    Every earlier arm hooked cross_attn only, on the stated design that
    self_attn is the 3D coherence mechanism and must stay frozen. This file
    deliberately breaks that rule -- that IS the ablation. Nothing else changes:
    .mlp and .adaLN_modulation remain untouched, and an arm whose target set has
    no 'sa_*' entry installs no self-attention hook at all and is bit-identical
    to a cross-attention-only arm.

    SHAPE NOTE, and why the self-attn hooks look like the cross-attn ones.
    SparseMultiHeadAttention._linear (modules.py:78-82) calls the nn.Linear with
    x.feats -- a PLAIN [N, ch] tensor -- not with the SparseTensor. So inp[0] and
    out inside these hooks are plain tensors in both cases, and the same
    `out + delta` form is correct for both. GATE-plain proves it.
    """
    handles = []
    for i, blk in enumerate(flow_model.blocks):
        lb = registry.get(i)
        if lb is None:
            continue

        def _q(mod, inp, out, _lb=lb):
            return out + _lb.to_q(inp[0]).to(out.dtype)

        def _kv(mod, inp, out, _lb=lb):
            return out + _lb.to_kv(inp[0]).to(out.dtype)

        def _o(mod, inp, out, _lb=lb):
            return out + _lb.to_out(inp[0]).to(out.dtype)

        def _v(mod, inp, out, _lb=lb):
            """KEY-LOCKED update: V rows get the delta, K rows get exact zero."""
            d = _lb.to_v(inp[0])                              # [..., ch]
            z = torch.zeros_like(d)
            return out + torch.cat([z, d], dim=-1).to(out.dtype)

        # the self-attention arm. inp[0] here is x.feats, [N, ch] -- the voxel features, NOT the
        # image tokens. That difference is the entire experiment.
        def _sq(mod, inp, out, _lb=lb):
            return out + _lb.sa_qkv(inp[0]).to(out.dtype)

        def _so(mod, inp, out, _lb=lb):
            return out + _lb.sa_out(inp[0]).to(out.dtype)

        # Only hook what the bundle actually carries. Blocks outside the active
        # set get no hook at all and run exactly frozen — which is what makes
        # early/mid/late a clean placement comparison.
        if lb.has('to_q'):
            handles.append(blk.cross_attn.to_q.register_forward_hook(_q))
        if lb.has('to_kv'):
            handles.append(blk.cross_attn.to_kv.register_forward_hook(_kv))
        if lb.has('to_v'):
            handles.append(blk.cross_attn.to_kv.register_forward_hook(_v))
        if lb.has('to_out'):
            handles.append(blk.cross_attn.to_out.register_forward_hook(_o))
        # the self-attention arm. blk.self_attn, NOT blk.cross_attn -- the two modules both own a
        # submodule named to_out, so hooking the wrong parent would look correct
        # and quietly reproduce the kvo arm.
        if lb.has('sa_qkv'):
            handles.append(blk.self_attn.to_qkv.register_forward_hook(_sq))
        if lb.has('sa_out'):
            handles.append(blk.self_attn.to_out.register_forward_hook(_so))

        if hasattr(lb, 'mlp_fc1'):
            def _f1(mod, inp, out, _lb=lb):
                d = _lb.mlp_fc1(inp[0].feats if hasattr(inp[0], 'feats') else inp[0])
                return (out.replace(out.feats + d.to(out.feats.dtype))
                        if hasattr(out, 'feats') else out + d.to(out.dtype))

            def _f2(mod, inp, out, _lb=lb):
                d = _lb.mlp_fc2(inp[0].feats if hasattr(inp[0], 'feats') else inp[0])
                return (out.replace(out.feats + d.to(out.feats.dtype))
                        if hasattr(out, 'feats') else out + d.to(out.dtype))
            handles.append(blk.mlp.mlp[0].register_forward_hook(_f1))
            handles.append(blk.mlp.mlp[2].register_forward_hook(_f2))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ── gradient health, the point of this run ───────────────────────────────────

def grad_report(registry, tag=''):
    """
    Per-target and per-block |dB|. Returns a dict; prints a readable block.
    Reported on B (not A) because B is zero-init: if B's gradient is zero, A's
    is zero too by the chain rule, and nothing can ever move.
    """
    per_target, per_block, all_norms, zeros, names = {}, {}, [], 0, []
    for bi, bundle in registry.blocks.items():
        bn = 0.0
        for tname, layer in bundle.named_children():
            g = layer.B.grad
            gn = 0.0 if g is None else float(g.norm())
            per_target.setdefault(tname, []).append(gn)
            all_norms.append(gn); names.append(f'b{bi}.{tname}')
            bn += gn ** 2
            if gn == 0.0:
                zeros += 1
        per_block[int(bi)] = math.sqrt(bn)

    a = np.array(all_norms)
    nz = a[a > 0]
    rep = dict(
        n_matrices=len(a), n_zero=int(zeros),
        min=float(a.min()), max=float(a.max()),
        median=float(np.median(a)),
        median_nonzero=float(np.median(nz)) if len(nz) else 0.0,
        dynamic_range=float(a.max() / max(nz.min(), 1e-30)) if len(nz) else float('inf'),
        per_target={k: float(np.mean(v)) for k, v in per_target.items()},
        per_block={k: per_block[k] for k in sorted(per_block)},
        dead=[names[i] for i in np.where(a == 0)[0][:12]])

    if tag:
        print(f'  [GRAD {tag}]  {rep["n_zero"]}/{rep["n_matrices"]} matrices at EXACTLY zero')
        print(f'     by target : ' + '  '.join(
            f'{k}={v:.3e}' for k, v in rep['per_target'].items()))
        pb = rep['per_block']
        ks = sorted(pb)
        print(f'     by block  : first4=' + ','.join(f'{pb[k]:.2e}' for k in ks[:4])
              + '   last4=' + ','.join(f'{pb[k]:.2e}' for k in ks[-4:]))
        print(f'     range     : min={rep["min"]:.3e}  median={rep["median"]:.3e}  '
              f'max={rep["max"]:.3e}  spread={rep["dynamic_range"]:.1e}x', flush=True)
        if rep['dead']:
            print(f'     DEAD      : {rep["dead"]}', flush=True)
    return rep


def param_norms(registry):
    out = {}
    for bi, bundle in registry.blocks.items():
        for tname, layer in bundle.named_children():
            out.setdefault(tname, []).append(float(layer.B.detach().float().norm()))
    return {k: float(np.mean(v)) for k, v in out.items()}


# ── camera ───────────────────────────────────────────────────────────────────
# Copied verbatim from TRELLIS v1's trellis/renderers/mesh_renderer.py so the
# view is IDENTICAL to every earlier arm. Re-deriving it would risk a
# sign error that would look like a texture bug. GATE-cam checks it anyway.

RENDER_RES_V1 = 518
_FX_N = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 1., 0., 2.],
                           [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX_N, 0., 0.5],
                           [0., _FX_N, 0.5],
                           [0., 0., 1.]], dtype=torch.float32)
NEAR, FAR = 0.5, 3.0


def intrinsics_to_projection(intr, near, far):
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    ret = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


def rodrigues(rv):
    th = float(np.linalg.norm(rv)) + 1e-12
    k = np.asarray(rv, dtype=np.float64) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


class SurfaceRenderer:
    """
    Fixed mesh, fixed camera -> rasterize ONCE, reuse forever.

    TRELLIS.2 cannot move the mesh and our camera never moves, so the raster,
    the silhouette and each pixel's 3D surface point are all constants. They are
    computed once here. Per training step the only work is sampling the PBR
    voxel field at those constant points, which is where the gradient lives.

    TWO COORDINATE FRAMES, and keeping them straight is the whole trick:
      v_raw  the mesh as it came off TRELLIS v1, optionally the decoder-side arm-aligned.
             Used for RASTERISATION, so the image lines up with the GT video
             exactly as in every earlier arm.
      v_pp   the same vertices after Trellis2TexturingPipeline.preprocess_mesh
             (normalised to [-0.5,0.5], axes permuted). The PBR voxel field
             lives in THIS frame, so v_pp is interpolated across the raster to
             give each pixel its sampling position.
    preprocess_mesh keeps faces and vertex ORDER, so v_raw[i] and v_pp[i] are
    the same vertex. That correspondence is what lets the two frames coexist.
    """

    def __init__(self, v_raw, v_pp, vn_raw, faces, res, resolution):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.ctx = dr.RasterizeCudaContext()
        self.res = res
        self.resolution = resolution

        ext = EXTRINSICS.to(DEVICE)
        intr = INTRINSICS.to(DEVICE)
        full_proj = (intrinsics_to_projection(intr, NEAR, FAR) @ ext).unsqueeze(0)

        v = v_raw.unsqueeze(0)
        v_homo = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
        self.v_clip = torch.bmm(v_homo, full_proj.transpose(-1, -2)).contiguous()
        self.faces = faces.int().contiguous()

        rast, _ = dr.rasterize(self.ctx, self.v_clip, self.faces, (res, res))
        self.rast = rast
        self.mask = (rast[0, ..., 3] > 0)                      # [H, W] bool
        self.mask_idx = torch.nonzero(self.mask.reshape(-1), as_tuple=False).squeeze(1)

        # PER-PIXEL INCIDENCE |n . v|, the confidence this pixel deserves.
        # 1 = surface square to the camera, one pixel covers a small patch and
        # the reading is trustworthy. ~0 = surface edge-on, one pixel stands in
        # for a long stretch of surface and the reading says almost nothing
        # about any individual voxel. Both currently enter the loss with equal
        # authority, which is the measured cause of the blown-out rim.
        # Normals and the eye are taken in the v_raw frame, the one the camera
        # actually lives in. eye = -R^T t from EXTRINSICS = (0, -2, 0).
        _R, _t = EXTRINSICS[:3, :3].to(DEVICE), EXTRINSICS[:3, 3].to(DEVICE)
        self.eye = -(_R.T @ _t)
        _n = dr.interpolate(vn_raw.unsqueeze(0).contiguous(), rast, self.faces)[0]
        _n = torch.nn.functional.normalize(_n[0], dim=-1)              # [H,W,3]
        _p = dr.interpolate(v_raw.unsqueeze(0).contiguous(), rast, self.faces)[0][0]
        _vd = torch.nn.functional.normalize(_p - self.eye, dim=-1)     # [H,W,3]
        self.cos_inc = (_n * _vd).sum(-1).abs().clamp(0, 1)            # [H,W]

        pos = dr.interpolate(v_pp.unsqueeze(0), rast, self.faces)[0]   # [1,H,W,3]
        self.pos_img = pos[0]                                          # [H,W,3]
        self.pos = pos[0].reshape(-1, 3)[self.mask_idx].contiguous()   # [K, 3]
        self.n_px = int(self.mask.sum())

        # FOOTPRINT: the surface area one pixel actually covers, from the screen
        # -space derivatives of the sampling position. This is the quantity the
        # whole diagnosis is about — head-on a pixel covers a small patch, at the
        # rim the surface recedes edge-on and one pixel spans a long stretch, so
        # its reading says almost nothing about any individual voxel. Note this
        # catches things cos(n,v) alone misses: high curvature at the silhouette
        # blows the footprint up even where the cosine is not yet tiny.
        _P = self.pos_img
        _dx = torch.zeros_like(_P); _dy = torch.zeros_like(_P)
        _dx[:, :-1] = _P[:, 1:] - _P[:, :-1]
        _dy[:-1, :] = _P[1:, :] - _P[:-1, :]
        # a difference taken ACROSS the silhouette straddles background and is
        # meaningless, so both neighbours must be covered for the value to count
        _vx = torch.zeros_like(self.mask); _vy = torch.zeros_like(self.mask)
        _vx[:, :-1] = self.mask[:, 1:] & self.mask[:, :-1]
        _vy[:-1, :] = self.mask[1:, :] & self.mask[:-1, :]
        self.fp_valid = _vx & _vy & self.mask
        self.footprint = torch.linalg.cross(_dx, _dy, dim=-1).norm(dim=-1)

    def sample(self, pbr_voxel, clamp=True):
        """
        PBR voxel field -> RGB image. Differentiable w.r.t. pbr_voxel.feats.

        clamp=False FOR THE TRAINING LOSS. torch.clamp has EXACTLY zero gradient
        outside its range, so a voxel the adapter pushes past 1.0 is frozen
        there forever: the loss still reports it as too bright, but no gradient
        can reach it to bring it back down. That is a one-way ratchet into
        white, and it is the mechanism that makes over-brightening permanent
        rather than self-correcting. The metrics and the saved images still
        clamp, because a displayed pixel has to be a colour.
        """
        from flex_gemm.ops.grid_sample import grid_sample_3d
        attrs = grid_sample_3d(
            pbr_voxel.feats,
            pbr_voxel.coords,
            shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
            grid=((self.pos + 0.5) * self.resolution).reshape(1, -1, 3),
            mode='trilinear',
        )                                                       # [K, 6]
        rgb = attrs[..., 0:3].reshape(-1, 3)
        if clamp:
            rgb = rgb.clamp(0, 1)
        base = torch.ones(self.res * self.res, 3, device=DEVICE, dtype=rgb.dtype)
        img = base.index_put((self.mask_idx,), rgb)             # white background
        return img.view(1, self.res, self.res, 3)


def largest_component(mask):
    """
    Keep only the biggest blob of a foreground mask.

    The GT frames carry a few stray dark pixels — frame 4 has ONE at (959, 0)
    valued [243,245,242], just under the 245 threshold. A raw min/max bbox over
    the union of all frames then spans the whole image: measured (0, 238, 802,
    1040) instead of the correct (212, 188, 802, 778), which both pads outside
    the 960px frame and shrinks the teapot inside the conditioning crop.
    Frame 4 has 89 connected components: the teapot at 102,778 px and the next
    largest at 3 px. Taking the biggest component is therefore decisive, not a
    tuned threshold.
    """
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def load_gt(frame_idx, gt_dir, res):
    img = Image.open(Path(gt_dir) / f'frame_{frame_idx:04d}.png').convert('RGB')
    img = img.resize((res, res), Image.LANCZOS)
    a = torch.from_numpy(np.array(img)).float().div(255.0).to(DEVICE)
    return a.unsqueeze(0)                                       # [1,H,W,3]


def psnr_ssim(pred, gt, mask):
    """
    BOTH metrics on the mask. SSIM used to be taken over the whole image, but
    the background is 89.4% of the pixels and is byte-identical white in the
    prediction and the target, so it dominated: measured 0.9846 / 0.9834 /
    0.9804 whole-image against 0.8867 / 0.8708 / 0.8406 on the silhouette.
    skimage has no masked SSIM, so take the full SSIM map and average it over
    the mask — same estimator, honest support.
    """
    from skimage.metrics import structural_similarity as ssim_fn
    p = pred[0].detach().float().cpu().numpy()
    g = gt[0].detach().float().cpu().numpy()
    m = mask.detach().cpu().numpy()
    mse = float(((p - g) ** 2)[m].mean())
    psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
    _, smap = ssim_fn(g, p, channel_axis=2, data_range=1.0, full=True)
    ssim = float(smap[m].mean())
    return psnr, ssim




# ── the sampling schedule, taken from FlowEulerSampler.sample ────────────────
# flow_euler.py:114-118.  steps=12, rescale_t=3.0 -> the knots are FRONT-LOADED
# (seven of twelve above t=0.75). Training timesteps are drawn from THIS list,
# not from a uniform [0,1], so the adapter is never trained where inference
# never goes.
STEPS, RESCALE_T, SIGMA_MIN = 12, 3.0, 1e-5
_ts = np.linspace(1, 0, STEPS + 1)
T_SEQ = (RESCALE_T * _ts / (1 + (RESCALE_T - 1) * _ts)).tolist()
T_PAIRS = [(T_SEQ[i], T_SEQ[i + 1]) for i in range(STEPS)]


def enable_grad_checkpointing(*modules):
    """
    Turn on gradient checkpointing wherever TRELLIS.2 supports it.

    THE FLOW ONLY. Not the decoder — that is deliberate and was measured.

    Both the flow blocks (modulated.py:74-76) and every decoder block type
    (sparse_unet_vae.py:86-88 and friends) call
    torch.utils.checkpoint.checkpoint(..., use_reentrant=False) when their
    use_checkpoint flag is set, and the shipped configs leave it off because
    inference never needs it.

    Checkpointing the DECODER breaks its backward:

        AttributeError: 'SubMConv3dNeighborCache' object has no attribute
                        'valid_signal_i'
                        (flex_gemm/ops/spconv/submanifold_conv3d.py:308)

    flex_gemm's submanifold convolution builds a neighbour cache during forward
    and consumes it during backward. Recomputing the forward under checkpointing
    hands backward a cache that does not carry those fields. The flow is pure
    attention plus SparseLinear — no spconv anywhere — so it checkpoints safely.

    That split is also the one that matters. The first OOM was in the DECODER
    FORWARD, not its backward: the flow's graded evaluation (30 blocks x 1536ch
    x ~9.2k voxels, all activations retained) had already taken the memory the
    decoder then needed. Checkpointing the flow frees exactly that, and leaves
    the decoder's own graph intact and functional.
    """
    n = 0
    for mod in modules:
        for m in mod.modules():
            if hasattr(m, 'use_checkpoint') and m is not mod:
                m.use_checkpoint = True
                n += 1
    return n


def purge_spconv_cache(*tensors):
    """
    Drop every cached submanifold-conv neighbour map so the next call rebuilds it
    with the backward fields present.

    flex_gemm builds that cache in two variants (submanifold_conv3d.py:86-101):

        need_grad = any(ctx.needs_input_grad)          # :331
        if need_grad: ... gray_code, sorted_idx, valid_signal_i/o/seg
        else:         ... gray_code, sorted_idx        # _no_bwd, fields omitted

    and its backward unconditionally reads valid_signal_i/o/seg under the
    MASKED_IMPLICIT_GEMM* algorithms, which are the default (spconv/__init__.py:23).

    encode_shape_slat() runs the shape encoder with frozen weights on an input
    that does not require grad, so need_grad is False and the REDUCED variant is
    what gets cached. TRELLIS.2 registers it on the SparseTensor
    (conv_flex_gemm.py:44-57), and SparseTensor.replace() passes _spatial_cache
    BY REFERENCE (basic.py:675) — so ss_n, the noise, the flow output and the
    decoder input are all one dict. Every later graded decode then reuses the
    reduced cache and dies with:

        AttributeError: 'SubMConv3dNeighborCache' has no attribute 'valid_signal_i'

    Priming with a grad-enabled call does NOT fix this: the cache is already
    populated, so the grad-enabled call takes the `neighbor_cache is not None`
    branch and never rebuilds. The entry has to be deleted. Only the
    SubMConv3d_* keys are removed; 'layout', 'shape' and 'seqlen' are
    grad-agnostic and expensive, so they stay.
    """
    removed = []
    for t in tensors:
        for k in [k for k in list(t._spatial_cache.keys())
                  if k.startswith('SubMConv3d_neighbor_cache')]:
            del t._spatial_cache[k]
            removed.append(k)
    return removed


def voxel_key(c, S):
    """flatten integer voxel coords to one sortable key"""
    return (c[:, 0] * S[1] + c[:, 1]) * S[2] + c[:, 2]


def build_neighbour_graph(coords, spatial_shape):
    """
    Pairs of voxels adjacent along +x, +y, +z in the sparse grid.

    Built ONCE. The occupancy pattern comes from shape_slat, which is a
    deterministic encode of a fixed mesh, so the coordinate set never changes
    between steps — only the features on it do. An assert downstream checks
    that rather than trusting it.
    """
    c = coords[:, 1:4].long()
    S = torch.tensor(list(spatial_shape), device=c.device, dtype=torch.long)
    key = voxel_key(c, S)
    order = torch.argsort(key)
    skey = key[order]
    out = []
    for d in range(3):
        cn = c.clone()
        cn[:, d] += 1
        ok = cn[:, d] < S[d]                       # do not wrap across an axis
        nk = voxel_key(cn, S)
        j = torch.searchsorted(skey, nk).clamp(max=len(skey) - 1)
        hit = ok & (skey[j] == nk)
        i_idx = torch.nonzero(hit).squeeze(1)
        out.append(torch.stack([i_idx, order[j[i_idx]]], dim=1))
    return torch.cat(out, dim=0), skey, order, S


def pred_to_xstart(x_t, t, pred):
    """FlowEulerSampler._pred_to_xstart, flow_euler.py:38-39. Not a re-derivation."""
    return (1 - SIGMA_MIN) * x_t - (SIGMA_MIN + (1 - SIGMA_MIN) * t) * pred


def flow_eval(flow, x, t, cond, concat_cond):
    """
    One flow evaluation.

    NOT sampler._inference_model: this sampler is a FlowEulerGuidanceIntervalSampler
    whose two CFG mixins override that method with extra REQUIRED positional args
    (guidance_strength, guidance_interval) and would raise TypeError.

    Calling the flow directly is not an approximation. The shipped texturing
    config uses guidance_strength=1.0, and classifier_free_guidance_mixin.py:10-11
    short-circuits that to a single cond-only forward pass. This IS inference.
    The t -> 1000*t scaling mirrors flow_euler.py:44-46.
    """
    t_ten = torch.tensor([1000.0 * t] * x.shape[0], device=DEVICE, dtype=torch.float32)
    return flow(x, t_ten, cond, concat_cond=concat_cond)


def run_ode(flow, x, cond, ss_n, pairs):
    """Euler prefix under no_grad, LoRA ACTIVE (inference has it active too)."""
    with torch.no_grad():
        for t, t_prev in pairs:
            v = flow_eval(flow, x, t, cond, ss_n)
            x = x - (t - t_prev) * v
    return x


def main():
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules import sparse as sp
    from trellis2.modules.sparse.conv import config as conv_config

    # ── THE SPCONV ALGORITHM MUST BE BACKWARD-CAPABLE ────────────────────────
    # TRELLIS.2 ships 'masked_implicit_gemm_splitk' (conv/config.py:2). Its
    # backward unconditionally reads valid_signal_i/o/seg from the neighbour
    # cache (submanifold_conv3d.py:292-310), but the cache only carries those
    # when the FIRST call on those coords had a grad-requiring input
    # (:331 need_grad = any(ctx.needs_input_grad), :86-101 two variants).
    # encode_shape_slat runs frozen on a non-grad input, so the reduced variant
    # is what gets cached — and SparseTensor.replace() shares _spatial_cache by
    # reference (basic.py:675), so it propagates to every later graded decode:
    #     AttributeError: 'SubMConv3dNeighborCache' has no attribute 'valid_signal_i'
    # 'implicit_gemm' takes a different branch whose backward needs only
    # neighbor_map, which every variant of the cache always carries. TRELLIS.2
    # re-reads this on every conv forward (conv_flex_gemm.py:38), so setting it
    # here is enough — no monkey-patching of their source.
    # Inference-only code never hit this because run() is @torch.no_grad().
    conv_config.FLEX_GEMM_ALGO = args.spconv_algo
    print(f'[SPCONV] algorithm = {conv_config.FLEX_GEMM_ALGO}  '
          f'(shipped default is masked_implicit_gemm_splitk, whose backward '
          f'needs cache fields a no_grad first call never stores)', flush=True)

    print('=' * 92)
    # The banner has to say which modules are actually adapted. Inherited text
    # read "CROSS-ATTENTION" unconditionally, which on the self-attention arm log would be a
    # flat lie in the first line someone reads.
    print(f'ARM {_ARM} — LoRA on TRELLIS.2 texture-flow '
          + ('CROSS-ATTENTION + SELF-ATTENTION' if _HAS_SA
             else 'CROSS-ATTENTION'))
    print(f'  run       : {LABEL}')
    print(f'  targets   : {args.targets}   rank {args.rank}')
    print(f'  mesh      : {args.mesh}   (FIXED — an input, never modified)')
    print(f'  frames    : {N_FRAMES}   train {len(TRAIN)}   held-out {len(HELD_OUT)}')
    print(f'  epochs    : {EPOCHS}   lr {args.lr}   loss_scale {args.loss_scale}')
    print(f'  out       : {OUT}')
    print('=' * 92, flush=True)
    json.dump(_CFG | {'run_id': RUN_ID, 'label': LABEL},
              open(OUT / 'config.json', 'w'), indent=2)

    pipe = Trellis2TexturingPipeline.from_pretrained(
        'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')

    # ── GATE-device ──────────────────────────────────────────────────────────
    # Trellis2TexturingPipeline.to() (trellis2_texturing.py:97-103) OVERRIDES the
    # base class and is a NO-OP when low_vram=True, which is the default because
    # texturing_pipeline.json does not set it. pipe.cuda() then sets a string and
    # moves nothing; models are walked to the GPU inside run() and walked back
    # after. We call the flow and decoder DIRECTLY, so that lazy scheme would
    # hand us CPU weights against CUDA activations — which is exactly how job
    # 2153281 died at input_layer. Turning low_vram off makes .cuda() mean what
    # it says, and the gate below proves it rather than trusting it.
    pipe.low_vram = False
    pipe.cuda()
    flow = pipe.models[f'tex_slat_flow_model_{args.resolution}']
    decoder = pipe.models['tex_slat_decoder']
    flow.to(DEVICE); decoder.to(DEVICE)
    for m in pipe.models.values():
        if isinstance(m, nn.Module):
            for q in m.parameters():
                q.requires_grad_(False)

    # ── The decoder's weights must REQUIRE grad, without being TRAINED ───────
    # This is a workaround for a flex_gemm bug, not a modelling choice. Its
    # backward kernel returns grad_weight=None when weight.requires_grad is False
    # (kernels/triton/.../bwd_implicit_gemm.py:166-169), and the caller then does
    # `grad_weight.reshape(...)` with no None check (submanifold_conv3d.py:270):
    #     AttributeError: 'NoneType' object has no attribute 'reshape'
    # The only other branch that guards this, explicit_gemm, is unusable here —
    # it OOMs at res 1024 and raises "flip_cuda not implemented for 'UInt32'" at
    # res 512, measured both ways.
    #
    # So we let the weights require grad purely to keep the kernel from returning
    # None, and throw the result away. They are NOT in the optimizer — it is
    # constructed over reg.parameters() alone — so nothing updates them, and
    # their .grad is zeroed each step so it cannot accumulate. The model stays
    # frozen in every sense that matters; only an autograd flag changed.
    if args.decoder_grad_weights:
        n_dec = sum(1 for _ in decoder.parameters())
        for q in decoder.parameters():
            q.requires_grad_(True)
        print(f'[SPCONV] decoder: {n_dec} weight tensors set requires_grad=True to '
              f'dodge the grad_weight=None crash. NOT optimised, grads discarded.',
              flush=True)
    _devs = {n: str(next(m.parameters()).device)
             for n, m in pipe.models.items() if isinstance(m, nn.Module)}
    print(f'\n[GATE-device] {_devs}')
    assert all('cuda' in d for d in _devs.values()), \
        f'GATE-device FAILED: a model is still on CPU: {_devs}'
    print('[GATE-device] PASSED', flush=True)

    if not args.no_grad_ckpt:
        n_ck = enable_grad_checkpointing(flow)
        print(f'[MEMORY] gradient checkpointing enabled on {n_ck} FLOW blocks. '
              f'The decoder is deliberately left uncheckpointed — flex_gemm\'s '
              f'spconv backward needs the neighbour cache its forward built, and '
              f'recomputation loses it.', flush=True)
    else:
        print('[MEMORY] gradient checkpointing DISABLED (--no-grad-ckpt)', flush=True)

    ch, ctx, nb = flow.model_channels, flow.cond_channels, len(flow.blocks)
    mlp_hidden = int(ch * flow.mlp_ratio)
    print(f'\n[FLOW] blocks={nb}  model_channels={ch}  cond_channels={ctx}')
    print(f'       cross_attn.to_q  Linear({ch}, {ch})')
    print(f'       cross_attn.to_kv Linear({ctx}, {2*ch})   <- image -> 3D k/v')
    print(f'       cross_attn.to_out Linear({ch}, {ch})')
    print(f'       self_attn.to_qkv  Linear({ch}, {3*ch})   <- self-attn '
          f'{"ADAPTED" if "sa_qkv" in _TSET else "frozen"}')
    print(f'       self_attn.to_out  Linear({ch}, {ch})   <- self-attn '
          f'{"ADAPTED" if "sa_out" in _TSET else "frozen"}', flush=True)
    # GATE-shape. The hook adds its delta to the RAW output of these Linears,
    # before any reshape/unbind, so the delta width must equal the Linear's
    # out_features exactly. A mismatch would broadcast instead of raising —
    # silently adding the wrong thing to the wrong channels.
    if _HAS_SA:
        _sa0 = flow.blocks[ACTIVE[0]].self_attn
        assert _sa0.to_qkv.out_features == 3 * ch, (
            f'GATE-shape FAILED: self_attn.to_qkv emits '
            f'{_sa0.to_qkv.out_features}, expected {3*ch}. The fused qkv layout '
            f'changed and sa_qkv would write the wrong width.')
        assert _sa0.to_out.in_features == ch and _sa0.to_out.out_features == ch, (
            f'GATE-shape FAILED: self_attn.to_out is '
            f'{_sa0.to_out.in_features}->{_sa0.to_out.out_features}, expected '
            f'{ch}->{ch}.')
        print(f'[GATE-shape] PASSED — self_attn.to_qkv {ch}->{3*ch}, '
              f'to_out {ch}->{ch}', flush=True)

    torch.manual_seed(args.seed)
    assert nb == _NB, f'expected {_NB} blocks, found {nb}'
    reg = LoRARegistry(nb, ch, ctx, args.rank,
                       with_mlp=(args.targets == 'cross+mlp'),
                       mlp_hidden=mlp_hidden,
                       targets=_TSET, active=ACTIVE,
                       alpha=args.lora_alpha).to(DEVICE)
    print(f'[LORA-SCALE] rank {args.rank}  alpha {args.lora_alpha}  '
          f'-> scaling {args.lora_alpha / args.rank:.4f}   '
          f'(1.0 reproduces every pre-sweep run exactly)')
    print(f'[PLACEMENT] blocks={args.blocks} -> {ACTIVE[0]}..{ACTIVE[-1]} '
          f'({len(ACTIVE)} of {nb});  targets={_TSET}')
    print(f'            an edit here has {nb - 1 - ACTIVE[-1]} self-attention '
          f'blocks downstream to propagate it', flush=True)
    n_par = sum(p.numel() for p in reg.parameters())
    n_mat = sum(1 for b in reg.blocks.values() for _ in b.named_children())
    print(f'\n[LORA] {n_par:,} params   {n_mat} B matrices   '
          f'(rank {args.rank} x {nb} blocks)')
    assert all(float(l.B.abs().max()) == 0
               for b in reg.blocks.values() for _, l in b.named_children()), \
        'B must be zero-init'
    print('[GATE 0] param count + B=0 identity  PASSED', flush=True)

    # ── mesh: two frames, one vertex ordering ────────────────────────────────
    mesh_in = trimesh.load(args.mesh, process=False, force='mesh')
    print(f'\n[MESH] {len(mesh_in.vertices):,} verts  {len(mesh_in.faces):,} faces  FIXED')
    mesh_pp = pipe.preprocess_mesh(mesh_in)
    assert len(mesh_pp.vertices) == len(mesh_in.vertices) and \
           np.array_equal(np.asarray(mesh_pp.faces), np.asarray(mesh_in.faces)), \
        'preprocess_mesh changed topology — the two-frame correspondence is broken'

    v_raw = torch.from_numpy(np.asarray(mesh_in.vertices)).float().to(DEVICE)
    v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().to(DEVICE)
    faces = torch.from_numpy(np.asarray(mesh_in.faces)).int().to(DEVICE)

    # Registration still matters here — the supervision is a per-pixel render
    # loss against the video, so a misregistered mesh still makes the adapter
    # paint the wrong part of the object. Moving the adapter upstream does not
    # fix registration; it is an independent correction.
    #
    # But the meshes under render/out/ are ALREADY registered: they came from
    # the export step, which applies the transform to the vertices before writing
    # the .ply.
    # Applying it again here double-transforms and OVERSHOOTS — measured
    # silhouette IoU 0.7276, i.e. worse than the decoder-side arm's *unaligned* 0.7611 and far
    # off its aligned 0.9054. Hence off by default; GATE-cam below is set to
    # catch exactly this mistake rather than let it pass as a texture problem.
    if args.align_mesh:
        al = json.load(open(args.alignment))
        R = torch.tensor(rodrigues(al['rotvec']), dtype=torch.float32, device=DEVICE)
        C = torch.tensor(al['centre'], dtype=torch.float32, device=DEVICE)
        T = torch.tensor(al['translation'], dtype=torch.float32, device=DEVICE)
        v_raw = float(al['scale']) * ((v_raw - C) @ R.T) + C + T
        print(f'[ALIGN] registration transform applied  s={al["scale"]:.4f}  '
              f'|rotvec|={np.linalg.norm(al["rotvec"]):.4f} rad', flush=True)
    else:
        print('[ALIGN] not re-applied — the input mesh is already registered '
              '(alignment already baked into the mesh)', flush=True)

    vn_raw = torch.from_numpy(np.asarray(mesh_in.vertex_normals).copy()
                              ).float().to(DEVICE)
    renderer = SurfaceRenderer(v_raw, v_pp, vn_raw, faces, args.render_res,
                               args.resolution)
    print(f'[RENDER] {args.render_res}px, silhouette {renderer.n_px:,} px '
          f'({100*renderer.n_px/args.render_res**2:.1f}% of image), '
          f'rasterised ONCE — geometry and camera are both constant', flush=True)

    # ── GATE-align ───────────────────────────────────────────────────────────
    # The targets were rasterised by a DIFFERENT script. If its silhouette is not
    # this one pixel-for-pixel, every per-pixel loss is partly a shape comparison
    # and no number below means anything. Not hypothetical: the orbit renderer
    # disagrees with the target rasteriser (bbox y 180-339 vs y 184-356), and
    # comparing across that gap manufactured 2,089 phantom "white holes" in an
    # earlier analysis. Count equality is NOT enough — two different silhouettes
    # can have equal area — so the masks themselves are compared.
    _mp = Path(args.gt_render_dir).parent / 'render_mask.npy'
    assert _mp.exists(), (
        f'GATE-align FAILED: {_mp} is missing. make_targets.py writes the '
        f'silhouette it built the targets on next to them; without it there is '
        f'nothing to check this render against. Regenerate the targets.')
    tgt_mask = torch.from_numpy(np.load(_mp)).to(DEVICE)
    assert tuple(tgt_mask.shape) == tuple(renderer.mask.shape), (
        f'GATE-align FAILED: targets are {tuple(tgt_mask.shape)} but our render is '
        f'{tuple(renderer.mask.shape)}. Pass --render-res {tgt_mask.shape[0]}.')
    _dis = int((tgt_mask ^ renderer.mask).sum())
    print(f'\n[GATE-align] target silhouette {int(tgt_mask.sum()):,} px vs our render '
          f'{renderer.n_px:,} px — disagreeing pixels: {_dis}')
    assert _dis == 0, (
        f'GATE-align FAILED: {_dis} pixels differ between the target rasterisation '
        f'({_mp}) and this training render. Same mesh, same camera and the same '
        f'--render-res are all required; a mismatch silently turns the texture '
        f'loss into a shape loss.')
    print('[GATE-align] PASSED — identical pixel for pixel', flush=True)

    # ── GATE-cam ─────────────────────────────────────────────────────────────
    # A wrong camera would look exactly like a texture failure. the decoder-side arm's lesson
    # was that three separate bugs were in the DIAGNOSTICS, not the method, so
    # the view is checked against the GT silhouette before anything trains.
    # ── GATE-cam, on COVERAGE rather than symmetric IoU ──────────────────────
    # The question this gate exists to ask is "is the mesh in the right place",
    # and symmetric IoU answers a different one. A cast shadow, or the video model
    # inflating the object as the clip runs, only ever ADDS to the video mask --
    # that craters IoU while our mesh sits exactly where it should. Measured on
    # these six with correct, ground-truth rotations: IoU 0.575-0.887, which the
    # old `iou > 0.80` would have failed on four of them, every one of which has
    # under 5% of its silhouette actually outside the video object.
    #
    # COVERAGE is the one-sided question -- what fraction of OUR silhouette falls
    # outside the video's object -- and a shadow cannot affect it. Probed across
    # the whole clip rather than at one frame, and gated on the MEDIAN, so a few
    # deformed frames cannot fail an otherwise sound object while a genuinely
    # misplaced mesh (which sits at 40%+ on every frame) still fails.
    _probe = list(range(1, N_FRAMES + 1, max(1, N_FRAMES // 9)))[:9]
    _unc = []
    for _f in _probe:
        _g = load_gt(_f, args.gt_dir, args.render_res)[0]
        _b = torch.cat([_g[0], _g[-1], _g[:, 0], _g[:, -1]], dim=0)
        _cut = float((_b.min(dim=1).values.median() - 6.0 / 255.0).clamp(0.0, 1.0))
        _vm = (_g.min(dim=2).values < _cut)
        _unc.append(100.0 * float((renderer.mask & ~_vm).sum()) / max(renderer.n_px, 1))
    _med = float(np.median(_unc))
    _iou0 = float((renderer.mask & (load_gt(_probe[len(_probe)//2], args.gt_dir,
                  args.render_res)[0].min(dim=2).values < 0.95)).sum())
    print(f'\n[GATE-cam] % of OUR silhouette outside the video object, across the clip:')
    print('           ' + '  '.join(f'f{f}:{u:.1f}%' for f, u in zip(_probe, _unc)))
    print(f'           median {_med:.2f}%   (gate: < 25%)')
    assert _med < 25.0, (
        f'GATE-cam FAILED: {_med:.2f}% of our silhouette falls outside the video object, '
        f'median over {len(_probe)} frames. The mesh is not where the video\'s object is. '
        f'Check the rotation in out/orient/orientation_guan.json and the mesh passed as --mesh.')
    if _med > 10.0:
        print(f'           *** WARNING: {_med:.2f}% is high. The targets are still built (the rim is')
        print(f'           filled from the nearest valid neighbour) but that fraction of every target')
        print(f'           is INVENTED colour, not copied video. Treat this object\'s numbers with care.')
    print('[GATE-cam] PASSED', flush=True)

    # ── conditioning + shape latent, both computed ONCE ──────────────────────
    shape_slat = pipe.encode_shape_slat(mesh_pp, args.resolution)
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(DEVICE)
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(DEVICE)
    ss_n = (shape_slat - ss_mean) / ss_std
    tex_std = torch.tensor(pipe.tex_slat_normalization['std'])[None].to(DEVICE)
    tex_mean = torch.tensor(pipe.tex_slat_normalization['mean'])[None].to(DEVICE)
    print(f'\n[SHAPE] shape_slat {tuple(shape_slat.feats.shape)} — '
          f'deterministic, computed ONCE, reused for every frame', flush=True)

    frames = TRAIN + HELD_OUT
    t0 = time.time()

    # ── Conditioning must be PREPROCESSED, exactly as the pipeline does it ───
    # get_cond() feeds DINOv3 directly; it does NOT crop or matte. run() normally
    # calls preprocess_image() first (trellis2_texturing.py:398), which removes the
    # background, crops square to the object's bbox, and premultiplies by alpha so
    # the background is BLACK. Handing it the raw 960x960 frame instead — a small
    # teapot in a large white field — is far out of distribution and yields a
    # black texture, even though the PBR field is bright when conditioned properly
    # (measured: base_color p99 0.55-0.79, diag_black_texture.py).
    #
    # We reproduce that preprocessing deterministically rather than calling it:
    # preprocess_image() runs BiRefNet per frame and crops to that frame's own
    # bbox, so both the matte and the framing would jitter frame to frame, and
    # that jitter would be indistinguishable from texture change. Alpha comes from
    # the near-white background by a fixed threshold, and the crop box is the
    # UNION over all frames — identical framing for every frame, contributing
    # exactly zero to any temporal measurement.
    raws = [np.array(Image.open(Path(args.gt_dir) / f'frame_{fi:04d}.png').convert('RGB'))
            for fi in frames]
    alphas = [largest_component(r.min(axis=2) < 245) for r in raws]
    union = np.zeros_like(alphas[0])
    for a in alphas:
        union |= a
    ys, xs = np.where(union)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    bbox = (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))

    conds, gts, gt_masks = {}, {}, {}
    for fi, r, a in zip(frames, raws, alphas):
        rgba = np.concatenate([r, (a * 255).astype(np.uint8)[..., None]], axis=-1)
        f = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
        cond_img = Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255).astype(np.uint8))
        conds[fi] = pipe.get_cond([cond_img], args.resolution)['cond'].detach()
        # SUPERVISION is the rendered backprojected GT, not the raw video frame.
        # Both are renders of the same mesh through the same camera, so the
        # silhouette matches exactly and the loss is purely texture.
        _gp = Path(args.gt_render_dir) / f'gt_{fi:04d}.png'
        _im = Image.open(_gp).convert('RGB')
        gts[fi] = (torch.from_numpy(np.array(_im)).float().div(255.0)
                   .to(DEVICE).unsqueeze(0))
        # This mask says WHICH PIXELS OF THE TARGET ARE REAL. make_targets.py
        # fills 100% of our silhouette, so it is all-true there; with the older
        # vertex-round-trip targets it excluded the unsupervised fringe.
        # An earlier version of this file loaded it and then OVERWROTE it with
        # the raw video's silhouette, which knows nothing about which target
        # pixels were fabricated. That let 447 pure-white px per frame into the
        # loss for all 30 epochs, and the adapter learned to paint them white.
        # Do not add a second assignment here.
        gt_masks[fi] = torch.from_numpy(
            np.load(Path(args.gt_render_dir) / f'valid_{fi:04d}.npy')).to(DEVICE)
    print(f'[COND] preprocessed like the pipeline: alpha by threshold (not BiRefNet), '
          f'fixed union crop {bbox}, premultiplied to black', flush=True)
    print(f'[COND] DINOv3 tokens for {len(frames)} frames in {time.time()-t0:.1f}s  '
          f'shape {tuple(conds[frames[0]].shape)} — cached, frozen', flush=True)

    # MCFM: swap the vanilla per-frame tokens for temporally blended ones. Every
    # downstream use of `conds` (run_ode, flow_eval, the gates, evaluate) reads
    # this dict, so blending here covers all of them. No-op when --mcfm is unset.
    if args.mcfm:
        sys.path.insert(0, str(_HERE))
        from modules.mcfm import blend_conds
        conds = blend_conds(conds, args.mcfm)

    # Noise is FIXED across frames, matching the baseline's fixed seed. shape_slat
    # is a deterministic encode, so with the noise pinned too the ONLY thing that
    # differs between frames is the image conditioning. Any change the adapter
    # learns is therefore attributable to the cross-attention pathway.
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    noise_feats = torch.randn(ss_n.coords.shape[0],
                              flow.in_channels - ss_n.feats.shape[1],
                              generator=g).to(DEVICE)
    noise = ss_n.replace(feats=noise_feats)

    # ── GATE-grad ────────────────────────────────────────────────────────────
    print('\n[GATE-grad] probing gradient health through decoder + flow ...', flush=True)
    reps = {}
    for t in (1.0, 0.75, 0.5, 0.25):
        reg.zero_grad(set_to_none=False)
        with lora_ctx(flow, reg):
            v = flow_eval(flow, noise, t, conds[frames[0]], ss_n)
            (v.feats.float() ** 2).mean().mul(args.loss_scale).backward()
        reps[t] = grad_report(reg, tag=f'probe @ t={t}')
    json.dump({str(k): v for k, v in reps.items()},
              open(OUT / 'logs' / 'gate_grad.json', 'w'), indent=2)
    assert reps[1.0]['n_zero'] == 0, (
        f'GATE-grad FAILED: {reps[1.0]["n_zero"]}/{reps[1.0]["n_matrices"]} B '
        f'matrices got zero gradient. Dead: {reps[1.0]["dead"]}')
    reg.zero_grad(set_to_none=True)
    print('[GATE-grad] PASSED — every B matrix receives gradient', flush=True)

    # ── GATE-plain: at B=0 our forward must equal the untouched model ────────
    with torch.no_grad():
        x_plain = run_ode(flow, noise, conds[frames[0]], ss_n, T_PAIRS)
        with lora_ctx(flow, reg):
            x_lora = run_ode(flow, noise, conds[frames[0]], ss_n, T_PAIRS)
        d = float((x_plain.feats - x_lora.feats).abs().max())
    print(f'\n[GATE-plain] max |frozen - lora(B=0)| = {d:.3e}  (tol 1e-4)')
    assert d < 1e-4, f'GATE-plain FAILED: adapter is not the identity at init ({d:.3e})'
    print('[GATE-plain] PASSED', flush=True)

    # ── GATE-selfattn (the self-attention arm only) ──────────────────────────────────────────
    # THE gate for this arm. Everything else in this file is inherited and
    # already proven; what is new is that a hook fires on self_attn rather than
    # cross_attn. Three ways that can be wrong while every other gate still
    # passes, so all three are checked explicitly:
    #
    #   1. hooked the wrong parent. cross_attn and self_attn BOTH own a
    #      submodule called to_out. Hooking cross_attn.to_out by mistake
    #      reproduces the kvo arm exactly and nothing anywhere would complain.
    #   2. hook never fires. A typo'd attribute name means the handle list is
    #      short and the adapter is simply inert — indistinguishable from
    #      "self-attention did not help", which is the very conclusion this run
    #      exists to test. That failure would be reported as a result.
    #   3. delta lands but does nothing measurable.
    #
    # B is zero-init, so at this point every arm would trivially pass any
    # comparison. The adapter has to be made non-trivial FIRST, then the module
    # outputs compared — the same construction as GATE-keylock.
    if _HAS_SA:
        _saved = {id(l): l.B.detach().clone()
                  for b in reg.blocks.values() for _, l in b.named_children()}
        _blk0 = flow.blocks[ACTIVE[0]]
        # 1 — count the handles that actually attach to self_attn submodules.
        _xa_mods = {id(m) for m in _blk0.cross_attn.modules()}
        _n_sa = _n_xa = 0
        with lora_ctx(flow, reg):
            for _m in (_blk0.self_attn.to_qkv, _blk0.self_attn.to_out):
                _n_sa += len(_m._forward_hooks)
            for _m in (_blk0.cross_attn.to_q, _blk0.cross_attn.to_kv,
                       _blk0.cross_attn.to_out):
                _n_xa += len(_m._forward_hooks)
        _want_sa = sum(1 for t in ('sa_qkv', 'sa_out') if t in _TSET)
        print(f'\n[GATE-selfattn] hooks on block {ACTIVE[0]}: '
              f'self_attn={_n_sa} (want {_want_sa})  cross_attn={_n_xa}')
        assert _n_sa == _want_sa, (
            f'GATE-selfattn FAILED: {_n_sa} hooks landed on self_attn, expected '
            f'{_want_sa}. The adapter is inert or attached to the wrong module, '
            f'and this run would report "self-attention does not help" while '
            f'never having adapted self-attention.')
        assert id(_blk0.self_attn.to_out) not in _xa_mods, \
            'GATE-selfattn FAILED: self_attn.to_out and cross_attn.to_out are the ' \
            'same object — the two adapters would be tied together.'
        # 2 — make B non-trivial and prove BOTH modules actually move.
        with torch.no_grad():
            for b in reg.blocks.values():
                for _, l in b.named_children():
                    l.B.normal_(0.0, 0.1)
            _h = torch.randn(64, ch, device=DEVICE,
                             dtype=_blk0.self_attn.to_qkv.weight.dtype)
            _qkv_p = _blk0.self_attn.to_qkv(_h)
            _out_p = _blk0.self_attn.to_out(_h)
            _xkv_p = _blk0.cross_attn.to_kv(
                conds[frames[0]].to(_blk0.cross_attn.to_kv.weight.dtype))
            with lora_ctx(flow, reg):
                _qkv_a = _blk0.self_attn.to_qkv(_h)
                _out_a = _blk0.self_attn.to_out(_h)
                _xkv_a = _blk0.cross_attn.to_kv(
                    conds[frames[0]].to(_blk0.cross_attn.to_kv.weight.dtype))
            _d_qkv = float((_qkv_a - _qkv_p).abs().max())
            _d_out = float((_out_a - _out_p).abs().max())
            _d_xkv = float((_xkv_a - _xkv_p).abs().max())
            for b in reg.blocks.values():
                for _, l in b.named_children():
                    l.B.copy_(_saved[id(l)])
        print(f'  with B ~ N(0,0.1):  max|d self_attn.to_qkv| = {_d_qkv:.3e}   '
              f'max|d self_attn.to_out| = {_d_out:.3e}')
        print(f'                      max|d cross_attn.to_kv| = {_d_xkv:.3e}')
        if 'sa_qkv' in _TSET:
            assert _d_qkv > 0.0, (
                'GATE-selfattn FAILED: self_attn.to_qkv did not move. The sa_qkv '
                'adapter is inert.')
        if 'sa_out' in _TSET:
            assert _d_out > 0.0, (
                'GATE-selfattn FAILED: self_attn.to_out did not move. The sa_out '
                'adapter is inert.')
        if 'to_kv' in _TSET:
            assert _d_xkv > 0.0, (
                'GATE-selfattn FAILED: cross_attn.to_kv did not move, so this is '
                'not the "self-attn IN ADDITION TO cross-attn" arm it claims to be.')
        else:
            assert _d_xkv == 0.0, (
                f'GATE-selfattn FAILED: cross_attn.to_kv moved by {_d_xkv:.3e} on '
                f'a self-attention-only arm. --targets sa must leave cross '
                f'attention exactly frozen or it is not a clean control.')
        print('[GATE-selfattn] PASSED — self-attention is genuinely adapted, '
              'and the cross-attention state matches the declared target set',
              flush=True)

    # ── GATE-keylock (the qkvo arm only) ───────────────────────────────────────────
    # The entire claim of the v-only arm is that K is untouched. B is zero-init,
    # so at this point EVERY arm would trivially pass a check — the adapter has
    # to be made non-trivial first, then the two halves compared separately.
    # K must move by EXACTLY zero (not 'a small amount'), V must actually move.
    if 'to_v' in _TSET:
        _saved = {id(l): l.B.detach().clone()
                  for b in reg.blocks.values() for _, l in b.named_children()}
        with torch.no_grad():
            for b in reg.blocks.values():
                for _, l in b.named_children():
                    l.B.normal_(0.0, 0.1)
            _kvmod = flow.blocks[ACTIVE[0]].cross_attn.to_kv
            # conds are cached in fp32; the flow casts to the block dtype before
            # cross-attention, so calling to_kv directly has to do the same or it
            # dies with 'mat1 and mat2 must have the same dtype'.
            _cin = conds[frames[0]].to(_kvmod.weight.dtype)
            _plain = _kvmod(_cin)
            with lora_ctx(flow, reg):
                _adapt = _kvmod(_cin)
            # Split with the MODEL's own reshape+unbind (modules.py:92-95), not
            # by slicing the way the hook does — otherwise this only proves the
            # zeros landed where I put them, not that that location is K.
            _nh = flow.blocks[ACTIVE[0]].cross_attn.num_heads
            _B, _L = _plain.shape[0], _plain.shape[1]
            _kp, _vp = _plain.reshape(_B, _L, 2, _nh, -1).unbind(dim=2)
            _ka, _va = _adapt.reshape(_B, _L, 2, _nh, -1).unbind(dim=2)
            _dK = float((_ka - _kp).abs().max())
            _dV = float((_va - _vp).abs().max())
            for b in reg.blocks.values():
                for _, l in b.named_children():
                    l.B.copy_(_saved[id(l)])
        print(f'\n[GATE-keylock] with B ~ N(0,0.1):  max|dK| = {_dK:.3e}   '
              f'max|dV| = {_dV:.3e}')
        assert _dK == 0.0, (
            f'GATE-keylock FAILED: the KEY half moved by {_dK:.3e}, so K is not '
            f'locked and the qkvo arm is not testing what it claims.')
        assert _dV > 0.0, (
            f'GATE-keylock FAILED: the VALUE half did not move, so the adapter '
            f'is inert.')
        print('[GATE-keylock] PASSED — K exactly frozen, V live', flush=True)

    def inter_mask(fi):
        """
        Pixels where BOTH our render covers the object AND the video has object.

        With the target now a RENDER of the same mesh through the same camera,
        the silhouettes agree by construction. What remains is the sliver where a
        covered pixel's triangle had an unsupervised vertex, which carries no GT
        colour and must not be scored.
        """
        return renderer.mask & gt_masks[fi]

    def decode_render(x0, clamp=True, return_pbr=False):
        """x0 (normalised tex latent) -> rendered image. Differentiable."""
        slat = x0 * tex_std + tex_mean
        pbr = decoder(slat) * 0.5 + 0.5
        img = renderer.sample(pbr, clamp=clamp)
        return (img, pbr) if return_pbr else img

    @torch.no_grad()
    def full_inference(fi):
        """The real thing: all 12 ODE steps with the adapter, then render."""
        with lora_ctx(flow, reg):
            x = run_ode(flow, noise, conds[fi], ss_n, T_PAIRS)
        return decode_render(x)

    @torch.no_grad()
    def evaluate(frame_list):
        rows = []
        for fi in frame_list:
            img = full_inference(fi)
            p, s = psnr_ssim(img, gts[fi], inter_mask(fi))
            rows.append(dict(frame=fi, psnr=p, ssim=s))
        return dict(psnr_mean=float(np.mean([r['psnr'] for r in rows])),
                    psnr_std=float(np.std([r['psnr'] for r in rows])),
                    ssim_mean=float(np.mean([r['ssim'] for r in rows])),
                    per_frame=rows)

    # ── GATE-bwd: prime the spconv neighbour cache WITH GRAD, and prove the
    #             decoder backward works before anything else touches it.
    #
    # flex_gemm caches a submanifold-conv neighbour map per coordinate set, and
    # builds a REDUCED version when the first call that touches those coords runs
    # under no_grad:
    #     submanifold_conv3d.py:331   need_grad = any(ctx.needs_input_grad)
    #     submanifold_conv3d.py:86    if need_grad: ... valid_signal_i/o/seg
    #                          :98    else:        ... _no_bwd variant, omitted
    # TRELLIS.2 stores it on the SparseTensor (conv_flex_gemm.py:44-57), and
    # SparseTensor.replace() passes _spatial_cache BY REFERENCE (basic.py:675).
    # ss_n, noise and every tensor derived from them therefore share ONE cache
    # dict for the whole process. So a single no_grad decode early on poisons
    # every graded decode afterwards with
    #     AttributeError: 'SubMConv3dNeighborCache' has no attribute 'valid_signal_i'
    # which is precisely how jobs 2154525/2154566/2154567 died — and why it was
    # NOT a gradient-checkpointing problem, though it looked like one.
    #
    # The rule this enforces: the first decoder call in the process must be
    # grad-enabled. Everything after it reuses a cache that is a strict superset.
    print('\n[GATE-bwd] purging + repriming spconv cache, testing decoder backward ...',
          flush=True)
    _purged = purge_spconv_cache(ss_n, shape_slat)
    print(f'  purged {len(_purged)} no-bwd neighbour cache(s) left by '
          f'encode_shape_slat: {_purged}', flush=True)
    x0_warm = ss_n.replace(feats=torch.randn(
        ss_n.coords.shape[0], 32, device=DEVICE).requires_grad_(True))
    img_warm = decode_render(x0_warm)
    # LOSS_SCALE is mandatory here, not decorative. The decoder is fp16
    # (tex_dec_next_dc_f16c32_fp16) and an unscaled mean over ~10^6 elements gives
    # gradients near 1e-7, below fp16's ~6e-8 subnormal floor, so they flush to
    # EXACTLY zero and this gate reports a dead path that is actually fine.
    # Measured directly (res 512, implicit_gemm_splitk):
    #     loss_scale 1     -> sum|grad| 0.0000e+00
    #     loss_scale 4096  -> sum|grad| 4.6620e+03
    #     loss_scale 65536 -> sum|grad| 7.7155e+04
    # The 65536/4096 ratio is 16.55 against an expected 16.0, so ~3% of gradient
    # mass is still underflowing at 4096 — adequate, not generous.
    (img_warm.square().mean() * args.loss_scale).backward()
    gsum = float(x0_warm.feats.grad.abs().sum()) if x0_warm.feats.grad is not None else 0.0
    print(f'  d(render)/d(tex_latent) sum|grad| = {gsum:.4e}   '
          f'peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB')
    assert gsum > 0, (
        'GATE-bwd FAILED: no gradient reaches the texture latent through '
        'decoder -> grid_sample_3d -> render. Nothing downstream can train.')
    print('[GATE-bwd] PASSED', flush=True)
    if args.decoder_grad_weights:
        for q in decoder.parameters():
            q.grad = None
    del x0_warm, img_warm
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # ── GATE-sample: is the field bright, and does our sampling reproduce it? ─
    # decode_render() samples the PBR voxel field at each pixel's 3D surface point
    # with grid_sample_3d, which returns EXACTLY 0 for points outside the occupied
    # sparse voxels. So a black render has two very different possible causes:
    # a genuinely dark field (generation), or correct field + wrong coordinates
    # (sampling). Measured independently here so the two can never be confused
    # again — diag_black_texture.py established the field reaches p99 0.55-0.79
    # when conditioning is right, so a dark SAMPLE against a bright FIELD is a
    # coordinate bug and nothing else.
    with torch.no_grad():
        with lora_ctx(flow, reg):
            _x = run_ode(flow, noise, conds[frames[0]], ss_n, T_PAIRS)
        _pbr = decoder(_x * tex_std + tex_mean) * 0.5 + 0.5
        _f = _pbr.feats.float()[:, :3]
        _im = renderer.sample(_pbr)
        _fg = _im[0].reshape(-1, 3)[renderer.mask_idx].float()
    _fq = lambda t, q: float(t.flatten().quantile(q))
    print(f'\n[GATE-sample] spatial_shape={tuple(_pbr.spatial_shape)}  '
          f'voxels={tuple(_pbr.feats.shape)}')
    print(f'  PBR FIELD  base_color mean={float(_f.mean()):.4f}  '
          f'p99={_fq(_f, 0.99):.4f}  max={float(_f.max()):.4f}')
    print(f'  SAMPLED    at surface  mean={float(_fg.mean()):.4f}  '
          f'p99={_fq(_fg, 0.99):.4f}  max={float(_fg.max()):.4f}  '
          f'frac>0.05={float((_fg.max(dim=1).values > 0.05).float().mean()):.3f}')
    # THRESHOLD 0.05, was 0.2. The 0.2 was calibrated on UNBLENDED runs, where the
    # ratio sampled/field sits at 0.41-0.66 (12 runs measured). --mcfm v3 pools over
    # time AND space, which lowers the pinned first frame's surface samples on the
    # dark teapot specifically: v3_C_teap 0.228, v3_D_teap 0.198. The latter tripped
    # the gate by 0.002 and lost the run, while v3_C_teap — same object, same v3
    # pooling, p99 0.024 vs 0.022 and max 0.328 vs 0.340, i.e. the SAME distribution
    # — passed and trained to epoch 20 with healthy monitor PSNR. So 0.198 is not a
    # coordinate bug; the constant was simply drawn through the middle of the v3
    # teapot cluster.
    #
    # What this gate exists to catch is grid_sample_3d missing the occupied voxels
    # entirely, which returns EXACTLY 0 and a ratio of ~0.00 — two orders of
    # magnitude below anything observed here. 0.05 keeps a 50x margin over that
    # failure while clearing the lowest legitimate observation by 4x. Widening it
    # cannot mask the bug it was written for.
    _ratio = float(_fg.mean()) / max(float(_f.mean()), 1e-12)
    print(f'  sampled/field ratio {_ratio:.3f}   (gate needs > 0.05; a coordinate '
          f'bug reads ~0.00, legitimate runs measured 0.198-0.660)')
    assert _ratio > 0.05, (
        f'GATE-sample FAILED: the field has mean {float(_f.mean()):.4f} but our '
        f'surface samples average {float(_fg.mean()):.4f} (ratio {_ratio:.4f}). '
        f'grid_sample_3d is returning ~0, i.e. we are sampling where the field '
        f'does not exist — a coordinate/frame bug in SurfaceRenderer, not a '
        f'texture problem.')
    print('[GATE-sample] PASSED', flush=True)

    # ── neighbour consistency (the confidence-weighted arm) ───────────────────────────────────────
    # The rim is SEEN but barely MEASURED: at grazing incidence one pixel stands
    # in for a long stretch of surface, so per-voxel evidence collapses while the
    # loss still grants every pixel equal authority. A shared LoRA direction then
    # runs away exactly there. Rather than fall back to the frozen texture (which
    # is globally wrong — mean 0.062 against a GT near 0.31), each voxel is pulled
    # toward its 3D neighbours, weighted by how unreliably it is observed. The
    # well-measured surface next to the rim is CORRECT, so that is the right thing
    # for the rim to inherit from.
    nb_pairs = nb_w = None
    if args.lambda_smooth > 0:
        from scipy.spatial import cKDTree
        _coords0 = _pbr.coords.detach().clone()
        _sshape = tuple(_pbr.spatial_shape)
        nb_pairs, _skey, _order, _S = build_neighbour_graph(_coords0, _sshape)
        N_vox = _coords0.shape[0]

        # CONFIDENCE FROM THE NEAREST VISIBLE SURFACE POINT, NOT FROM PIXEL HITS.
        # The voxel grid is ~30x finer than the render, so mapping each pixel to
        # one voxel leaves 96.7% of voxels unlabelled and the weight collapses to
        # a constant 1 — global smoothing, which would erase whatever the
        # cross-attention propagated to the back. Instead every voxel is matched
        # to the nearest point the camera actually resolved, and inherits both
        # that point's incidence and its distance.
        _vc = ((_coords0[:, 1:4].float() + 0.5) / args.resolution - 0.5)
        _surf = renderer.pos.detach().cpu().numpy()
        _sinc = renderer.cos_inc[renderer.mask].detach().cpu().numpy()
        _dist, _near = cKDTree(_surf).query(_vc.cpu().numpy(), k=1)
        conf = torch.from_numpy(_sinc[_near]).float().to(DEVICE)
        dist = torch.from_numpy(_dist).float().to(DEVICE)

        # Only voxels NEAR the resolved surface are smoothed. A rim voxel sits
        # right on visible surface but at grazing incidence -> heavy weight, and
        # its neighbours are well-observed and correct, so it inherits something
        # true. A back-side voxel is far from any resolved surface -> weight 0,
        # left entirely to cross-attention, which is the propagation path we are
        # deliberately not touching.
        _near_surf = dist < args.smooth_radius
        _unrel = ((1.0 - conf).clamp(0, 1) ** args.smooth_gamma) * _near_surf.float()
        nb_w = torch.maximum(_unrel[nb_pairs[:, 0]], _unrel[nb_pairs[:, 1]])
        nb_wsum = nb_w.sum().clamp(min=1.0)

        print(f'\n[SMOOTH] lambda={args.lambda_smooth}  gamma={args.smooth_gamma}  '
              f'radius={args.smooth_radius}')
        print(f'  voxels {N_vox:,}   adjacent pairs {nb_pairs.shape[0]:,}')
        print(f'  near the resolved surface (dist<{args.smooth_radius}): '
              f'{int(_near_surf.sum()):,} ({100*float(_near_surf.float().mean()):.1f}%)'
              f'   the rest are left to cross-attention')
        _ns = _near_surf
        print(f'  incidence of those: mean {float(conf[_ns].mean()):.3f}   '
              f'grazing(<0.3) {100*float((conf[_ns] < 0.3).float().mean()):.1f}%')
        print(f'  pair weight: mean {float(nb_w.mean()):.4f}   '
              f'nonzero pairs {int((nb_w > 0).sum()):,} '
              f'({100*float((nb_w > 0).float().mean()):.1f}%)')
        print(f'  -> head-on voxels ~0 weight, grazing rim ~1, back side exactly 0',
              flush=True)

    def smooth_term(pbr):
        """neighbour consistency on base_color, weighted by unreliability"""
        assert pbr.coords.shape[0] == nb_pairs_n, (
            f'voxel count changed ({pbr.coords.shape[0]} vs {nb_pairs_n}) — the '
            f'neighbour graph was built on a different coordinate set')
        c = pbr.feats[:, :3]
        d = c[nb_pairs[:, 0]] - c[nb_pairs[:, 1]]
        return (nb_w * (d * d).sum(-1)).sum() / nb_wsum

    nb_pairs_n = _pbr.coords.shape[0] if args.lambda_smooth > 0 else 0
    del _x, _pbr, _f, _im, _fg
    torch.cuda.empty_cache()

    # ── baseline: the frozen model's own score, for reference ────────────────
    _sizes = [int(inter_mask(f).sum()) for f in ALL_FRAMES]
    print(f'\n[REGION] loss/eval taken on render AND gt (per frame).')
    print(f'  render mask (fixed)      {int(renderer.mask.sum()):,} px')
    print(f'  target-valid intersection min {min(_sizes):,}  max {max(_sizes):,}  '
          f'mean {sum(_sizes)//len(_sizes):,}')
    # With make_targets.py filling 100% of the silhouette, valid_*.npy is
    # all-true there and GATE-align has already proven the two silhouettes are
    # the same pixels — so this region IS the render mask, every frame, and the
    # PSNR denominator is constant across frames AND across arms. The older
    # targets left a per-frame fringe out, which is why this used to carry a
    # "not comparable to a render-mask PSNR" caveat. It no longer applies.
    assert min(_sizes) == max(_sizes) == int(renderer.mask.sum()), (
        f'expected the loss region to be exactly the render mask '
        f'({int(renderer.mask.sum()):,}), got {min(_sizes):,}..{max(_sizes):,}')
    print(f'  -> constant across all {len(ALL_FRAMES)} frames and identical for '
          f'every arm, so PSNR is directly comparable', flush=True)

    # ── CONFIDENCE FIELD (the confidence-weighted arm) ────────────────────────────────────────────
    # w = cos(n,v) / footprint, hard-zeroed below tau. Constant: the mesh and the
    # camera are both fixed, so this is built once alongside renderer.mask.
    conf_w = torch.ones_like(renderer.cos_inc)
    CONF_ON = args.conf_weight or args.lambda_con > 0 or args.lambda_smo > 0
    if CONF_ON:
        # tau either fixed (the confidence-weighted arm) or read off this object's own distribution
        # (the regularised arms). The quantile form is what transfers: 'drop the worst q of
        # what THIS camera sees of THIS mesh' needs no per-object tuning.
        if args.conf_quantile >= 0:
            TAU = float(torch.quantile(renderer.cos_inc[renderer.mask].float(),
                                       args.conf_quantile))
            print(f'\n[TAU] quantile q={args.conf_quantile} of this object -> '
                  f'tau={TAU:.4f}   (fixed --conf-tau would have been '
                  f'{args.conf_tau})', flush=True)
        else:
            TAU = args.conf_tau

        # the confidence-ramp arm: FEATHERED CUTOFF. The step 1[cos >= TAU] is replaced by a
        # smoothstep ramp between TAU_LO and TAU_HI. Rationale: a hard 0/1 edge
        # makes the loss discontinuous across the surface — a voxel is supervised
        # and its neighbour, a fraction of a voxel away, is not — while those
        # voxels are coupled through the frozen self-attention that does the
        # propagating. smoothstep is C1: value AND slope are 0 at TAU_LO and 1 at
        # TAU_HI, so neither the weight nor its derivative jumps.
        #
        # The band is set by QUANTILES of this object's own incidence
        # distribution, not by absolute cosines. 'the worst q of what THIS camera
        # sees of THIS mesh' transfers between objects; a fixed 0.15/0.35 does
        # not — on a flat panel or a concave interior the same cosines select a
        # completely different fraction of the surface.
        #
        # TAU_LO == TAU_HI reproduces the confidence-weighted arm step exactly. That is the gate.
        if args.conf_ramp_qlo >= 0 and args.conf_ramp_qhi >= 0:
            _cm = renderer.cos_inc[renderer.mask].float()
            TAU_LO = float(torch.quantile(_cm, args.conf_ramp_qlo))
            TAU_HI = float(torch.quantile(_cm, args.conf_ramp_qhi))
            print(f'\n[RAMP] band from quantiles q=({args.conf_ramp_qlo},'
                  f'{args.conf_ramp_qhi}) of this object -> '
                  f'tau_lo={TAU_LO:.4f}  tau_hi={TAU_HI:.4f}', flush=True)
        else:
            TAU_LO = TAU_HI = TAU          # exact the confidence-weighted arm behaviour
        assert TAU_LO <= TAU_HI, f'need tau_lo <= tau_hi, got {TAU_LO} > {TAU_HI}'

        _fp = renderer.footprint.clamp_min(1e-12)
        _w = renderer.cos_inc / _fp
        _w = torch.where(renderer.fp_valid, _w, torch.zeros_like(_w))
        if TAU_HI > TAU_LO:
            _t = ((renderer.cos_inc - TAU_LO) / (TAU_HI - TAU_LO)).clamp(0.0, 1.0)
            _sig = _t * _t * (3.0 - 2.0 * _t)          # smoothstep, C1 at both ends
        else:
            _sig = (renderer.cos_inc >= TAU_LO).to(_w.dtype)   # the original step
        _w = _w * _sig
        _w = torch.where(renderer.mask, _w, torch.zeros_like(_w))
        _w = _w / _w[_w > 0].median().clamp_min(1e-12)      # keep the scale sane
        conf_w = _w

        _m0 = renderer.mask
        _kept = (conf_w > 0) & _m0
        _cos = renderer.cos_inc[_m0]
        print(f'\n[CONF] Unstructured-Lumigraph weighting  tau={TAU:.4f}')
        print(f'  covered px {int(_m0.sum()):,}   kept {int(_kept.sum()):,} '
              f'({100*float(_kept.sum())/float(_m0.sum()):.1f}%)   '
              f'discarded {int(_m0.sum()) - int(_kept.sum()):,}')
        print(f'  incidence over the mask: mean {float(_cos.mean()):.3f}   '
              f'below tau {100*float((_cos < TAU).float().mean()):.1f}%   '
              f'invalid footprint (silhouette-straddling) '
              f'{int((_m0 & ~renderer.fp_valid).sum()):,}')
        _fpm = renderer.footprint[_kept]
        print(f'  footprint on kept px: median {float(_fpm.median()):.3e}  '
              f'p99 {float(_fpm.quantile(0.99)):.3e}  '
              f'ratio p99/median {float(_fpm.quantile(0.99)/_fpm.median()):.1f}x')
        print(f'  weight: median {float(conf_w[_kept].median()):.3f}  '
              f'p1 {float(conf_w[_kept].quantile(0.01)):.4f}  '
              f'p99 {float(conf_w[_kept].quantile(0.99)):.3f}', flush=True)
        assert float(_kept.sum()) / float(_m0.sum()) > 0.4, (
            f'CONF gate: tau={TAU:.4f} discards '
            f'{100*(1-float(_kept.sum())/float(_m0.sum())):.1f}% of the supervised '
            f'pixels — too aggressive, the adapter would be fit from a sliver.')

        # GATE-ramp. The one number that says whether the feather did anything and
        # whether it did too much. the confidence-weighted arm's weights sit at median 1.0 with p1 0.11,
        # so the pixels the ramp re-admits are ALREADY nearly weightless: the band
        # can only ever claim a small share of the loss. Print it rather than
        # assume it. If the share is ~0 the run cannot differ from the confidence-weighted arm and
        # there is no point spending the GPU; if it is large, the unreliable
        # pixels now dominate, which is exactly what the hard cutoff existed to
        # prevent. Either way this is knowable BEFORE training, unlike the 4.8%
        # confidence field that ran a full arm before anyone looked at it.
        _tot_w = float(conf_w[_m0].sum())
        if TAU_HI > TAU_LO:
            _band = _m0 & (renderer.cos_inc >= TAU_LO) & (renderer.cos_inc < TAU_HI)
            _band_w = float(conf_w[_band].sum())
            _share = _band_w / max(_tot_w, 1e-12)
            print(f'\n[GATE-ramp] feather band cos in [{TAU_LO:.4f}, {TAU_HI:.4f})')
            print(f'  pixels in band {int(_band.sum()):,} '
                  f'({100*float(_band.sum())/float(_m0.sum()):.2f}% of covered)')
            print(f'  band share of total loss mass  {100*_share:.3f}%')
            print(f'  kept px {int(_kept.sum()):,}  vs the hard-cut behaviour '
                  f'{int(((renderer.cos_inc >= TAU_HI) & renderer.fp_valid & _m0).sum()):,}',
                  flush=True)
            if _share < 1e-4:
                print('  [GATE-ramp] WARNING: band carries <0.01% of the loss — '
                      'this run cannot measurably differ from the confidence-weighted arm.', flush=True)
        else:
            print(f'\n[GATE-ramp] ramp DISABLED (tau_lo == tau_hi == {TAU_LO:.4f}) '
                  f'— exact hard cutoff', flush=True)

    # ── the regularised arms: per-voxel confidence, frozen cache, regularisers ────────────
    REG_ON = args.lambda_con > 0 or args.lambda_smo > 0
    # the neighbour-consistency arm: the pairwise term is OFF. R_smo penalises ||d_i - d_j||^2 between
    # neighbours, and the minimiser of that, chained across the graph, is a single
    # CONSTANT edit for every voxel. the regularised arms measured the consequence: 19.51 dB
    # (the confidence-weighted arm, no reg) -> 12.70 dB, SSIM 0.728 -> 0.404, output collapsed toward
    # frozen. With one fixed camera seeing 16.8% of the object, the term was
    # applied to the ~85% nothing observes, and those voxels are coupled to the
    # visible ones through 30 frozen self-attention blocks, so flattening them
    # dragged the visible surface with them. RegNeRF (CVPR 2022) reached the same
    # conclusion: it smooths GEOMETRY but regularises APPEARANCE with a learned
    # patch likelihood, never a smoothness term.
    # R_con survives because it is a CEILING, not a target: relu(||d||^2 - m^2) is
    # exactly zero inside the budget, so direction and pattern stay free and the
    # cross-attention propagation is untouched. It only clips the blow-ups, which
    # is what the rim white patches are. This is NOT the 'anchor to frozen' design
    # that was measured to produce a dark band at alpha=0 -- frozen is globally
    # dark (0.062 vs a truth near 0.31) and a target would drag everything there.
    _SMO_ON = args.lambda_smo > 0
    r_pairs = r_u = r_c = frz_cache = None
    r_wij = r_wijsum = None
    if REG_ON:
        from scipy.spatial import cKDTree
        # coords come from the frozen decode below; the set is deterministic
        # (it depends only on shape_slat's structure) and asserted every frame.
        with torch.no_grad():
            _x0f = run_ode(flow, noise, conds[ALL_FRAMES[0]], ss_n, T_PAIRS)
            _p0f = decoder(_x0f * tex_std + tex_mean) * 0.5 + 0.5
        _c0 = _p0f.coords.detach().clone()
        _ss = tuple(_p0f.spatial_shape)
        N_vox = _c0.shape[0]
        del _x0f, _p0f
        torch.cuda.empty_cache()
        # only the pairwise term needs the adjacency; skip the build when it is off
        if _SMO_ON:
            r_pairs, _sk, _or, _S = build_neighbour_graph(_c0, _ss)

        # c_i : how well-measured voxel i is. Taken from the NEAREST RESOLVED
        # SURFACE POINT rather than from direct pixel hits — the voxel grid is
        # ~30x finer than the render, so a pixel-hit map labels only ~3% of
        # voxels and the weight degenerates to a constant.
        _vc = ((_c0[:, 1:4].float() + 0.5) / args.resolution - 0.5).cpu().numpy()
        _surf = renderer.pos.detach().cpu().numpy()
        _pw = conf_w[renderer.mask].detach().cpu().numpy()      # the confidence-weighted arm weight
        # normalise by a high PERCENTILE, not the max: one outlier pixel would
        # otherwise squash every other weight toward zero, flattening c_i and
        # silently turning a targeted regulariser into a uniform one.
        _p99 = float(np.percentile(_pw, 99))
        print(f'  pixel weight: median {float(np.median(_pw)):.3f}  p99 {_p99:.3f}  '
              f'max {float(_pw.max()):.3f}   (normalising by p99)')
        _pw = np.clip(_pw / max(_p99, 1e-12), 0.0, 1.0)
        _dist, _near = cKDTree(_surf).query(_vc, k=1)
        # falloff measured in VOXELS, so it is resolution- and scale-free
        _sigma = args.reg_k_voxels / float(args.resolution)
        r_c = torch.from_numpy(
            _pw[_near] * np.exp(-(_dist / _sigma) ** 2)).float().to(DEVICE)
        r_c = r_c.clamp(0, 1)
        r_u = (1.0 - r_c) ** args.reg_gamma                     # the switch
        r_hi = r_c > 0.5          # the VERIFIED voxels the magnitude budget uses
        _HAS_HI = bool(r_hi.any())        # constant; do not recompute per step
        # GATE-budget. If the verified set were empty, m2 falls back to dn2.max()
        # and relu(dn2 - max) is zero EVERYWHERE — the ceiling would silently do
        # nothing and the run would look like it regularised when it did not.
        # the regularised arms lost a run to a mask that was quietly wrong; fail loudly instead.
        _n_hi = int(r_hi.sum())
        assert _HAS_HI and _n_hi >= 1000, (
            f'GATE-budget FAILED: only {_n_hi} voxels have c>0.5, so the p99 '
            f'magnitude budget has no credible support and R_con would be inert. '
            f'Raise --reg-k-voxels (currently {args.reg_k_voxels}) or check that '
            f'conf_w / renderer.pos are sane.')
        print(f'[GATE-budget] PASSED — budget estimated from {_n_hi:,} verified '
              f'voxels', flush=True)
        r_usum = r_u.sum().clamp_min(1.0)
        r_csum = r_c.sum().clamp_min(1e-6)
        if _SMO_ON:
            r_wij = torch.maximum(r_u[r_pairs[:, 0]], r_u[r_pairs[:, 1]])
            r_wijsum = r_wij.sum().clamp_min(1.0)

        if _SMO_ON:
            print(f'\n[REG] voxels {N_vox:,}   adjacent pairs {r_pairs.shape[0]:,}')
        else:
            print(f'\n[REG] voxels {N_vox:,}   PAIRWISE TERM OFF: '
                  f'R_smo disabled, magnitude ceiling R_con only')
        print(f'  confidence c_i: mean {float(r_c.mean()):.4f}   '
              f'>0.5 {int((r_c > 0.5).sum()):,} ({100*float((r_c > 0.5).float().mean()):.1f}%)'
              f'   ==0 {int((r_c == 0).sum()):,}')
        print(f'  verified set for the magnitude budget (c>0.5): '
              f'{int(r_hi.sum()):,} voxels')
        print(f'  falloff sigma = {args.reg_k_voxels} voxels = {_sigma:.5f} '
              f'in normalised units (scale free)', flush=True)

        # FROZEN reference per frame. Noise and conditioning are both fixed, so
        # this is deterministic and cached once: [F+1, N, 3] fp16.
        _t0 = time.time()
        frz_cache = torch.zeros(N_FRAMES + 1, N_vox, 3,
                                dtype=torch.float16, device=DEVICE)
        with torch.no_grad():
            for _fi in ALL_FRAMES:
                _xf = run_ode(flow, noise, conds[_fi], ss_n, T_PAIRS)   # no lora_ctx
                _pf = decoder(_xf * tex_std + tex_mean) * 0.5 + 0.5
                assert _pf.coords.shape[0] == N_vox and torch.equal(
                    _pf.coords, _c0), (
                    'frozen voxel COORDS changed between frames — the cache would '
                    'be misaligned with the training decode, silently corrupting d')
                frz_cache[_fi] = _pf.feats[:, :3].half()
                del _xf, _pf
        print(f'[REG] frozen reference cached for {len(ALL_FRAMES)} frames in '
              f'{time.time()-_t0:.0f}s   '
              f'{frz_cache.numel()*2/2**30:.2f} GiB', flush=True)
        torch.cuda.empty_cache()

    _CHK = {'done': False}

    def reg_terms(pbr, fi):
        """
        Both terms act on d = y - y_frozen, THE ADAPTER'S EDIT, never on the
        absolute colour. That distinction matters: frozen already carries real
        spatial detail, and constraining y would erase it — on the back, where
        u_ij = 1 for 73% of voxels, an absolute smoothness term would flatten
        the entire far side into a single colour.

        R_con : magnitude budget. An unsupervised voxel may edit in any
                direction and any pattern, but not harder than the hardest edit
                seen on VERIFIED surface. Zero penalty inside the budget.
        R_smo : the edit should vary smoothly where it is unverified. Frozen's
                own texture passes through untouched.
        """
        if not _CHK['done']:
            assert pbr.coords.shape[0] == N_vox and torch.equal(pbr.coords, _c0), (
                'the TRAINING decode has different voxel coords than the frozen '
                'cache — d = y - y_frozen would be comparing different voxels')
            _CHK['done'] = True
            print(f'  [REG-check] training decode coords match the frozen cache '
                  f'exactly ({N_vox:,} voxels)', flush=True)
        assert pbr.coords.shape[0] == N_vox, 'voxel count changed mid-run'
        # .float() is mandatory: the texture decoder is fp16, and squaring small
        # differences in fp16 underflows to exactly zero — the same flush-to-zero
        # that made GATE-bwd read 0.0 until LOSS_SCALE was introduced.
        y = pbr.feats[:, :3].float()
        d = y - frz_cache[fi].float()
        # A MAGNITUDE BUDGET, not a target edit.
        #
        # Matching every unsupervised voxel to one mean edit d_bar would make the
        # whole back `frozen + d_bar` — frozen's pattern under a constant shift —
        # overwriting whatever cross-attention propagated there. That destroys the
        # very thing the cross-attention placement exists to produce.
        #
        # The failure we actually observed is runaway MAGNITUDE (p99 luminance
        # 0.875 against a median of 0.337), not unwanted variation. So budget the
        # magnitude and leave direction and pattern completely free: inside the
        # budget the penalty is exactly zero. m is read off the verified surface,
        # so it is self-calibrating per object and per frame.
        # SQUARED magnitude, never .norm(). At init B = 0 so d is EXACTLY zero
        # for every voxel, and the gradient of sqrt at 0 is undefined — norm()
        # would emit NaN on the very first step and poison the whole run.
        # (d*d).sum() is smooth everywhere and its gradient at d=0 is 0.
        dn2 = (d * d).sum(-1)
        m2 = (torch.quantile(dn2[r_hi].detach(), 0.99) if _HAS_HI
              else dn2.detach().max())
        R_con = (r_u * torch.relu(dn2 - m2)).sum() / r_usum
        # the neighbour-consistency arm: OFF by default. See the block at REG_ON for why — its minimiser
        # is a constant edit, which collapses the output toward frozen.
        if _SMO_ON:
            # on the EDIT, not the colour — see the docstring
            dv = d[r_pairs[:, 0]] - d[r_pairs[:, 1]]
            R_smo = (r_wij * (dv * dv).sum(-1)).sum() / r_wijsum
        else:
            R_smo = dn2.new_zeros(())
        return R_con, R_smo


    # ALL frames, and the SAME frames the final number uses. Previously this was
    # evaluate(HELD_OUT[:4]) against a final evaluate(HELD_OUT) — 4 frames vs 15,
    # printed side by side as if comparable. Difficulty rises monotonically over
    # the sequence (24.56 dB at frame 5 down to 19.02 at 145), so the frozen arm
    # was sitting only the easy quarter of the exam.
    base = evaluate(ALL_FRAMES)
    print(f'\n[FROZEN] all {len(ALL_FRAMES)} frames before training: '
          f'PSNR {base["psnr_mean"]:.3f}  SSIM {base["ssim_mean"]:.4f}', flush=True)

    # ── resume ───────────────────────────────────────────────────────────────
    # ── PERCEPTUAL TERM (optional) ───────────────────────────────────────────
    # Built once. The crop box comes from renderer.mask, which is constant
    # because the mesh and the camera are, so the framing contributes exactly
    # zero to any comparison between frames.
    lpips_term = None
    if args.lpips:
        import lpips as _lpips_pkg
        _lp_net = _lpips_pkg.LPIPS(net='alex', verbose=False).to(DEVICE).eval()
        for _p in _lp_net.parameters():
            _p.requires_grad_(False)          # frozen judge, never optimised
        _ys, _xs = torch.nonzero(renderer.mask, as_tuple=True)
        _y0, _y1 = int(_ys.min()), int(_ys.max()) + 1
        _x0, _x1 = int(_xs.min()), int(_xs.max()) + 1

        def lpips_term(_img, _gt):
            # [1,H,W,3] in [0,1] -> [1,3,h,w] in [-1,1], AlexNet's domain.
            # NOT clamped and NOT resized: clamping kills the gradient on any
            # pixel pushed out of range, and downsampling would discard the high
            # frequencies this term exists to restore.
            a = _img[0, _y0:_y1, _x0:_x1].permute(2, 0, 1).unsqueeze(0).float()
            b = _gt[0, _y0:_y1, _x0:_x1].permute(2, 0, 1).unsqueeze(0).float()
            return _lp_net(a * 2 - 1, b * 2 - 1).mean()

        print(f'\n[LPIPS] AlexNet, frozen, {sum(p.numel() for p in _lp_net.parameters()):,} params\n'
              f'  crop y[{_y0}:{_y1}] x[{_x0}:{_x1}] = {_y1-_y0}x{_x1-_x0} at native resolution\n'
              f'  weight {args.w_lpips}  (loss = recon + w * lpips)', flush=True)

    opt = torch.optim.AdamW(reg.parameters(), lr=args.lr, weight_decay=0.0)
    start_ep, best = 1, -1e9
    # Newest first. A 4h wall-clock kill can land in the middle of a torch.save,
    # and the newest file is then the ONE that is torn. Taking ckpts[-1] and
    # letting torch.load raise turns a recoverable timeout into a job that fails
    # identically on every requeue until --max-retries is spent. So walk down:
    # the epoch before a torn write is intact, and losing one epoch is nothing.
    # _atomic_save makes torn files rare; this makes them harmless.
    ckpts = sorted((OUT / 'ckpts').glob('lora_e*.pt'), reverse=True)
    for _ck in ckpts:
        try:
            st = torch.load(_ck, map_location=DEVICE, weights_only=False)
        except Exception as _e:
            print(f'[RESUME] {_ck.name} unreadable ({type(_e).__name__}: {_e}) '
                  f'-- falling back to the epoch before it', flush=True)
            continue
        # GATE-resume. Belt to the config hash's braces: a checkpoint may only be
        # inherited by a run using the SAME mesh and the SAME targets. Without
        # this, two runs that hash alike silently continue each other's training
        # on different data, which is exactly what happened once and
        # produced two unusable results with no error anywhere.
        _prev = st.get('cfg', {})
        for _k in ('mesh', 'gt_render_dir', 'render_res'):
            _a, _b = _prev.get(_k), _CFG.get(_k)
            assert _a is None or _a == _b, (
                f'GATE-resume FAILED: {_ck.name} was written by a run whose '
                f'{_k} was\n    {_a}\nbut this run uses\n    {_b}\n'
                f'Refusing to resume — that would continue another experiment\'s '
                f'adapter on different data. Delete {OUT} or fix the paths.')
        reg.load_state_dict(st['reg']); opt.load_state_dict(st['opt'])
        start_ep, best = st['epoch'] + 1, st['best']
        print(f'[RESUME] from {_ck.name}, epoch {start_ep}   '
              f'[GATE-resume] PASSED — same mesh and targets', flush=True)
        break
    else:
        print('[RESUME] no checkpoint — starting fresh', flush=True)

    csv = OUT / 'logs' / 'epoch_metrics.csv'
    if not csv.exists():
        csv.write_text('epoch,loss,monitor_psnr,monitor_ssim,B_mean,B_max,'
                       # the self-attention arm adds the two self-attention columns. Without them
                       # the CSV cannot answer "did the self-attn half train at
                       # all, or did cross-attn do everything?" — which is the
                       # first question to ask of this run's result.
                       'dB_toq,dB_tokv,dB_toout,dB_saqkv,dB_saout,'
                       'n_zero,time_s\n')

    print(f'\n[TRAIN] epochs {start_ep}->{EPOCHS}   {len(TRAIN)} frames/epoch   '
          f'1 graded flow eval + up to {STEPS-1} no_grad evals per step', flush=True)
    rng = np.random.default_rng(args.seed)
    _ed = _ec = _es = 1e-8            # running scales for the ratio-based lambdas
    history = []

    for ep in range(start_ep, EPOCHS + 1):
        te = time.time()
        order = rng.permutation(TRAIN)
        tot, last_rep = 0.0, None
        _oob, _nb = 0.0, 0            # fraction of predicted px outside [0,1]
        _smtot, _smn = 0.0, 0         # neighbour-consistency term
        _rctot = _rstot = 0.0; _rn = 0 # the regularised arms regularisers
        _lptot, _lpn = 0.0, 0         # perceptual term, logged separately from
                                      # the data term so their RATIO is visible —
                                      # that ratio is how w_lpips gets chosen
        for si, fi in enumerate(order, 1):
            fi = int(fi)
            k = int(rng.integers(0, STEPS))          # a knot of the REAL schedule
            t_k = T_SEQ[k]

            # BACKWARD MUST RUN INSIDE lora_ctx. Gradient checkpointing recomputes
            # the flow forward during backward, so the hooks have to still be
            # installed at that moment. With .backward() outside the block the
            # recomputed graph is the BARE model and torch raises
            #   CheckpointError: A different number of tensors was saved during the
            #   original forward and recomputation.  forward: 74  recomputation: 57
            # — the 17 missing tensors being exactly the LoRA deltas.
            with lora_ctx(flow, reg):
                x_k = run_ode(flow, noise, conds[fi], ss_n, T_PAIRS[:k])
                v = flow_eval(flow, x_k, t_k, conds[fi], ss_n)
                x0 = pred_to_xstart(x_k, t_k, v)
                _need_pbr = args.lambda_smooth > 0 or REG_ON
                _ret = decode_render(x0, clamp=False, return_pbr=_need_pbr)
                img, _pbr_t = _ret if _need_pbr else (_ret, None)
                _m = inter_mask(fi)
                if args.conf_weight:
                    # l2 branch is the original expression unchanged; l1 swaps the
                    # square for an absolute value and touches nothing else --
                    # same mask, same weights, same normalisation, same reduction.
                    _d  = (img - gts[fi])[0][_m]                 # [N,3]
                    _se = _d.abs() if args.recon == 'l1' else _d ** 2
                    _wm = conf_w[_m]                             # [N]
                    loss_data = (_wm[:, None] * _se).sum() / (_wm.sum() * 3).clamp_min(1e-9)
                else:
                    _d = (img - gts[fi])[:, _m]
                    loss_data = (_d.abs() if args.recon == 'l1' else _d ** 2).mean()
                loss = loss_data
                _lpv = 0.0
                if lpips_term is not None:
                    _lp = lpips_term(img, gts[fi])
                    loss = loss + args.w_lpips * _lp
                    _lpv = float(_lp)
                    _lptot += _lpv; _lpn += 1
                if REG_ON:
                    _Rc, _Rs = reg_terms(_pbr_t, fi)
                    # lambda is a RATIO, not an absolute: each regulariser is
                    # scaled to contribute a fixed fraction of the data term.
                    # An absolute lambda does not transfer between objects — the
                    # same 0.1 was 0.07% of the loss here and could be 70%
                    # elsewhere. EMAs are detached, so no gradient flows through
                    # the scaling itself.
                    _ed = 0.9 * _ed + 0.1 * float(loss_data)
                    _ec = 0.9 * _ec + 0.1 * float(_Rc)
                    _es = 0.9 * _es + 0.1 * float(_Rs)
                    if args.lambda_con > 0:
                        loss = loss + (args.lambda_con * _ed / max(_ec, 1e-12)) * _Rc
                    if args.lambda_smo > 0:
                        loss = loss + (args.lambda_smo * _ed / max(_es, 1e-12)) * _Rs
                    _rctot += float(_Rc); _rstot += float(_Rs); _rn += 1
                _ls = 0.0
                if args.lambda_smooth > 0:
                    _sm = smooth_term(_pbr_t)
                    loss = loss + args.lambda_smooth * _sm
                    _ls = float(_sm)
                    _smtot += _ls; _smn += 1
                with torch.no_grad():
                    _px = img[:, _m]
                    _oob += float(((_px < 0.0) | (_px > 1.0)).float().mean())
                    _nb += 1

                opt.zero_grad(set_to_none=True)
                (loss * args.loss_scale).backward()
            for p in reg.parameters():
                if p.grad is not None:
                    p.grad.div_(args.loss_scale)
            if si == len(order):
                last_rep = grad_report(reg)
            torch.nn.utils.clip_grad_norm_(reg.parameters(), args.grad_clip)
            opt.step()
            # The decoder's weights carry requires_grad only to keep flex_gemm from
            # returning grad_weight=None; nothing optimises them. opt.zero_grad()
            # does not touch them (the optimizer holds reg.parameters() alone), so
            # clear them here or they accumulate across the whole run.
            if args.decoder_grad_weights:
                for q in decoder.parameters():
                    q.grad = None
            tot += float(loss_data)          # data term only, so the number is
            lv = float(loss_data)            # comparable across lambda settings
            del loss, loss_data, img, x0, v, x_k

            if si % 25 == 0 or si == 1:
                print(f'  e{ep:02d} [{si:03d}/{len(order)}] f{fi:04d} k={k:02d} '
                      f't={t_k:.3f}  {args.recon}={lv:.5f}  '
                      + (f'lpips={_lpv:.5f} (w*lp={args.w_lpips*_lpv:.5f}, '
                         f'{100*args.w_lpips*_lpv/max(lv+args.w_lpips*_lpv,1e-12):.0f}% of loss)  '
                         if lpips_term is not None else '')
                      + f'peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB', flush=True)

        ev = evaluate(MONITOR)          # monitoring subset; reported numbers use ALL_FRAMES
        pn = param_norms(reg)
        dt = time.time() - te
        row = dict(epoch=ep, loss=tot / len(order), **{k: v for k, v in ev.items()
                                                       if k != 'per_frame'})
        history.append(row)
        print(f'[EPOCH {ep}/{EPOCHS}] loss={row["loss"]:.5f}  '
              f'monitor PSNR={ev["psnr_mean"]:.3f}+-{ev["psnr_std"]:.3f}  '
              f'SSIM={ev["ssim_mean"]:.4f}  |B|={np.mean(list(pn.values())):.4f}  '
              f'oob={100*_oob/max(_nb,1):.2f}%'
              + (f'  Rcon={_rctot/max(_rn,1):.3e}(x{args.lambda_con*_ed/max(_ec,1e-12):.1f})'
                 f'  Rsmo={_rstot/max(_rn,1):.3e}(x{args.lambda_smo*_ed/max(_es,1e-12):.1f})'
                 if REG_ON else '')
              + (f'  smooth={_smtot/max(_smn,1):.5f}'
                 f'({100*args.lambda_smooth*(_smtot/max(_smn,1))/max(tot/len(order),1e-9):.1f}%'
                 f' of data loss)' if args.lambda_smooth > 0 else '')
              + f'  {dt:.0f}s', flush=True)
        with open(csv, 'a') as fh:
            fh.write(f'{ep},{row["loss"]:.6f},{ev["psnr_mean"]:.4f},'
                     f'{ev["ssim_mean"]:.5f},{np.mean(list(pn.values())):.5f},'
                     f'{max(pn.values()):.5f},'
                     f'{(last_rep or {}).get("per_target",{}).get("to_q",0):.4e},'
                     f'{(last_rep or {}).get("per_target",{}).get("to_kv",0):.4e},'
                     f'{(last_rep or {}).get("per_target",{}).get("to_out",0):.4e},'
                     f'{(last_rep or {}).get("per_target",{}).get("sa_qkv",0):.4e},'
                     f'{(last_rep or {}).get("per_target",{}).get("sa_out",0):.4e},'
                     f'{(last_rep or {}).get("n_zero",-1)},{dt:.1f}\n')

        _atomic_save(dict(reg=reg.state_dict(), opt=opt.state_dict(),
                          epoch=ep, best=best, cfg=_CFG),
                     OUT / 'ckpts' / f'lora_e{ep:03d}.pt')
        if ev['psnr_mean'] > best:
            best = ev['psnr_mean']
            _atomic_save(dict(reg=reg.state_dict(), epoch=ep, psnr=best, cfg=_CFG),
                         OUT / 'ckpts' / 'lora_best.pt')
            print(f'  [CKPT] new best -> lora_best.pt ({best:.3f} dB)', flush=True)

        if ep % 5 == 0 or ep == EPOCHS:
            strip = []
            for fi in MONITOR[:3]:
                strip.append(np.concatenate([
                    (gts[fi][0].cpu().numpy() * 255).astype(np.uint8),
                    (full_inference(fi)[0].cpu().numpy() * 255).astype(np.uint8)], axis=1))
            Image.fromarray(np.concatenate(strip, axis=0)).save(
                OUT / 'diag' / f'e{ep:03d}_gt_vs_ours.jpg', quality=92)
        json.dump(history, open(OUT / 'loss_history.json', 'w'), indent=2)
        torch.cuda.empty_cache()

    st = torch.load(OUT / 'ckpts' / 'lora_best.pt', map_location=DEVICE,
                    weights_only=False)
    reg.load_state_dict(st['reg'])
    final = evaluate(ALL_FRAMES)          # identical frame set to `base`
    json.dump(dict(frozen=base, final=final, best_epoch=st['epoch']),
              open(OUT / 'final_eval.json', 'w'), indent=2)
    print(f'\n[FINAL] frozen  PSNR {base["psnr_mean"]:.3f}  SSIM {base["ssim_mean"]:.4f}')
    print(f'[FINAL] arm{_ARM}  PSNR {final["psnr_mean"]:.3f}  SSIM {final["ssim_mean"]:.4f}  '
          f'(epoch {st["epoch"]})')
    print('[FINAL] geometry unchanged BY CONSTRUCTION — TRELLIS.2 takes the mesh '
          'as an input and emits 32/6 channels, none of which is a position.')
    print(f'\n[SAVE] {OUT}\n[DONE]', flush=True)


if __name__ == '__main__':
    # Required to TRAIN, but not to import. bake.py, render.py and export.py
    # import this module with a synthetic argv to rebuild the adapter registry
    # that matches a checkpoint; they never touch the reference frames. Marking
    # these required=True in argparse would break that import path.
    for _flag, _val in (('--gt-dir', args.gt_dir),
                        ('--gt-render-dir', args.gt_render_dir)):
        if not _val:
            raise SystemExit(f'{_flag} is required to train')
    _begin_run()
    main()
