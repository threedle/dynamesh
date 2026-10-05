"""paths.py — resolve TRELLIS.2 and the HuggingFace cache without hard-coding a machine.

WHY THIS FILE EXISTS
  The research tree these scripts came from hard-coded absolute paths in five
  places: the TRELLIS.2 checkout, the HuggingFace cache, and three default
  argument values. Every one of them resolved to one cluster's filesystem, so a
  fresh clone could not run any of the commands in the README.

  All of them route through here instead. Resolution order is the same in every
  case: an explicit flag beats an environment variable, which beats a sensible
  default relative to this repository. Nothing silently falls back to a path that
  only exists on one machine.

TRELLIS.2
  The `trellis2` package ships vendored inside this repository at ./trellis2/
  (three of its files are modified; see the README), so the default root is
  this repository itself. Override with $TRELLIS2_ROOT or --trellis2 to use a
  separate checkout instead. We raise rather than guess: a missing backbone
  must fail loudly at startup, not as an ImportError several seconds into a run.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def trellis2_root(explicit=None) -> Path:
    """Locate the trellis2 package: explicit > $TRELLIS2_ROOT > the vendored copy here."""
    for candidate in (explicit, os.environ.get('TRELLIS2_ROOT'), REPO_ROOT):
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / 'trellis2').is_dir():
            return path
    raise SystemExit(
        "trellis2 package not found. This repository ships a vendored copy under "
        "./trellis2/ — if it is missing, re-clone the repository, or point at a "
        "separate checkout by exporting $TRELLIS2_ROOT.")


def add_trellis2_to_path(explicit=None) -> Path:
    """Put TRELLIS.2 on sys.path so `import trellis2` works, and return its root."""
    root = trellis2_root(explicit)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def hf_home() -> str:
    """HuggingFace cache. Honour the user's own $HF_HOME; never overwrite it."""
    return os.environ.setdefault(
        'HF_HOME', str(Path.home() / '.cache' / 'huggingface'))


def set_offline_from_env() -> bool:
    """Offline HuggingFace mode, OFF unless the user asks for it.

    The research tree forced HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 at import
    time because the cluster's compute nodes have no outbound network and the
    weights were already cached. On any normal machine that setting is fatal on a
    first run: it stops the TRELLIS.2 weights being fetched at all, and the error
    surfaces as a confusing cache miss rather than "you are offline".

    So it is opt-in here. Export DYNAMESH_OFFLINE=1 on an air-gapped node.
    """
    offline = os.environ.get('DYNAMESH_OFFLINE', '') == '1'
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return offline
