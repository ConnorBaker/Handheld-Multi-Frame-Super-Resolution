"""Poisson-Gaussian noise (alpha, beta) from a darktable-format noise profile.

darktable's ``noiseprofiles.json`` measures, per camera + ISO, the heteroscedastic
raw noise as ``Var(x) = a*x + b`` for ``x`` the raw value normalized to [0,1]
(black=0, white=1), per channel. That is the SAME model and convention the pipeline
reads from the DNG ``NoiseProfile`` tag -- verified against the Samsung DNG the
pipeline already uses (a6300 ISO200 alpha 8.3e-6 vs phone 2.2e-4, ~26x cleaner, and
``run_fast_MC`` produces a sane curve). So **averaging the per-channel a/b gives the
(alpha, beta) the pipeline expects** -- the noise source for raw formats that lack a
DNG NoiseProfile tag (e.g. Sony ARW).

The bundled ``noiseprofiles_a6300.json`` is the ILCE-6300 subset of darktable's data
(GPL-3.0; attribution inside it). Pass ``path=`` a full ``noiseprofiles.json`` for
any other camera.
"""

from pathlib import Path
import json

import numpy as np

_DEFAULT_PATH = Path(__file__).resolve().parent / "noiseprofiles_a6300.json"


def _model_profiles(path, model):
    """Return the ISO-sorted profile list for ``model`` in a darktable-format JSON."""
    data = json.loads(Path(path).read_text())
    for maker in data["noiseprofiles"]:
        for m in maker["models"]:
            if m["model"] == model:
                return sorted(m["profiles"], key=lambda p: p["iso"])
    raise KeyError(f"model {model!r} not found in {path}")


def lookup(model, iso, path=None):
    """(alpha, beta) for a camera ``model`` + ``iso``, as the pipeline's noise params.

    Averages the per-channel ``a``/``b`` of darktable's profiles and log-log
    interpolates over ISO (noise scales ~linearly with ISO). ISO outside the
    measured range is clamped to the nearest profile.
    """
    profs = _model_profiles(path or _DEFAULT_PATH, model)
    isos = np.array([p["iso"] for p in profs], dtype=np.float64)
    a = np.array([np.mean(p["a"]) for p in profs], dtype=np.float64)
    b = np.array([np.mean(p["b"]) for p in profs], dtype=np.float64)
    li = np.log(np.clip(float(iso), isos.min(), isos.max()))
    log_isos = np.log(isos)
    alpha = float(np.exp(np.interp(li, log_isos, np.log(a))))
    beta = float(np.exp(np.interp(li, log_isos, np.log(np.clip(b, 1e-30, None)))))
    return alpha, beta
