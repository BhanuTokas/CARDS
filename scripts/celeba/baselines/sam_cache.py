"""SAM automatic masks per CelebA-HQ image, cached (bit-packed .npz) in $CARDS_SAM_CACHE when that is set, so EAC,
KernelSHAP and every black box see identical segments and SAM runs once per image. Unset = no caching."""
import os
from pathlib import Path

import numpy as np

CACHE = Path(os.environ["CARDS_SAM_CACHE"]) if os.environ.get("CARDS_SAM_CACHE") else None


def sam_masks(gen, key, raw: np.ndarray) -> np.ndarray:
    """(n_segments, H, W) bool masks of `raw` from SamAutomaticMaskGenerator `gen`; `key` names the cache file."""
    f = CACHE / f"{key}.npz" if CACHE else None
    if f is not None and f.exists():
        z = np.load(f)
        return np.unpackbits(z["m"], axis=-1, count=int(z["w"])).astype(bool)
    m = np.stack([x["segmentation"] for x in gen.generate(raw)])
    if f is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, m=np.packbits(m, axis=-1), w=m.shape[-1])
    return m
