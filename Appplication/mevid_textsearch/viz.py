# mevid_textsearch/viz.py
from pathlib import Path
from typing import List
from PIL import Image

def _resize_keep_aspect(im: Image.Image, max_side: int = 320) -> Image.Image:
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        im = im.resize((nw, nh), Image.BILINEAR)
    return im

def make_gif_from_images(paths: List[Path], out_path: Path, max_frames: int = 12, fps: int = 6, max_side: int = 320):
    """
    Create an animated GIF from a list of image paths using Pillow only.
    No FFmpeg, no imageio dependency.
    """
    try:
        if not paths:
            return False

        # pick evenly spaced indices
        if len(paths) > max_frames:
            import numpy as np
            sel = np.linspace(0, len(paths) - 1, num=max_frames, dtype=int).tolist()
        else:
            sel = list(range(len(paths)))

        frames = []
        for i in sel:
            p = paths[i]
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            im = _resize_keep_aspect(im, max_side=max_side)
            frames.append(im)

        if not frames:
            return False

        duration_ms = int(1000 / max(1, fps))
        first = frames[0]
        rest = frames[1:] if len(frames) > 1 else []
        # ensure parent exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        first.save(
            out_path,
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            optimize=False,
            format="GIF",
        )
        return True
    except Exception:
        return False
