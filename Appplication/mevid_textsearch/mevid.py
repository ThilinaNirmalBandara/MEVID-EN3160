# mevid_textsearch/mevid.py
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class Tracklet:
    tid: int
    pid: int
    outfit: int
    camid: int
    names_start: int
    names_end: int
    split: str            # "train" or "test"
    frames: List[str]     # relative paths (e.g., "bbox_test/0201/0201O...jpg")

def _load_lines(path: Path) -> List[str]:
    """Read non-empty, stripped lines."""
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]

def _with_prefix(name: str, split: str) -> str:
    """Ensure the name is prefixed with bbox_<split>/ once, normalize slashes."""
    prefix = f"bbox_{split}/"
    name = name.replace("\\", "/")
    if name.startswith(prefix):
        return name
    return prefix + name

def _f2i(x: str) -> int:
    """Robust float->int for '0', '12.0', '9.000000e+00', etc."""
    return int(float(x))

def build_tracklets(mevid_root: Path, split: str = "test") -> List[Tracklet]:
    """
    Parse MEVID annotation files into Tracklets for a split ("train" or "test").
    Expects:
      <mevid_root>/<split>_name.txt
      <mevid_root>/track_<split>_info.txt
      and images under <mevid_root>/bbox_<split>/**.jpg
    """
    assert split in ("train", "test")
    names_file = mevid_root / f"{split}_name.txt"
    info_file  = mevid_root / f"track_{split}_info.txt"
    if not names_file.exists():
        raise FileNotFoundError(f"Missing {names_file}")
    if not info_file.exists():
        raise FileNotFoundError(f"Missing {info_file}")

    names = _load_lines(names_file)
    info_lines = _load_lines(info_file)
    tracks: List[Tracklet] = []

    for tid, line in enumerate(info_lines):
        # tolerate arbitrary whitespace/commas; ignore extra columns after the first 5
        parts = line.replace(",", " ").split()
        if len(parts) < 5:
            continue
        s_str, e_str, pid_str, outfit_str, camid_str = parts[:5]

        # Convert possibly-scientific-notation strings to ints
        s      = _f2i(s_str)
        e      = _f2i(e_str)
        pid    = _f2i(pid_str)
        outfit = _f2i(outfit_str)
        camid  = _f2i(camid_str)

        # bound check & inclusive indexing safeguard
        s = max(0, min(s, len(names) - 1))
        e = max(0, min(e, len(names) - 1))
        if e < s:
            s, e = e, s

        frame_names = names[s:e+1]
        rels = [_with_prefix(n, split) for n in frame_names]

        tracks.append(Tracklet(
            tid=tid, pid=pid, outfit=outfit, camid=camid,
            names_start=s, names_end=e, split=split, frames=rels
        ))
    return tracks

def save_jsonl(objs: List[Dict[str, Any]], path: Path):
    """Write a JSONL file (one JSON object per line)."""
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
