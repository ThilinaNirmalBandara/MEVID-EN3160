import sys, os
from pathlib import Path
import json

# ensure we can import config + package when running as: python scripts/parse_mevid.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# app/app.py
import streamlit as st
from pathlib import Path
import pickle
import numpy as np
from config import ARTIFACTS, THUMBS_DIR
from mevid_textsearch.faiss_utils import load_index
from mevid_textsearch.clip_utils import encode_text

st.set_page_config(page_title="MEVID Text→Video Person Search", layout="wide")

@st.cache_resource
def _load_index_and_meta():
    index = load_index(ARTIFACTS / "faiss.index")
    meta = pickle.load(open(ARTIFACTS / "meta_test.pkl", "rb"))
    return index, meta

index, meta = _load_index_and_meta()

st.title("🔎 MEVID Text→Tracklet Search (CLIP + FAISS)")
query = st.text_input("Describe the person (e.g., 'man in black hoodie with red backpack'):", "")
topk = st.slider("Top-K", 5, 50, 20)

if st.button("Search") or query:
    with st.spinner("Encoding & searching..."):
        qvec = encode_text([query]).astype("float32")
        sims, ids = index.search(qvec, topk)
        ids, sims = ids[0].tolist(), sims[0].tolist()

    st.subheader(f"Results for: “{query}”")
    cols = st.columns(4, gap="small")
    for rank, (i, s) in enumerate(zip(ids, sims), 1):
        m = meta[i]
        c = cols[(rank-1) % 4]
        with c:
            thumb_gif = THUMBS_DIR / f"track_{i}.gif"
            if thumb_gif.exists():
                st.image(str(thumb_gif), caption=f"rank {rank} • score {s:.3f}")
            else:
                st.text(f"[no preview] rank {rank} • score {s:.3f}")
            st.caption(f"tid {m['tid']}  pid {m['pid']}  outfit {m['outfit']}  cam {m['camid']}")
