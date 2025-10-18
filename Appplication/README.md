# MEVID Text→Tracklet Search (CLIP + FAISS)

A minimal, end-to-end system:
- Parse MEVID test tracklets
- Extract CLIP embeddings (frame → pooled tracklet)
- Build FAISS index
- Search with natural-language + Streamlit UI

## 1) Setup
```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
