import streamlit as st
import sys
from pathlib import Path
import sys, os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config import ARTIFACTS, MEVID_ROOT, THUMBS_DIR
from mevid_textsearch.hybrid_search import HybridSearchEngine
from PIL import Image
import numpy as np

# Page config
st.set_page_config(
    page_title="MEvid Person Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .result-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .score-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        margin: 2px;
    }
    .score-high {
        background-color: #4CAF50;
        color: white;
    }
    .score-medium {
        background-color: #FF9800;
        color: white;
    }
    .score-low {
        background-color: #f44336;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def get_score_class(score):
    """Get CSS class for score badge"""
    if score >= 0.8:
        return "score-high"
    elif score >= 0.5:
        return "score-medium"
    else:
        return "score-low"


def display_results(results, query_text):
    """Display search results in a grid"""
    
    if not results:
        st.warning("No results found. Try adjusting your query or settings.")
        return
    
    # Display in grid
    cols_per_row = 3
    
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(results):
                result = results[i + j]
                
                with col:
                    # Load thumbnail
                    thumb_path = THUMBS_DIR / f"track_{result.track_id}.gif"
                    
                    if thumb_path.exists():
                        try:
                            img = Image.open(thumb_path)
                            st.image(img, use_container_width=True)
                        except:
                            st.image("https://via.placeholder.com/320x240?text=No+Image", use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/320x240?text=No+Thumbnail", use_container_width=True)
                    
                    # Result info card
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>Rank #{result.rank}</h4>
                        <p><strong>Track ID:</strong> {result.track_id}</p>
                        <p><strong>Person ID:</strong> {result.person_id}</p>
                        <p><strong>Outfit:</strong> {result.outfit} | <strong>Camera:</strong> {result.camera_id}</p>
                        
                        <div>
                            <span class="score-badge {get_score_class(result.clip_score)}">
                                CLIP: {result.clip_score:.3f}
                            </span>
                            <span class="score-badge {get_score_class(result.reid_score)}">
                                ReID: {result.reid_score:.3f}
                            </span>
                        </div>
                        
                        <div style="margin-top: 10px;">
                            <span class="score-badge score-high">
                                Combined: {result.combined_score:.3f}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Expandable frame paths
                    with st.expander("📁 Frame Paths"):
                        for frame in result.frames[:5]:
                            st.code(frame, language="text")
    
    # Download results
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        # Create CSV download
        import pandas as pd
        
        df = pd.DataFrame([
            {
                'Rank': r.rank,
                'Track_ID': r.track_id,
                'Person_ID': r.person_id,
                'Outfit': r.outfit,
                'Camera': r.camera_id,
                'CLIP_Score': r.clip_score,
                'ReID_Score': r.reid_score,
                'Combined_Score': r.combined_score
            }
            for r in results
        ])
        
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"search_results_{query_text.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# Initialize session state
if 'engine' not in st.session_state:
    with st.spinner("🔄 Loading search engine..."):
        st.session_state.engine = HybridSearchEngine(
            artifacts_dir=ARTIFACTS,
            reid_model_path=ARTIFACTS / "reid_model.pth" if (ARTIFACTS / "reid_model.pth").exists() else None
        )
        st.session_state.engine_loaded = True

# Header
st.markdown('<h1 class="main-header">🔍 MEvid Person Search Engine</h1>', unsafe_allow_html=True)
st.markdown("**Hybrid CLIP + Video ReID System** - Search for people across multiple cameras")

# Add explanation of how it works
with st.expander("ℹ️ How This Works", expanded=False):
    st.markdown("""
    ### 🔄 Search Process:
    
    **Your Input** → Text description (e.g., "man in black jacket")
    
    **Stage 1: CLIP (Fast Text-to-Video Search)**
    - Converts your text into a 512-dimensional vector
    - Searches through all 1,754 tracklets in ~0.1 seconds
    - Returns top 50 candidates that match your description
    
    **Stage 2: Video ReID (Accurate Re-ranking)**
    - Takes the best CLIP result as reference
    - Analyzes video sequences using temporal attention
    - Identifies which tracklets show the SAME person
    - Re-ranks: 60% CLIP score + 40% ReID score
    
    **Output** → Top 10 tracklets with:
    - 🎬 Animated GIF showing the person walking
    - 📊 Scores (CLIP, ReID, Combined)
    - 📍 Track ID, Person ID, Camera, Outfit
    - 📁 Frame paths for detailed analysis
    
    ---
    
    ### 🎯 What's a Tracklet?
    
    A **tracklet** is a video sequence of one person captured by one camera:
    - Contains 10-200 frames showing the person moving
    - Each person may have multiple tracklets from different cameras
    - The app shows these as animated GIFs!
    
    **Example:**
    ```
    Person ID: 231 has tracklets in:
    - Camera 507 (Track 1548) ← You search for this
    - Camera 340 (Track 1524) ← ReID finds this match!
    - Camera 329 (Track 1619) ← ReID finds this too!
    ```
    
    This lets you **track a person across multiple cameras** 🎥
    """)

# Sidebar
st.sidebar.header("⚙️ Search Settings")

search_mode = st.sidebar.radio(
    "Search Mode",
    ["Text Search", "Video-to-Video ReID"],
    help="Text: Describe the person | Video: Find same person across cameras"
)

st.sidebar.markdown("---")

# Mode-specific settings
if search_mode == "Text Search":
    st.sidebar.subheader("🔤 Text Search Options")
    
    use_reid = st.sidebar.checkbox(
        "Enable Video ReID",
        value=True,
        help="Use temporal model for better accuracy (slower)"
    )
    
    alpha = st.sidebar.slider(
        "CLIP Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="Higher = more weight on text matching"
    )
    
    topk_clip = st.sidebar.slider(
        "CLIP Candidates",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Number of candidates for re-ranking"
    )
    
    diversity = st.sidebar.checkbox(
        "Camera Diversity",
        value=True,
        help="Prefer results from different cameras"
    )
    
else:
    st.sidebar.subheader("🎥 Video ReID Options")
    
    exclude_same_cam = st.sidebar.checkbox(
        "Exclude Same Camera",
        value=True,
        help="Only show results from different cameras"
    )

topk = st.sidebar.slider(
    "Number of Results",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**💡 Tips:**\n"
    "- Be specific in descriptions\n"
    "- Use clothing colors/items\n"
    "- Enable ReID for accuracy\n"
    "- Increase candidates for better recall"
)

# Main content
if search_mode == "Text Search":
    st.header("🔤 Text-Based Person Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Describe the person you're looking for:",
            placeholder="e.g., man in black jacket with backpack",
            help="Describe clothing, accessories, or appearance"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Example queries
    with st.expander("📝 Example Queries"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👔 Man in black jacket"):
                query = "man in black jacket"
                search_button = True
        
        with col2:
            if st.button("👗 Woman in red shirt"):
                query = "woman in red shirt"
                search_button = True
        
        with col3:
            if st.button("🎒 Person with backpack"):
                query = "person with backpack"
                search_button = True
    
    if search_button and query:
        with st.spinner(f"🔎 Searching for: **{query}**..."):
            results = st.session_state.engine.search(
                query=query,
                mevid_root=MEVID_ROOT,
                topk_clip=topk_clip,
                topk_final=topk,
                alpha=alpha,
                use_reid_rerank=use_reid,
                diversity_penalty=0.03 if diversity else 0.0
            )
        
        st.success(f"✅ Found {len(results)} results!")
        
        # Display results
        display_results(results, query)

else:  # Video-to-Video ReID
    st.header("🎥 Video-to-Video Re-Identification")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        track_id = st.number_input(
            "Enter Track ID:",
            min_value=0,
            max_value=1753,
            value=100,
            help="Track ID from previous search or dataset"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("🔍 Find Matches", type="primary", use_container_width=True)
    
    if search_button:
        with st.spinner(f"🔎 Finding matches for Track {track_id}..."):
            results = st.session_state.engine.person_reidentification(
                reference_track_id=track_id,
                mevid_root=MEVID_ROOT,
                topk=topk,
                exclude_same_camera=exclude_same_cam
            )
        
        st.success(f"✅ Found {len(results)} matches!")
        
        # Display reference tracklet info
        import pickle
        with open(ARTIFACTS / "meta_test.pkl", "rb") as f:
            meta = pickle.load(f)
        
        ref_meta = meta[track_id]
        
        st.info(
            f"**Reference Track:** {track_id} | "
            f"**Person ID:** {ref_meta['pid']} | "
            f"**Outfit:** {ref_meta['outfit']} | "
            f"**Camera:** {ref_meta['camid']}"
        )
        
        # Display results
        display_results(results, f"Track {track_id}")



# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>MEvid Hybrid Person Search | CLIP + Video ReID</p>
        <p>Built with Streamlit 🎈 | Powered by PyTorch 🔥</p>
    </div>
    """,
    unsafe_allow_html=True
)