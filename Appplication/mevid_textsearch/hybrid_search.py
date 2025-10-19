# mevid_textsearch/hybrid_search.py
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from .clip_utils import encode_text
from .temporal_reid import VideoReIDExtractor
from .sota_reid import SOTAReIDExtractor  # NEW: SOTA models
from .faiss_utils import load_index

@dataclass
class SearchResult:
    """Container for search results"""
    track_id: int
    person_id: int
    outfit: int
    camera_id: int
    clip_score: float
    reid_score: float
    combined_score: float
    frames: List[str]
    rank: int

class HybridSearchEngine:
    """
    Two-stage hybrid search:
    1. CLIP text-to-image retrieval (coarse search)
    2. Video ReID refinement (fine-grained matching)
    """
    
    def __init__(
        self,
        artifacts_dir: Path,
        reid_model_path: Path = None,
        device: str = None,
        reid_type: str = 'ap3d'  # ADD THIS PARAMETER (default='ap3d')
    ):
        """
        Args:
            artifacts_dir: Directory containing FAISS index and metadata
            reid_model_path: Path to pretrained Video ReID model (optional)
            device: 'cuda' or 'cpu'
            reid_type: ReID model type - 'temporal', 'ap3d', 'transreid', 'fastreid'  # ADD THIS
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.reid_type = reid_type  # ADD THIS
        
        # Load CLIP-based FAISS index
        print("[Hybrid] Loading CLIP index...")
        self.clip_index = load_index(self.artifacts_dir / "faiss.index")
        
        # Load metadata
        with open(self.artifacts_dir / "meta_test.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        
        # Load CLIP features
        self.clip_features = np.load(self.artifacts_dir / "vecs_test.npy")
        
        # Initialize Video ReID extractor based on type  # ADD THIS SECTION
        print(f"[Hybrid] Initializing {reid_type.upper()} ReID model...")
        
        if reid_type == 'temporal':
            # Original temporal attention model
            self.reid_extractor = VideoReIDExtractor(
                model_path=reid_model_path,
                device=device
            )
        else:
            # SOTA models (ap3d, transreid, fastreid)
            self.reid_extractor = SOTAReIDExtractor(
                model_type=reid_type,
                model_path=reid_model_path,
                device=device
            )
        # END OF ADDED SECTION
        
        # Cache for ReID features
        self._reid_cache = {}
        
        print("[Hybrid] Ready!")
    
    def _get_reid_feature(self, track_idx: int, mevid_root: Path) -> np.ndarray:
        """Get or compute ReID feature for a tracklet"""
        if track_idx in self._reid_cache:
            return self._reid_cache[track_idx]
        
        # Resolve frame paths
        meta = self.metadata[track_idx]
        frame_paths = [self._resolve_path(f, mevid_root) for f in meta['frames']]
        
        # Extract ReID feature
        feat = self.reid_extractor.extract_tracklet_feature(frame_paths, max_frames=16)
        self._reid_cache[track_idx] = feat
        
        return feat
    
    def _resolve_path(self, rel_path: str, mevid_root: Path) -> Path:
        """Resolve relative path to absolute path"""
        rel_path = rel_path.replace("\\", "/")
        p1 = mevid_root / rel_path
        if p1.exists():
            return p1
        
        # Try alternative structure
        parts = rel_path.split("/")
        if parts and parts[0].startswith("bbox_"):
            split_dir = parts[0]
            fname = parts[-1]
            if len(fname) >= 4:
                pid4 = fname[:4]
                p2 = mevid_root / split_dir / pid4 / fname
                if p2.exists():
                    return p2
        
        return p1
    
    def search(
        self,
        query: str,
        mevid_root: Path,
        topk_clip: int = 50,
        topk_final: int = 10,
        alpha: float = 0.6,
        use_reid_rerank: bool = True,
        diversity_penalty: float = 0.03,
        reid_reference_topk: int = 3,  # NEW: Use top-K CLIP results as references
        reid_weight_decay: float = 0.5  # NEW: Weight decay for lower-ranked references
    ) -> List[SearchResult]:
        """
        Hybrid search: CLIP retrieval + Video ReID re-ranking
        
        Args:
            query: Text query describing the person
            mevid_root: Root directory of MEvid dataset
            topk_clip: Number of candidates from CLIP stage
            topk_final: Final number of results to return
            alpha: Weight for combining scores (alpha*clip + (1-alpha)*reid)
            use_reid_rerank: Whether to use ReID for re-ranking
            diversity_penalty: Penalty for repeated cameras
            reid_reference_topk: Number of top CLIP results to use as ReID references
            reid_weight_decay: Weight decay for lower-ranked references (exponential)
        
        Returns:
            List of SearchResult objects, ranked by combined score
        """
        # Stage 1: CLIP text-to-video retrieval
        print(f"[Stage 1] CLIP search for: '{query}'")
        query_vec = encode_text([query])  # (1, 512)
        
        clip_sims, clip_ids = self.clip_index.search(query_vec, topk_clip)
        clip_ids = clip_ids[0].tolist()
        clip_sims = clip_sims[0].tolist()
        
        # Normalize CLIP scores to [0, 1]
        clip_min, clip_max = min(clip_sims), max(clip_sims)
        clip_range = clip_max - clip_min if clip_max > clip_min else 1.0
        clip_sims_norm = [(s - clip_min) / clip_range for s in clip_sims]
        
        if not use_reid_rerank:
            # Return CLIP results only
            results = []
            for rank, (tid, score) in enumerate(zip(clip_ids, clip_sims_norm), 1):
                meta = self.metadata[tid]
                results.append(SearchResult(
                    track_id=tid,
                    person_id=meta['pid'],
                    outfit=meta['outfit'],
                    camera_id=meta['camid'],
                    clip_score=score,
                    reid_score=0.0,
                    combined_score=score,
                    frames=meta['frames'],
                    rank=rank
                ))
            return results[:topk_final]
        
        # Stage 2: Video ReID re-ranking with multiple references
        print(f"[Stage 2] Video ReID re-ranking with top-{reid_reference_topk} references")
        
        # Extract ReID features for top-K CLIP results as references
        reference_feats = []
        reference_weights = []
        
        for i in range(min(reid_reference_topk, len(clip_ids))):
            ref_tid = clip_ids[i]
            ref_feat = self._get_reid_feature(ref_tid, mevid_root)
            
            # Weight decreases exponentially: 1.0, 0.5, 0.25, ...
            weight = reid_weight_decay ** i
            
            reference_feats.append(ref_feat)
            reference_weights.append(weight)
        
        # Normalize weights to sum to 1
        weight_sum = sum(reference_weights)
        reference_weights = [w / weight_sum for w in reference_weights]
        
        print(f"[Stage 2] Using {len(reference_feats)} references with weights: {[f'{w:.3f}' for w in reference_weights]}")
        
        # Compute ReID similarities for all candidates using weighted average
        reid_scores = []
        for tid in clip_ids:
            candidate_feat = self._get_reid_feature(tid, mevid_root)
            
            # Compute weighted similarity with all references
            weighted_sim = 0.0
            for ref_feat, weight in zip(reference_feats, reference_weights):
                sim = np.dot(ref_feat, candidate_feat)
                weighted_sim += weight * sim
            
            reid_scores.append(weighted_sim)
        
        # Normalize ReID scores
        reid_min, reid_max = min(reid_scores), max(reid_scores)
        reid_range = reid_max - reid_min if reid_max > reid_min else 1.0
        reid_scores_norm = [(s - reid_min) / reid_range for s in reid_scores]
        
        # Combine scores
        combined_scores = [
            alpha * clip_s + (1 - alpha) * reid_s
            for clip_s, reid_s in zip(clip_sims_norm, reid_scores_norm)
        ]
        
        # Build results
        results = []
        for tid, clip_s, reid_s, comb_s in zip(clip_ids, clip_sims_norm, reid_scores_norm, combined_scores):
            meta = self.metadata[tid]
            results.append(SearchResult(
                track_id=tid,
                person_id=meta['pid'],
                outfit=meta['outfit'],
                camera_id=meta['camid'],
                clip_score=clip_s,
                reid_score=reid_s,
                combined_score=comb_s,
                frames=meta['frames'],
                rank=0  # Will be set after sorting
            ))
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Apply diversity penalty (penalize repeated cameras)
        if diversity_penalty > 0:
            seen_cameras = set()
            for r in results:
                if r.camera_id in seen_cameras:
                    r.combined_score -= diversity_penalty
                seen_cameras.add(r.camera_id)
            
            # Re-sort after diversity penalty
            results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Assign final ranks
        for rank, r in enumerate(results, 1):
            r.rank = rank
        
        return results[:topk_final]
    
    def person_reidentification(
        self,
        reference_track_id: int,
        mevid_root: Path,
        topk: int = 20,
        exclude_same_camera: bool = True
    ) -> List[SearchResult]:
        """
        Pure video-to-video re-identification (no text query)
        Given a reference tracklet, find the same person in other cameras
        
        Args:
            reference_track_id: Track ID to use as query
            mevid_root: Root of MEvid dataset
            topk: Number of results
            exclude_same_camera: Whether to exclude results from same camera
        
        Returns:
            List of SearchResult objects
        """
        print(f"[ReID] Finding matches for track {reference_track_id}")
        
        # Extract query feature
        query_feat = self._get_reid_feature(reference_track_id, mevid_root)
        ref_meta = self.metadata[reference_track_id]
        ref_camera = ref_meta['camid']
        
        # Compare against all tracklets
        results = []
        for tid in range(len(self.metadata)):
            if tid == reference_track_id:
                continue
            
            meta = self.metadata[tid]
            
            # Skip same camera if requested
            if exclude_same_camera and meta['camid'] == ref_camera:
                continue
            
            # Compute similarity
            candidate_feat = self._get_reid_feature(tid, mevid_root)
            sim = np.dot(query_feat, candidate_feat)
            
            results.append(SearchResult(
                track_id=tid,
                person_id=meta['pid'],
                outfit=meta['outfit'],
                camera_id=meta['camid'],
                clip_score=0.0,
                reid_score=sim,
                combined_score=sim,
                frames=meta['frames'],
                rank=0
            ))
        
        # Sort by ReID score
        results.sort(key=lambda x: x.reid_score, reverse=True)
        
        # Assign ranks
        for rank, r in enumerate(results, 1):
            r.rank = rank
        
        return results[:topk]
    
    def precompute_reid_features(self, mevid_root: Path, save_path: Path = None):
        """
        Precompute all ReID features to speed up search
        
        Args:
            mevid_root: Root directory of MEvid dataset
            save_path: Path to save features (optional)
        """
        print(f"[Precompute] Extracting ReID features for {len(self.metadata)} tracklets...")
        
        all_tracklets = []
        for meta in self.metadata:
            frame_paths = [self._resolve_path(f, mevid_root) for f in meta['frames']]
            all_tracklets.append(frame_paths)
        
        # Batch extraction
        features = self.reid_extractor.extract_batch_features(
            all_tracklets,
            max_frames=16,
            batch_size=8
        )
        
        # Cache all features
        for i, feat in enumerate(features):
            self._reid_cache[i] = feat
        
        # Save if requested
        if save_path:
            np.save(save_path, features)
            print(f"[Precompute] Saved ReID features to {save_path}")
        
        return features