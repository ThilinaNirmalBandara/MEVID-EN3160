import sys, os
from pathlib import Path
import json

# ensure we can import config + package when running as: python scripts/parse_mevid.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config import ARTIFACTS, MEVID_ROOT
from mevid_textsearch.hybrid_search import HybridSearchEngine

def format_result(result, show_frames=True):
    """Format a search result for display"""
    print(f"\n{'='*70}")
    print(f"Rank #{result.rank}")
    print(f"  Track ID: {result.track_id}")
    print(f"  Person ID: {result.person_id}")
    print(f"  Outfit: {result.outfit}")
    print(f"  Camera: {result.camera_id}")
    print(f"  Scores:")
    print(f"    - CLIP:     {result.clip_score:.4f}")
    print(f"    - ReID:     {result.reid_score:.4f}")
    print(f"    - Combined: {result.combined_score:.4f}")
    
    if show_frames:
        print(f"  Sample frames: {result.frames[:3]}")
    
    thumb_path = ARTIFACTS / "thumbs" / f"track_{result.track_id}.gif"
    if thumb_path.exists():
        print(f"  Thumbnail: {thumb_path}")

def main():
    parser = argparse.ArgumentParser(description="Hybrid CLIP + Video ReID Search")
    parser.add_argument("--query", "-q", type=str, help="Text query for person search")
    parser.add_argument("--track_id", "-t", type=int, help="Track ID for video-to-video ReID")
    parser.add_argument("--topk", "-k", type=int, default=10, help="Number of results")
    parser.add_argument("--topk_clip", type=int, default=50, help="CLIP candidates for reranking")
    parser.add_argument("--alpha", type=float, default=0.6, help="CLIP weight (0-1)")
    parser.add_argument("--no_reid", action="store_true", help="Use CLIP only (no ReID)")
    parser.add_argument("--reid_model", type=str, help="Path to pretrained ReID model")
    parser.add_argument("--precompute", action="store_true", help="Precompute all ReID features")
    parser.add_argument("--no_diversity", action="store_true", help="Disable camera diversity")
    
    args = parser.parse_args()
    
    # Initialize hybrid search engine
    print("\n" + "="*70)
    print("HYBRID CLIP + VIDEO REID SEARCH ENGINE")
    print("="*70)
    
    engine = HybridSearchEngine(
        artifacts_dir=ARTIFACTS,
        reid_model_path=args.reid_model
    )
    
    # Precompute features if requested
    if args.precompute:
        print("\nPrecomputing ReID features...")
        features = engine.precompute_reid_features(
            mevid_root=MEVID_ROOT,
            save_path=ARTIFACTS / "reid_features.npy"
        )
        print(f"Done! Computed {features.shape[0]} feature vectors of dim {features.shape[1]}")
        return
    
    # Text-based search
    if args.query:
        print(f"\nSearch Query: '{args.query}'")
        print(f"Mode: {'CLIP only' if args.no_reid else 'Hybrid CLIP + ReID'}")
        
        results = engine.search(
            query=args.query,
            mevid_root=MEVID_ROOT,
            topk_clip=args.topk_clip,
            topk_final=args.topk,
            alpha=args.alpha,
            use_reid_rerank=not args.no_reid,
            diversity_penalty=0.0 if args.no_diversity else 0.03
        )
        
        print(f"\nTop {len(results)} Results:")
        for result in results:
            format_result(result)
    
    # Video-to-video ReID
    elif args.track_id is not None:
        print(f"\nVideo Re-Identification")
        print(f"Query Track ID: {args.track_id}")
        
        results = engine.person_reidentification(
            reference_track_id=args.track_id,
            mevid_root=MEVID_ROOT,
            topk=args.topk,
            exclude_same_camera=True
        )
        
        print(f"\nTop {len(results)} Matches:")
        for result in results:
            format_result(result)
    
    else:
        parser.print_help()
        print("\nError: Please provide either --query or --track_id")
        sys.exit(1)
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()