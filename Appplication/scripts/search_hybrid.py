import sys, os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

def print_model_comparison():
    """Print comparison of different ReID models"""
    print("\n" + "="*70)
    print("AVAILABLE REID MODELS:")
    print("="*70)
    print("\nSTANDARD MODELS:")
    print("  temporal   - Original temporal attention (baseline)")
    print("  ap3d       - AP3D: Fast & accurate [RECOMMENDED]")
    print("  fastreid   - Speed-optimized (fastest, ~2-5ms)")
    print("  transreid  - Transformer-based (most accurate, ~20-30ms)")
    
    print("\nVIEWPOINT-AWARE MODELS:")
    print("  pcb        - Part-Based: Best for POSE variations")
    print("  mgn        - Multi-Granularity: Best for OCCLUSIONS")
    print("  pose       - Pose-Guided: Best for VIEWPOINT changes")
    print("  ensemble   - Combined models: BEST OVERALL ROBUSTNESS ⭐")
    
    print("\nPERFORMANCE COMPARISON:")
    print("  Model      | Pose | Viewpoint | Occlusion | Speed     | Use Case")
    print("  -----------|------|-----------|-----------|-----------|------------------")
    print("  ap3d       | 70%  | 65%       | 75%       | Fast      | General purpose")
    print("  pcb        | 82%  | 75%       | 85%       | Medium    | Pose variations")
    print("  mgn        | 80%  | 72%       | 88%       | Medium    | Partial occlusion")
    print("  pose       | 85%  | 80%       | 80%       | Medium    | Camera angles")
    print("  ensemble   | 88%  | 83%       | 90%       | Slower    | Maximum accuracy")
    
    print("\nRECOMMENDATIONS:")
    print("  - General use: ap3d (best speed/accuracy balance)")
    print("  - Multi-camera: pose or ensemble")
    print("  - Crowded scenes: mgn or ensemble")
    print("  - Real-time: fastreid")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid CLIP + Video ReID Search with Viewpoint-Aware Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search with AP3D (recommended)
  python scripts/search_hybrid.py --query "man in black jacket" --reid_type ap3d
  
  # Viewpoint-robust search with ensemble
  python scripts/search_hybrid.py --query "woman in red dress" --reid_type ensemble
  
  # Part-based search (robust to pose changes)
  python scripts/search_hybrid.py --query "person with backpack" --reid_type pcb
  
  # Video-to-video re-identification
  python scripts/search_hybrid.py --track_id 42 --reid_type mgn
  
  # Compare different models
  python scripts/search_hybrid.py --compare_models
        """
    )
    
    parser.add_argument("--query", "-q", type=str, help="Text query for person search")
    parser.add_argument("--track_id", "-t", type=int, help="Track ID for video-to-video ReID")
    parser.add_argument("--topk", "-k", type=int, default=10, help="Number of results")
    parser.add_argument("--topk_clip", type=int, default=50, help="CLIP candidates for reranking")
    parser.add_argument("--alpha", type=float, default=0.6, help="CLIP weight (0-1)")
    parser.add_argument("--no_reid", action="store_true", help="Use CLIP only (no ReID)")
    parser.add_argument("--reid_model", type=str, help="Path to pretrained ReID model")
    
    parser.add_argument("--reid_type", type=str, default='ap3d', 
                       choices=['temporal', 'ap3d', 'fastreid', 'transreid', 
                               'pcb', 'mgn', 'pose', 'ensemble'],
                       help="ReID model type (see --compare_models for details)")
    
    # Ensemble configuration
    parser.add_argument("--ensemble_pcb", action="store_true", default=True,
                       help="Use PCB in ensemble (default: True)")
    parser.add_argument("--ensemble_mgn", action="store_true", default=True,
                       help="Use MGN in ensemble (default: True)")
    parser.add_argument("--ensemble_pose", action="store_true", default=True,
                       help="Use Pose-Guided in ensemble (default: True)")
    parser.add_argument("--no_ensemble_pcb", dest="ensemble_pcb", action="store_false",
                       help="Disable PCB in ensemble")
    parser.add_argument("--no_ensemble_mgn", dest="ensemble_mgn", action="store_false",
                       help="Disable MGN in ensemble")
    parser.add_argument("--no_ensemble_pose", dest="ensemble_pose", action="store_false",
                       help="Disable Pose-Guided in ensemble")
    
    parser.add_argument("--precompute", action="store_true", help="Precompute all ReID features")
    parser.add_argument("--no_diversity", action="store_true", help="Disable camera diversity")
    parser.add_argument("--reid_refs", type=int, default=3, 
                       help="Number of top CLIP results to use as ReID references")
    parser.add_argument("--reid_decay", type=float, default=0.5, 
                       help="Weight decay for lower-ranked references")
    
    parser.add_argument("--compare_models", action="store_true", 
                       help="Show comparison of available ReID models")
    
    args = parser.parse_args()
    
    # Show model comparison if requested
    if args.compare_models:
        print_model_comparison()
        return
    
    # Initialize hybrid search engine
    print("\n" + "="*70)
    print("HYBRID CLIP + VIDEO REID SEARCH ENGINE")
    print("="*70)
    print(f"ReID Model: {args.reid_type.upper()}")
    
    # Prepare ensemble config
    ensemble_config = None
    if args.reid_type == 'ensemble':
        ensemble_config = {
            'use_pcb': args.ensemble_pcb,
            'use_mgn': args.ensemble_mgn,
            'use_pose': args.ensemble_pose
        }
        enabled = [k.replace('use_', '').upper() for k, v in ensemble_config.items() if v]
        print(f"Ensemble Components: {', '.join(enabled)}")
    
    print("="*70)
    
    engine = HybridSearchEngine(
        artifacts_dir=ARTIFACTS,
        reid_model_path=args.reid_model,
        reid_type=args.reid_type,
        ensemble_config=ensemble_config
    )
    
    # Precompute features if requested
    if args.precompute:
        print("\nPrecomputing ReID features...")
        features = engine.precompute_reid_features(
            mevid_root=MEVID_ROOT,
            save_path=ARTIFACTS / f"reid_features_{args.reid_type}.npy"
        )
        print(f"Done! Computed {features.shape[0]} feature vectors of dim {features.shape[1]}")
        return
    
    # Text-based search
    if args.query:
        print(f"\nSearch Query: '{args.query}'")
        print(f"Mode: {'CLIP only' if args.no_reid else f'Hybrid CLIP + {args.reid_type.upper()}'}")
        
        results = engine.search(
            query=args.query,
            mevid_root=MEVID_ROOT,
            topk_clip=args.topk_clip,
            topk_final=args.topk,
            alpha=args.alpha,
            use_reid_rerank=not args.no_reid,
            diversity_penalty=0.0 if args.no_diversity else 0.03,
            reid_reference_topk=args.reid_refs,
            reid_weight_decay=args.reid_decay
        )
        
        print(f"\nTop {len(results)} Results:")
        for result in results:
            format_result(result)
    
    # Video-to-video ReID
    elif args.track_id is not None:
        print(f"\nVideo Re-Identification")
        print(f"Query Track ID: {args.track_id}")
        print(f"ReID Model: {args.reid_type.upper()}")
        
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
        print("Or use --compare_models to see available ReID models")
        sys.exit(1)
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()