import sys, os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import argparse
from collections import defaultdict

from config import ARTIFACTS, MEVID_ROOT
from mevid_textsearch.hybrid_search import HybridSearchEngine

def compute_cmc_map(query_indices, gallery_pids, query_pids, similarity_matrix, ranks=[1, 5, 10, 20]):
    """
    Compute CMC (Cumulative Matching Characteristics) and mAP
    
    Args:
        query_indices: List of query indices
        gallery_pids: Array of person IDs for gallery
        query_pids: Array of person IDs for queries
        similarity_matrix: (num_queries, num_gallery) similarity scores
        ranks: Rank positions to compute CMC
    
    Returns:
        cmc: Dict of Rank-k accuracies
        mAP: Mean Average Precision
    """
    num_queries = len(query_indices)
    cmc_scores = {k: 0 for k in ranks}
    aps = []
    
    for i, qidx in enumerate(query_indices):
        qpid = query_pids[i]
        
        # Get similarities for this query
        sims = similarity_matrix[i]
        
        # Create labels (1 for same person, 0 for different)
        labels = (gallery_pids == qpid).astype(int)
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(-sims)
        sorted_labels = labels[sorted_indices]
        
        # CMC: Check if correct match appears in top-k
        for k in ranks:
            if sorted_labels[:k].sum() > 0:
                cmc_scores[k] += 1
        
        # Average Precision
        if labels.sum() > 0:
            ap = average_precision_score(labels, sims)
            aps.append(ap)
    
    # Normalize CMC scores
    cmc = {k: v / num_queries for k, v in cmc_scores.items()}
    mAP = np.mean(aps) if aps else 0.0
    
    return cmc, mAP


def evaluate_reid_model(
    engine, 
    mevid_root, 
    query_indices, 
    meta, 
    model_name,
    filter_by_camera=False
):
    """Evaluate a single ReID model"""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")
    
    # Precompute all ReID features
    print("Precomputing ReID features...")
    reid_features = engine.precompute_reid_features(mevid_root)
    
    # Build query and gallery sets
    all_pids = np.array([m['pid'] for m in meta])
    all_cams = np.array([m['camid'] for m in meta])
    num_total = len(meta)
    
    # Gallery: all except queries
    gallery_mask = np.ones(num_total, dtype=bool)
    gallery_mask[query_indices] = False
    gallery_indices = np.where(gallery_mask)[0]
    
    gallery_pids = all_pids[gallery_indices]
    query_pids = all_pids[query_indices]
    
    # Compute similarity matrix
    print("Computing similarities...")
    query_features = reid_features[query_indices]
    gallery_features = reid_features[gallery_indices]
    
    # Cosine similarity
    similarity_matrix = query_features @ gallery_features.T
    
    # Overall metrics
    cmc, mAP = compute_cmc_map(query_indices, gallery_pids, query_pids, similarity_matrix)
    
    print(f"\nOverall Performance:")
    for k in sorted(cmc.keys()):
        print(f"  Rank-{k:2d}: {cmc[k]*100:.2f}%")
    print(f"  mAP:      {mAP*100:.2f}%")
    
    # Per-camera analysis (viewpoint robustness)
    if filter_by_camera:
        print(f"\nPer-Camera Analysis (Viewpoint Robustness):")
        query_cams = all_cams[query_indices]
        gallery_cams = all_cams[gallery_indices]
        
        cam_results = defaultdict(lambda: {'cmc': {k: [] for k in [1, 5, 10]}, 'aps': []})
        
        for i, (qidx, qpid, qcam) in enumerate(zip(query_indices, query_pids, query_cams)):
            # For this query, only consider gallery from different cameras
            diff_cam_mask = gallery_cams != qcam
            
            if not diff_cam_mask.any():
                continue
            
            # Filter similarity scores
            filtered_sims = similarity_matrix[i][diff_cam_mask]
            filtered_pids = gallery_pids[diff_cam_mask]
            
            # Labels
            labels = (filtered_pids == qpid).astype(int)
            
            # Sort by similarity
            sorted_indices = np.argsort(-filtered_sims)
            sorted_labels = labels[sorted_indices]
            
            # CMC for this query
            for k in [1, 5, 10]:
                if sorted_labels[:k].sum() > 0:
                    cam_results[qcam]['cmc'][k].append(1)
                else:
                    cam_results[qcam]['cmc'][k].append(0)
            
            # AP
            if labels.sum() > 0:
                ap = average_precision_score(labels, filtered_sims)
                cam_results[qcam]['aps'].append(ap)
        
        # Print per-camera results
        print(f"  {'Camera':<10} {'Rank-1':<10} {'Rank-5':<10} {'Rank-10':<10} {'mAP':<10}")
        print(f"  {'-'*50}")
        
        for cam_id in sorted(cam_results.keys()):
            results = cam_results[cam_id]
            r1 = np.mean(results['cmc'][1]) * 100 if results['cmc'][1] else 0
            r5 = np.mean(results['cmc'][5]) * 100 if results['cmc'][5] else 0
            r10 = np.mean(results['cmc'][10]) * 100 if results['cmc'][10] else 0
            map_val = np.mean(results['aps']) * 100 if results['aps'] else 0
            
            print(f"  {cam_id:<10} {r1:>6.2f}%    {r5:>6.2f}%    {r10:>6.2f}%     {map_val:>6.2f}%")
    
    return {'cmc': cmc, 'mAP': mAP}


def compare_all_models(mevid_root, query_indices, meta, models_to_test):
    """Compare multiple ReID models"""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*70)
    
    all_results = {}
    
    for model_type in models_to_test:
        try:
            print(f"\n[{model_type.upper()}] Initializing...")
            
            # Special config for ensemble
            ensemble_config = None
            if model_type == 'ensemble':
                ensemble_config = {
                    'use_pcb': True,
                    'use_mgn': True,
                    'use_pose': True
                }
            
            engine = HybridSearchEngine(
                artifacts_dir=ARTIFACTS,
                reid_type=model_type,
                ensemble_config=ensemble_config
            )
            
            results = evaluate_reid_model(
                engine, 
                mevid_root, 
                query_indices, 
                meta,
                model_type.upper(),
                filter_by_camera=True
            )
            
            all_results[model_type] = results
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_type}: {e}")
            continue
    
    # Print comparison table
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Model':<15} {'Rank-1':<10} {'Rank-5':<10} {'Rank-10':<10} {'mAP':<10}")
    print("-"*70)
    
    for model_type in models_to_test:
        if model_type not in all_results:
            continue
        
        results = all_results[model_type]
        cmc = results['cmc']
        mAP = results['mAP']
        
        print(f"{model_type:<15} {cmc[1]*100:>6.2f}%   {cmc[5]*100:>6.2f}%   "
              f"{cmc[10]*100:>6.2f}%    {mAP*100:>6.2f}%")
    
    # Find best model
    if all_results:
        best_model = max(all_results.items(), key=lambda x: x[1]['mAP'])
        print("\n" + "="*70)
        print(f"✓ BEST MODEL: {best_model[0].upper()} (mAP: {best_model[1]['mAP']*100:.2f}%)")
        print("="*70)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Viewpoint-Aware ReID Models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--reid_type", type=str, default='ap3d',
                       choices=['temporal', 'ap3d', 'fastreid', 'transreid',
                               'pcb', 'mgn', 'pose', 'ensemble'],
                       help="ReID model to evaluate")
    
    parser.add_argument("--compare_all", action="store_true",
                       help="Compare all available models")
    
    parser.add_argument("--models", nargs='+',
                       choices=['temporal', 'ap3d', 'fastreid', 'transreid',
                               'pcb', 'mgn', 'pose', 'ensemble'],
                       help="Specific models to compare")
    
    parser.add_argument("--per_camera", action="store_true",
                       help="Show per-camera results (viewpoint analysis)")
    
    parser.add_argument("--reid_model", type=str,
                       help="Path to pretrained ReID model")
    
    args = parser.parse_args()
    
    # Load metadata and query indices
    print("Loading dataset metadata...")
    with open(ARTIFACTS / "meta_test.pkl", "rb") as f:
        import pickle
        meta = pickle.load(f)
    
    query_indices = json.loads((ARTIFACTS / "query_idx.json").read_text())
    print(f"Loaded {len(meta)} tracklets, {len(query_indices)} queries")
    
    # Compare multiple models or evaluate single model
    if args.compare_all:
        models = ['ap3d', 'fastreid', 'pcb', 'mgn', 'pose', 'ensemble']
        results = compare_all_models(MEVID_ROOT, query_indices, meta, models)
    elif args.models:
        results = compare_all_models(MEVID_ROOT, query_indices, meta, args.models)
    else:
        # Evaluate single model
        ensemble_config = None
        if args.reid_type == 'ensemble':
            ensemble_config = {
                'use_pcb': True,
                'use_mgn': True,
                'use_pose': True
            }
        
        engine = HybridSearchEngine(
            artifacts_dir=ARTIFACTS,
            reid_model_path=args.reid_model,
            reid_type=args.reid_type,
            ensemble_config=ensemble_config
        )
        
        results = evaluate_reid_model(
            engine,
            MEVID_ROOT,
            query_indices,
            meta,
            args.reid_type.upper(),
            filter_by_camera=args.per_camera
        )
    
    # Save results
    results_path = ARTIFACTS / f"evaluation_{args.reid_type}.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")


if __name__ == "__main__":
    main()