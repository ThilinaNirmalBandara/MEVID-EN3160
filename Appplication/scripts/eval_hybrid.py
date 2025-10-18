import sys, os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import argparse

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
        if labels.sum() > 0:  # Only compute if there are positive samples
            ap = average_precision_score(labels, sims)
            aps.append(ap)
    
    # Normalize CMC scores
    cmc = {k: v / num_queries for k, v in cmc_scores.items()}
    mAP = np.mean(aps) if aps else 0.0
    
    return cmc, mAP


def evaluate_reid_only(engine, mevid_root, query_indices, meta):
    """Evaluate pure video ReID performance"""
    print("\n" + "="*70)
    print("EVALUATING: Pure Video ReID (Video-to-Video)")
    print("="*70)
    
    # Precompute all ReID features
    print("Precomputing ReID features...")
    reid_features = engine.precompute_reid_features(mevid_root)
    
    # Build query and gallery sets
    all_pids = np.array([m['pid'] for m in meta])
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
    
    # Compute metrics
    cmc, mAP = compute_cmc_map(query_indices, gallery_pids, query_pids, similarity_matrix)
    
    print("\nResults:")
    for k in sorted(cmc.keys()):
        print(f"  Rank-{k:2d}: {cmc[k]*100:.2f}%")
    print(f"  mAP:      {mAP*100:.2f}%")
    
    return cmc, mAP


def evaluate_text_search(engine, mevid_root, test_queries, meta, use_reid=True):
    """
    Evaluate text-based person search
    
    Args:
        engine: HybridSearchEngine instance
        mevid_root: Path to MEvid dataset
        test_queries: List of (query_text, target_pid) tuples
        meta: Metadata list
        use_reid: Whether to use ReID reranking
    """
    mode = "Hybrid CLIP + ReID" if use_reid else "CLIP Only"
    print("\n" + "="*70)
    print(f"EVALUATING: Text-Based Search ({mode})")
    print("="*70)
    
    recalls = {1: 0, 5: 0, 10: 0, 20: 0}
    total_queries = len(test_queries)
    
    for query_text, target_pid in tqdm(test_queries, desc="Evaluating"):
        results = engine.search(
            query=query_text,
            mevid_root=mevid_root,
            topk_clip=100,
            topk_final=20,
            alpha=0.6,
            use_reid_rerank=use_reid,
            diversity_penalty=0.0
        )
        
        # Check if target person appears in top-k
        for k in recalls.keys():
            top_k_pids = [r.person_id for r in results[:k]]
            if target_pid in top_k_pids:
                recalls[k] += 1
    
    print("\nResults:")
    for k in sorted(recalls.keys()):
        print(f"  Recall@{k:2d}: {recalls[k]/total_queries*100:.2f}%")
    
    return {k: v/total_queries for k, v in recalls.items()}


def create_test_queries():
    """
    Create test queries for text-based evaluation
    These should be manually created based on MEvid person attributes
    """
    # Example queries - you should expand this based on actual MEvid annotations
    queries = [
        ("man in black jacket", 201),
        ("woman with red shirt", 205),
        ("person wearing blue jeans", 210),
        ("man with backpack", 215),
        ("woman in white dress", 220),
        # Add more queries based on your dataset
    ]
    return queries


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid CLIP + ReID System")
    parser.add_argument("--eval_reid", action="store_true", help="Evaluate pure ReID")
    parser.add_argument("--eval_text", action="store_true", help="Evaluate text search")
    parser.add_argument("--reid_model", type=str, help="Path to trained ReID model")
    parser.add_argument("--eval_both", action="store_true", help="Compare CLIP vs Hybrid")
    
    args = parser.parse_args()
    
    # Load metadata and query indices
    with open(ARTIFACTS / "meta_test.pkl", "rb") as f:
        import pickle
        meta = pickle.load(f)
    
    query_indices = json.loads((ARTIFACTS / "query_idx.json").read_text())
    
    # Initialize engine
    print("Initializing Hybrid Search Engine...")
    engine = HybridSearchEngine(
        artifacts_dir=ARTIFACTS,
        reid_model_path=args.reid_model
    )
    
    results = {}
    
    # Evaluate pure video ReID
    if args.eval_reid:
        cmc, mAP = evaluate_reid_only(engine, MEVID_ROOT, query_indices, meta)
        results['reid'] = {'cmc': cmc, 'mAP': mAP}
    
    # Evaluate text-based search
    if args.eval_text:
        test_queries = create_test_queries()
        
        if args.eval_both:
            # Compare CLIP-only vs Hybrid
            print("\nComparing CLIP-only vs Hybrid...")
            
            clip_recalls = evaluate_text_search(engine, MEVID_ROOT, test_queries, meta, use_reid=False)
            hybrid_recalls = evaluate_text_search(engine, MEVID_ROOT, test_queries, meta, use_reid=True)
            
            print("\n" + "="*70)
            print("COMPARISON: CLIP vs Hybrid")
            print("="*70)
            print(f"{'Metric':<15} {'CLIP Only':<15} {'Hybrid':<15} {'Improvement':<15}")
            print("-"*70)
            
            for k in sorted(clip_recalls.keys()):
                clip_val = clip_recalls[k] * 100
                hybrid_val = hybrid_recalls[k] * 100
                improvement = hybrid_val - clip_val
                print(f"Recall@{k:<8} {clip_val:>6.2f}% {hybrid_val:>14.2f}% {improvement:>+13.2f}%")
            
            results['text_clip'] = clip_recalls
            results['text_hybrid'] = hybrid_recalls
        else:
            recalls = evaluate_text_search(engine, MEVID_ROOT, test_queries, meta, use_reid=True)
            results['text'] = recalls
    
    # Save results
    results_path = ARTIFACTS / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")


if __name__ == "__main__":
    main()