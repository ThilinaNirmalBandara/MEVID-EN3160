import sys, os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, os
from pathlib import Path
import json

# ensure we can import config + package when running as: python scripts/parse_mevid.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from config import ARTIFACTS, MEVID_ROOT, THUMBS_DIR
from mevid_textsearch.hybrid_search import HybridSearchEngine
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_result_grid(results, thumbs_dir, output_path, query_text=None):
    """
    Create a visual grid of search results
    
    Args:
        results: List of SearchResult objects
        thumbs_dir: Directory containing thumbnail GIFs
        output_path: Where to save the output image
        query_text: Optional query text to display at top
    """
    thumbs_dir = Path(thumbs_dir)
    
    # Load thumbnails (convert GIFs to static images)
    images = []
    for r in results:
        thumb_path = thumbs_dir / f"track_{r.track_id}.gif"
        if thumb_path.exists():
            try:
                img = Image.open(thumb_path).convert('RGB')
                images.append((img, r))
            except:
                continue
    
    if not images:
        print("No thumbnails found!")
        return
    
    # Grid layout
    cols = min(5, len(images))
    rows = (len(images) + cols - 1) // cols
    
    # Thumbnail size
    thumb_w, thumb_h = 320, 240
    padding = 20
    text_h = 80
    header_h = 60 if query_text else 0
    
    # Create canvas
    grid_w = cols * (thumb_w + padding) + padding
    grid_h = header_h + rows * (thumb_h + text_h + padding) + padding
    canvas = Image.new('RGB', (grid_w, grid_h), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to use a nice font
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Draw query text
    if query_text:
        query_display = f'Query: "{query_text}"'
        draw.text((padding, 20), query_display, fill='black', font=title_font)
    
    # Place thumbnails
    for idx, (img, result) in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        x = padding + col * (thumb_w + padding)
        y = header_h + padding + row * (thumb_h + text_h + padding)
        
        # Resize thumbnail to fit
        img_resized = img.resize((thumb_w, thumb_h), Image.LANCZOS)
        canvas.paste(img_resized, (x, y))
        
        # Draw info box
        info_y = y + thumb_h + 5
        
        # Rank and scores
        rank_text = f"#{result.rank}"
        draw.text((x, info_y), rank_text, fill='red', font=label_font)
        
        # Person info
        info_text = f"PID:{result.person_id} | Cam:{result.camera_id}"
        draw.text((x, info_y + 18), info_text, fill='black', font=label_font)
        
        # Scores
        score_text = f"CLIP:{result.clip_score:.3f} | ReID:{result.reid_score:.3f}"
        draw.text((x, info_y + 36), score_text, fill='blue', font=label_font)
        
        combined_text = f"Combined: {result.combined_score:.3f}"
        draw.text((x, info_y + 54), combined_text, fill='green', font=label_font)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"✓ Saved result grid to {output_path}")
    
    return canvas


def create_comparison_view(query_text, results_clip, results_hybrid, thumbs_dir, output_path):
    """
    Create side-by-side comparison of CLIP vs Hybrid results
    """
    thumbs_dir = Path(thumbs_dir)
    
    # Get top 5 from each
    clip_top5 = results_clip[:5]
    hybrid_top5 = results_hybrid[:5]
    
    # Load thumbnails
    def load_thumbs(results):
        imgs = []
        for r in results:
            thumb_path = thumbs_dir / f"track_{r.track_id}.gif"
            if thumb_path.exists():
                try:
                    img = Image.open(thumb_path).convert('RGB')
                    imgs.append((img, r))
                except:
                    imgs.append((None, r))
            else:
                imgs.append((None, r))
        return imgs
    
    clip_imgs = load_thumbs(clip_top5)
    hybrid_imgs = load_thumbs(hybrid_top5)
    
    # Layout
    thumb_w, thumb_h = 256, 192
    padding = 15
    header_h = 80
    text_h = 70
    
    rows = 5
    cols = 2  # CLIP | Hybrid
    
    canvas_w = 2 * (thumb_w + padding) + 3 * padding
    canvas_h = header_h + rows * (thumb_h + text_h + padding) + padding
    
    canvas = Image.new('RGB', (canvas_w, canvas_h), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        title_font = header_font = label_font = ImageFont.load_default()
    
    # Title
    draw.text((padding, 15), f'Query: "{query_text}"', fill='black', font=title_font)
    
    # Column headers
    col1_x = padding
    col2_x = padding * 2 + thumb_w
    
    draw.text((col1_x + 50, header_h - 35), "CLIP Only", fill='blue', font=header_font)
    draw.text((col2_x + 50, header_h - 35), "Hybrid (CLIP + ReID)", fill='green', font=header_font)
    
    # Draw results
    for i in range(rows):
        y = header_h + i * (thumb_h + text_h + padding)
        
        # CLIP result
        if i < len(clip_imgs):
            img, result = clip_imgs[i]
            if img:
                img_resized = img.resize((thumb_w, thumb_h), Image.LANCZOS)
                canvas.paste(img_resized, (col1_x, y))
            
            info_y = y + thumb_h + 5
            draw.text((col1_x, info_y), f"#{result.rank} | PID:{result.person_id}", fill='blue', font=label_font)
            draw.text((col1_x, info_y + 18), f"Score: {result.clip_score:.3f}", fill='black', font=label_font)
            draw.text((col1_x, info_y + 36), f"Cam:{result.camera_id}", fill='gray', font=label_font)
        
        # Hybrid result
        if i < len(hybrid_imgs):
            img, result = hybrid_imgs[i]
            if img:
                img_resized = img.resize((thumb_w, thumb_h), Image.LANCZOS)
                canvas.paste(img_resized, (col2_x, y))
            
            info_y = y + thumb_h + 5
            draw.text((col2_x, info_y), f"#{result.rank} | PID:{result.person_id}", fill='green', font=label_font)
            draw.text((col2_x, info_y + 18), f"CLIP: {result.clip_score:.3f} | ReID: {result.reid_score:.3f}", fill='black', font=label_font)
            draw.text((col2_x, info_y + 36), f"Combined: {result.combined_score:.3f} | Cam:{result.camera_id}", fill='gray', font=label_font)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"✓ Saved comparison to {output_path}")
    
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visual Demo for Hybrid Search")
    parser.add_argument("--query", "-q", required=True, help="Text query")
    parser.add_argument("--topk", type=int, default=10, help="Number of results")
    parser.add_argument("--compare", action="store_true", help="Compare CLIP vs Hybrid")
    parser.add_argument("--output", "-o", default="demo_results.jpg", help="Output image path")
    parser.add_argument("--reid_model", type=str, help="Path to trained ReID model")
    
    args = parser.parse_args()
    
    print("="*70)
    print("VISUAL DEMO - Hybrid CLIP + Video ReID")
    print("="*70)
    
    # Initialize engine
    print("\nInitializing search engine...")
    engine = HybridSearchEngine(
        artifacts_dir=ARTIFACTS,
        reid_model_path=args.reid_model
    )
    
    if args.compare:
        # Get both CLIP and Hybrid results
        print(f"\nSearching with CLIP only...")
        results_clip = engine.search(
            query=args.query,
            mevid_root=MEVID_ROOT,
            topk_clip=50,
            topk_final=args.topk,
            use_reid_rerank=False
        )
        
        print(f"\nSearching with Hybrid (CLIP + ReID)...")
        results_hybrid = engine.search(
            query=args.query,
            mevid_root=MEVID_ROOT,
            topk_clip=50,
            topk_final=args.topk,
            use_reid_rerank=True,
            alpha=0.6
        )
        
        print(f"\nCreating comparison visualization...")
        create_comparison_view(
            query_text=args.query,
            results_clip=results_clip,
            results_hybrid=results_hybrid,
            thumbs_dir=THUMBS_DIR,
            output_path=args.output
        )
    else:
        # Single search
        print(f"\nSearching for: '{args.query}'")
        results = engine.search(
            query=args.query,
            mevid_root=MEVID_ROOT,
            topk_clip=50,
            topk_final=args.topk,
            use_reid_rerank=True,
            alpha=0.6
        )
        
        print(f"\nCreating result grid...")
        create_result_grid(
            results=results,
            thumbs_dir=THUMBS_DIR,
            output_path=args.output,
            query_text=args.query
        )
    
    print("\n" + "="*70)
    print(f"✓ Demo complete! Open {args.output} to view results.")
    print("="*70)


if __name__ == "__main__":
    main()