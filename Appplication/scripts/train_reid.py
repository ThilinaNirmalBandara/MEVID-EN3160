
import sys, os
from pathlib import Path
import json

# ensure we can import config + package when running as: python scripts/parse_mevid.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

from config import MEVID_ROOT, ARTIFACTS
from mevid_textsearch.temporal_reid import VideoReIDModel

class MevidTrackletDataset(Dataset):
    """Dataset for training Video ReID on MEvid tracklets"""
    
    def __init__(self, tracks_jsonl, mevid_root, split='train', frames_per_track=16):
        self.mevid_root = Path(mevid_root)
        self.frames_per_track = frames_per_track
        
        # Load tracklets
        with open(tracks_jsonl, 'r') as f:
            self.tracklets = [json.loads(line) for line in f]
        
        # Filter by split
        self.tracklets = [t for t in self.tracklets if t['split'] == split]
        
        # Build person ID mapping
        pids = sorted(set(t['pid'] for t in self.tracklets))
        self.pid_to_label = {pid: i for i, pid in enumerate(pids)}
        self.num_pids = len(pids)
        
        print(f"[Dataset] Loaded {len(self.tracklets)} tracklets, {self.num_pids} persons ({split})")
        
        # Transforms
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _resolve_path(self, rel_path):
        """Resolve frame path"""
        rel_path = rel_path.replace("\\", "/")
        p1 = self.mevid_root / rel_path
        if p1.exists():
            return p1
        
        parts = rel_path.split("/")
        if parts and parts[0].startswith("bbox_"):
            split_dir = parts[0]
            fname = parts[-1]
            if len(fname) >= 4:
                pid4 = fname[:4]
                p2 = self.mevid_root / split_dir / pid4 / fname
                if p2.exists():
                    return p2
        return p1
    
    def __len__(self):
        return len(self.tracklets)
    
    def __getitem__(self, idx):
        track = self.tracklets[idx]
        
        # Sample frames evenly
        frame_paths = [self._resolve_path(f) for f in track['frames']]
        if len(frame_paths) > self.frames_per_track:
            indices = np.linspace(0, len(frame_paths)-1, self.frames_per_track, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
        
        # Load frames
        frames = []
        for fp in frame_paths:
            if fp.exists():
                try:
                    img = Image.open(fp).convert('RGB')
                    frames.append(self.transform(img))
                except:
                    continue
        
        # Handle missing frames
        if not frames:
            frames = [torch.zeros(3, 256, 128)]
        
        # Pad if needed
        while len(frames) < self.frames_per_track:
            frames.append(frames[-1].clone())
        
        # Stack: (seq_len, C, H, W)
        frames_tensor = torch.stack(frames[:self.frames_per_track])
        
        # Get label
        label = self.pid_to_label[track['pid']]
        
        return frames_tensor, label, track['tid']


class TripletLoss(nn.Module):
    """Triplet loss with hard mining"""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, features, labels):
        """
        Args:
            features: (batch, feat_dim) normalized features
            labels: (batch,) person IDs
        """
        n = features.size(0)
        
        # Compute pairwise distances
        dist_mat = torch.cdist(features, features, p=2)
        
        # For each sample, find hardest positive and hardest negative
        losses = []
        for i in range(n):
            # Positive samples (same person)
            pos_mask = labels == labels[i]
            pos_mask[i] = False  # exclude self
            
            if pos_mask.sum() == 0:
                continue
            
            # Hardest positive (farthest same-person sample)
            hard_pos_dist = dist_mat[i][pos_mask].max()
            
            # Negative samples (different person)
            neg_mask = labels != labels[i]
            
            if neg_mask.sum() == 0:
                continue
            
            # Hardest negative (closest different-person sample)
            hard_neg_dist = dist_mat[i][neg_mask].min()
            
            # Triplet loss
            loss = torch.clamp(hard_pos_dist - hard_neg_dist + self.margin, min=0.0)
            losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=features.device)


def train_epoch(model, dataloader, optimizer, criterion_ce, criterion_triplet, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_ce_loss = 0
    total_triplet_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for frames, labels, _ in pbar:
        frames = frames.to(device)  # (batch, seq_len, C, H, W)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        feat_norm, logits = model(frames)
        
        # Classification loss
        ce_loss = criterion_ce(logits, labels)
        
        # Triplet loss
        triplet_loss = criterion_triplet(feat_norm, labels)
        
        # Combined loss
        loss = ce_loss + triplet_loss
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_triplet_loss += triplet_loss.item()
        
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'tri': f'{triplet_loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'ce_loss': total_ce_loss / len(dataloader),
        'triplet_loss': total_triplet_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


@torch.no_grad()
def validate(model, dataloader, criterion_ce, criterion_triplet, device):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    total_ce_loss = 0
    total_triplet_loss = 0
    correct = 0
    total = 0
    
    for frames, labels, _ in tqdm(dataloader, desc="Validation"):
        frames = frames.to(device)
        labels = labels.to(device)
        
        feat_norm, logits = model(frames)
        
        ce_loss = criterion_ce(logits, labels)
        triplet_loss = criterion_triplet(feat_norm, labels)
        loss = ce_loss + triplet_loss
        
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_triplet_loss += triplet_loss.item()
        
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'ce_loss': total_ce_loss / len(dataloader),
        'triplet_loss': total_triplet_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def main():
    parser = argparse.ArgumentParser(description="Train Video ReID on MEvid")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--frames", type=int, default=16, help="Frames per tracklet")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--output", type=str, default="reid_model.pth", help="Output model path")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    train_jsonl = ARTIFACTS / "tracks_train.jsonl"
    test_jsonl = ARTIFACTS / "tracks_test.jsonl"
    
    # Check if train tracks exist, if not create them
    if not train_jsonl.exists():
        print("Creating train tracklets...")
        from mevid_textsearch.mevid import build_tracklets, save_jsonl
        train_tracks = build_tracklets(MEVID_ROOT, split="train")
        save_jsonl([t.__dict__ for t in train_tracks], train_jsonl)
    
    train_dataset = MevidTrackletDataset(train_jsonl, MEVID_ROOT, split='train', frames_per_track=args.frames)
    test_dataset = MevidTrackletDataset(test_jsonl, MEVID_ROOT, split='test', frames_per_track=args.frames)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = VideoReIDModel(num_classes=train_dataset.num_pids, dropout=0.5)
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    
    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=args.margin)
    
    # Optimizer with different learning rates for backbone and head
    base_params = list(model.base.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('base')]
    
    optimizer = optim.Adam([
        {'params': base_params, 'lr': args.lr * 0.1},
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*70)
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion_ce, criterion_triplet, device, epoch)
        val_metrics = validate(model, test_loader, criterion_ce, criterion_triplet, device)
        
        scheduler.step()
        
        print(f"\nTraining   - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Validation - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            
            save_path = ARTIFACTS / args.output
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, save_path)
            
            print(f"✓ Saved best model to {save_path} (acc: {best_acc:.2f}%)")
    
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()