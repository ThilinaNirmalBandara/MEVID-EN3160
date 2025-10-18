# mevid_textsearch/temporal_reid.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

class TemporalAttention(nn.Module):
    """Temporal attention mechanism to weight frame importance."""
    def __init__(self, feat_dim=2048):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.Tanh(),
            nn.Linear(feat_dim // 4, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, feat_dim)
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted = x * attn_weights
        return weighted.sum(dim=1), attn_weights.squeeze(-1)

class VideoReIDModel(nn.Module):
    """
    Video-based Person Re-Identification Model
    Uses ResNet50 backbone + Temporal Attention + Triplet Loss
    """
    def __init__(self, num_classes=None, feat_dim=2048, dropout=0.5):
        super().__init__()
        
        # Backbone: ResNet50 pretrained on ImageNet
        resnet = models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(feat_dim)
        
        # Bottleneck layer
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Classification head (optional, for training)
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        
        self._init_params()
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, seq_len, C, H, W) - video frames
            return_attention: whether to return attention weights
        Returns:
            feat: (batch, feat_dim) - normalized feature vector
            logits: (batch, num_classes) - classification scores (if num_classes set)
            attn: (batch, seq_len) - attention weights (if return_attention=True)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape for backbone processing
        x = x.view(batch_size * seq_len, *x.shape[2:])
        
        # Extract frame features
        x = self.base(x)  # (batch*seq_len, 2048, h, w)
        x = self.gap(x)   # (batch*seq_len, 2048, 1, 1)
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, 2048)
        
        # Temporal attention aggregation
        feat, attn_weights = self.temporal_attention(x)  # (batch, 2048)
        
        # Bottleneck normalization
        feat = self.bottleneck(feat)
        
        # L2 normalization for metric learning
        feat_norm = F.normalize(feat, p=2, dim=1)
        
        if self.training and self.num_classes is not None:
            if self.dropout is not None:
                feat = self.dropout(feat)
            logits = self.classifier(feat)
            
            if return_attention:
                return feat_norm, logits, attn_weights
            return feat_norm, logits
        
        if return_attention:
            return feat_norm, attn_weights
        return feat_norm


class VideoReIDExtractor:
    """
    Wrapper for extracting features from video tracklets using VideoReIDModel
    """
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = VideoReIDModel(num_classes=None)
        
        # Load pretrained weights if available
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[VideoReID] Loaded weights from {model_path}")
        else:
            print("[VideoReID] Using ImageNet pretrained ResNet50 backbone")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = T.Compose([
            T.Resize((256, 128)),  # Standard ReID resolution
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract_tracklet_feature(self, frame_paths: List[Path], max_frames=16) -> np.ndarray:
        """
        Extract feature from a tracklet (sequence of frames)
        
        Args:
            frame_paths: List of paths to frames
            max_frames: Maximum number of frames to use
        Returns:
            feat: (feat_dim,) normalized feature vector
        """
        # Sample frames evenly
        if len(frame_paths) > max_frames:
            indices = np.linspace(0, len(frame_paths)-1, max_frames, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
        
        # Load and transform frames
        frames = []
        for fp in frame_paths:
            if not fp.exists():
                continue
            try:
                img = Image.open(fp).convert('RGB')
                frames.append(self.transform(img))
            except Exception as e:
                print(f"[Warning] Failed to load {fp}: {e}")
                continue
        
        if not frames:
            # Return zero vector if no valid frames
            return np.zeros(2048, dtype=np.float32)
        
        # Stack frames: (seq_len, C, H, W)
        frames_tensor = torch.stack(frames).unsqueeze(0).to(self.device)  # (1, seq_len, C, H, W)
        
        # Extract feature
        feat = self.model(frames_tensor)  # (1, 2048)
        
        return feat.cpu().numpy()[0]  # (2048,)
    
    @torch.no_grad()
    def extract_batch_features(self, tracklets: List[List[Path]], max_frames=16, batch_size=8) -> np.ndarray:
        """
        Extract features for multiple tracklets in batches
        
        Args:
            tracklets: List of tracklets (each is a list of frame paths)
            max_frames: Maximum frames per tracklet
            batch_size: Batch size for processing
        Returns:
            feats: (N, feat_dim) feature matrix
        """
        all_feats = []
        
        for i in range(0, len(tracklets), batch_size):
            batch = tracklets[i:i+batch_size]
            batch_frames = []
            
            for frame_paths in batch:
                # Sample frames
                if len(frame_paths) > max_frames:
                    indices = np.linspace(0, len(frame_paths)-1, max_frames, dtype=int)
                    frame_paths = [frame_paths[i] for i in indices]
                
                frames = []
                for fp in frame_paths:
                    if not fp.exists():
                        continue
                    try:
                        img = Image.open(fp).convert('RGB')
                        frames.append(self.transform(img))
                    except:
                        continue
                
                if frames:
                    batch_frames.append(torch.stack(frames))
                else:
                    # Dummy frame if all failed
                    batch_frames.append(torch.zeros(1, 3, 256, 128))
            
            # Pad sequences to same length
            max_len = max(f.size(0) for f in batch_frames)
            padded = []
            for frames in batch_frames:
                if frames.size(0) < max_len:
                    padding = torch.zeros(max_len - frames.size(0), 3, 256, 128)
                    frames = torch.cat([frames, padding], dim=0)
                padded.append(frames)
            
            # Stack: (batch, seq_len, C, H, W)
            batch_tensor = torch.stack(padded).to(self.device)
            
            # Extract features
            feats = self.model(batch_tensor)  # (batch, 2048)
            all_feats.append(feats.cpu().numpy())
        
        return np.vstack(all_feats)