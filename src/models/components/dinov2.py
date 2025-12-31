import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Tuple

class FPNDecoder(nn.Module):
    def __init__(self, in_channels: List[int], num_classes: int, feat_channels=256):
        super().__init__()
        
        # 1. Lateral layers (1x1 conv to force all scales to 256 channels)
        self.lats = nn.ModuleList([
            nn.Conv2d(c, feat_channels, 1) for c in in_channels
        ])
        
        # 2. Smooth layers (3x3 conv to reduce aliasing after upsampling)
        self.smooths = nn.ModuleList([
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1) for _ in in_channels
        ])
        
        # 3. Final Prediction Head
        self.classifier = nn.Sequential(
            nn.Conv2d(feat_channels * len(in_channels), feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, num_classes, 1)
        )
        
        # Optional: Boundary Head (Keep this, it's good!)
        self.boundary_head = nn.Conv2d(feat_channels, 1, 1)

    def forward(self, feats: List[torch.Tensor]):
        # feats are ordered [Shallow -> Deep] e.g., [layer 6, 12, 18, 23]
        
        # 1. Process Lateral Connections
        laterals = [lat(f) for lat, f in zip(self.lats, feats)]
        
        # 2. Top-Down Pathway (The "True" FPN part)
        # We start from the deepest layer and work backwards
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Take the deeper feature
            prev_shape = laterals[i - 1].shape[2:]
            
            # Upsample current feature to match the one below it
            top_down_features = F.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=False
            )
            
            # ADD them (don't concatenate)
            laterals[i - 1] = laterals[i - 1] + top_down_features
            
        # 3. Smooth the outputs
        outs = [self.smooths[i](laterals[i]) for i in range(used_backbone_levels)]
        
        # 4. Feature Aggregation (PPM-style or UPerNet-style)
        # Resize everything to the largest scale (usually the shallowest feature, e.g., 1/4 or 1/16)
        target_size = outs[0].shape[2:]
        for i in range(1, len(outs)):
            outs[i] = F.interpolate(outs[i], size=target_size, mode="bilinear", align_corners=False)
            
        # Concatenate for the final head (Now we are concatenating refined features)
        fused = torch.cat(outs, dim=1)
        
        logits = self.classifier(fused)
        boundary = self.boundary_head(outs[0]) # Use the highest res feature for boundary
        
        return logits, boundary

class DinoV2Segmentation(nn.Module):
    """
    DINOv2/v3 Backbone with FPN Decoder.
    """
    def __init__(self, backbone_name: str, num_classes: int, taps: List[int] = [6, 12, 18, 23], 
                 patch_size: int = 16, feat_channels: int = 256, num_register_tokens: int = 0):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
                    param.requires_grad = False
        self.patch_size = patch_size
        self.taps = sorted(taps)
        self.register_tokens = num_register_tokens
        
        embed_dim = self.backbone.embed_dim
        in_channels = [embed_dim] * len(self.taps)
        
        self.decoder = FPNDecoder(in_channels=in_channels, num_classes=num_classes, feat_channels=feat_channels)
        
        # Hooks
        self._feats = {}
        for idx, blk in enumerate(self.backbone.blocks):
            if idx in self.taps:
                blk.register_forward_hook(self._get_hook(idx))

    def _get_hook(self, idx):
        def hook(module, inp, out):
            self._feats[idx] = out
        return hook

    def forward(self, x):
        B, _, H, W = x.shape
        self._feats.clear()
        _ = self.backbone(x)
        
        # Extract feature maps from hooks
        maps = []
        for i in self.taps:
            feat = self._feats[i]
            # [B, N, C] -> Remove CLS/Register tokens
            feat = feat[:, 1 + self.register_tokens:, :]
            
            # Reshape to [B, C, h, w]
            
            h, w = H // self.patch_size, W // self.patch_size
            feat = feat.transpose(1, 2).reshape(B, -1, h, w)
            maps.append(feat)
            
        logits, boundary = self.decoder(maps)
        
        # Upsample to input resolution
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        boundary = F.interpolate(boundary, size=(H, W), mode="bilinear", align_corners=False)
        
        return logits, boundary