# multi_view_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SVR.encoder import resnet18  # or resnet18ReturnMid if you need mid-layer
from models.SVR.decoder import SP_DecoderEigen3steps
from utils.model_utils import get_spherepoints



class MultiViewFeatureFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)

    def forward(self, features):
        # features shape: (batch_size, num_views, feature_dim)
        attention_output, _ = self.self_attention(features, features, features)
        fused_feature = attention_output.mean(dim=1)  # Aggregate across views
        return fused_feature
class Model(nn.Module):
    def __init__(self, args, num_views=3):
        super().__init__()
        self.args = args
        self.num_views = num_views

        # Use standard ResNet-18 for image encoding
        self.encoder = resnet18(pretrained=False, num_classes=1000)
        self.linear = nn.Linear(1000, 128)

        # Multi-head self-attention for fusing view features
        self.feature_fusion = MultiViewFeatureFusion(feature_dim=128)
        # You can tweak num_heads or embed_dim as desired

        # The same decoder from your SVR_3DAttriFlow or PC_3DAttriFlow
        self.decoder = SP_DecoderEigen3steps(args)

    def forward(self, images, sphere_xyz=None):
        """
        images: (B, V, 3, H, W), where V = self.num_views
        sphere_xyz: optional sphere points (B, 3, N)
        Returns: Reconstructed points (B, N, 3)
        """
        B, V, C, H, W = images.shape
        # Flatten the batch and view dimension to encode each view independently
        images_flat = images.view(B*V, C, H, W)

        # Encode each view
        feat_1000 = self.encoder(images_flat)         # (B*V, 1000)
        feat_128 = self.linear(feat_1000)             # (B*V, 128)

        # Reshape back to (B, V, 128)
        feat_128 = feat_128.view(B, V, 128)

        # Self-attention across views (treat each of the V features as "tokens")
        # Query = Key = Value = feat_128
        fused_feat, attn_weights = self.feature_fusion(feat_128, feat_128, feat_128)
        # fused_feat shape: (B, V, 128)
        # attn_weights shape: (B, V, V)

        # Option 1: just average the V tokens -> (B, 128) final feature
        fused_feat = fused_feat.mean(dim=1)   # (B, 128)

        # Prepare sphere points
        if sphere_xyz is not None:
            sphere_points = sphere_xyz.unsqueeze(0).repeat(B, 1, 1).cuda()
        else:
            sphere_points = get_spherepoints(2048, self.args.radius)
            sphere_points = torch.FloatTensor(sphere_points).unsqueeze(0).repeat(B, 1, 1).cuda()
            sphere_points = sphere_points.transpose(2,1).contiguous()

        # Decode fused feature into 3D shape
        outputs = self.decoder(sphere_points, fused_feat)  # (B, 3, 2048)
        outputs = outputs.transpose(2,1).contiguous()      # (B, 2048, 3)
        return outputs, fused_feat
