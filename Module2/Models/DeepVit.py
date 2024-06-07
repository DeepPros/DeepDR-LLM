from functools import partial
import torch
import torch.nn as nn
from Models.vision_transformer import VisionTransformer as VIT
from Models.pos_embed import interpolate_pos_embed

def vit_base_patch16(**kwargs):
    model = VIT(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
