"""
Use EVA-X series as your backbone. You could get 
EVA-X representations simply with timm. Try them 
with your own X-ray tasks. 
Enjoy!

Reference:
    https://github.com/baaivision/EVA
    https://github.com/huggingface/pytorch-image-models
Thanks for their work!
    
by Jingfeng Yao 
from HUST-VL
"""

import torch
from timm.models.eva import Eva
from timm.layers import resample_abs_pos_embed, resample_patch_embed

def checkpoint_filter_fn(
        state_dict,
        model,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict

from typing import Tuple, Optional
import torch
from timm.layers import resample_abs_pos_embed

class EVA_X(Eva):
    def __init__(self, **kwargs):
        super(EVA_X, self).__init__(**kwargs)
    
    def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Position embedding method from timm's Eva implementation
        This is the exact implementation that EVA_X.forward_features() expects
        """
        if self.dynamic_img_size:
            B, H, W, C = x.shape              
            if self.pos_embed is not None:
                prev_grid_size = self.patch_embed.grid_size                                 
                pos_embed = resample_abs_pos_embed(                   
                    self.pos_embed,    
                    new_size=(H, W),                                                        
                    old_size=prev_grid_size,                                                
                    num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
                )                                                                           
            else:                                                                           
                pos_embed = None                                                                                                                                                        
            x = x.view(B, -1, C)                                                            
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None                                                                                        
        else:                                                                                                                                                                           
            pos_embed = self.pos_embed                                                      
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None        
                                              
        to_cat = []                           
        if self.cls_token is not None:                                                      
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if self.no_embed_class:
            # position embedding does not overlap with class / reg token
            if pos_embed is not None:
                x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # pos_embed has entry for class / reg token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            if pos_embed is not None:
                x = x + pos_embed
        x = self.pos_drop(x)
        
        # apply patch dropout to patches and rotary position embedding
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                from timm.layers import apply_keep_indices_nlc
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
                # After applying keep indices to rope embeds, batch dim is added
                if getattr(self, 'rope_mixed', False):
                    # B, D, nH, N, dim -> D, B, nH, N, dim. For consistent iteration over depth at index 0.
                    rot_pos_embed = rot_pos_embed.transpose(0, 1)
                else:
                    # B, N, dim -> B, 1, N, dim.  Need head dim singleton for correct dim alignment in axial mode.
                    rot_pos_embed = rot_pos_embed.unsqueeze(1)
        
        return x, rot_pos_embed
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x
    
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool: # 얘 안써야함 전체 뽑으려면
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        
        if pre_logits:
            # x shape == [b, seq, dim]
            return x 
        else:
            # x shape == [b, seq, dim]
            return self.head(x)
        
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x, pre_logits=True)
        return x

def eva_x_tiny_patch16(pretrained=False):
    model = EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),  # 224/16
    )
    eva_ckpt = checkpoint_filter_fn(torch.load(pretrained, map_location='cpu',weights_only=False), 
                        model)
    msg = model.load_state_dict(eva_ckpt, strict=False)
    print(msg)
    return model

def eva_x_small_patch16(pretrained=False):
    model = EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),   # 224/16
    )
    eva_ckpt = checkpoint_filter_fn(torch.load(pretrained, map_location='cpu',weights_only=False), 
                        model)
    msg = model.load_state_dict(eva_ckpt, strict=False)
    print(msg)
    return model

def eva_x_base_patch16(pretrained=False):
    model = EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),  # 224/16
    )
    eva_ckpt = checkpoint_filter_fn(torch.load(pretrained, map_location='cpu',weights_only=False), 
                        model)
    msg = model.load_state_dict(eva_ckpt, strict=False)
    print(msg)
    return model

if __name__ == '__main__':

    eva_x_ti_pt = '/home/jingfengyao/code/medical/EVA-X/classification/pretrained/eva_x_ti_16.pt'
    eva_x_s_pt = '/home/jingfengyao/code/medical/EVA-X/classification/pretrained/eva_x_s_16.pt'
    eva_x_b_pt = '/home/jingfengyao/code/medical/EVA-X/classification/pretrained/eva_x_b_16.pt'
    
    eva_x_ti = eva_x_tiny_patch16(pretrained=eva_x_ti_pt)
    eva_x_s = eva_x_small_patch16(pretrained=eva_x_s_pt)
    eva_x_b = eva_x_base_patch16(pretrained=eva_x_b_pt)