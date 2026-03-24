import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip


# =================================================================================
# 1. Helper Modules
# =================================================================================

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class ResidualAttentionBlock(nn.Module):
    """Standard ViT Block used for Decoder."""
    def __init__(self, d_model, n_head=8):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):
    """
    Simple Decoder composed of 3 Residual Attention Blocks.
    Reconstructs features in reverse order (Deep -> Shallow).
    """
    def __init__(self, dim, num_layers=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: Bottleneck feature from MoE [B, N, Dim]
        reconstructions = []
        current_feat = x
        
        # Process sequentially: 
        # MoE -> Block3 -> Rec3 -> Block2 -> Rec2 -> Block1 -> Rec1
        for block in self.blocks:
            current_feat = block(current_feat)
            reconstructions.append(current_feat)
            
        # Reverse order: Deepest -> Shallowest
        return reconstructions[::-1]

class Projector(nn.Module):
    """
    Layer-specific projector.
    Applies LayerNorm -> Projection to map Visual features (768) to Text space (512).
    """
    def __init__(self, width, output_dim):
        super().__init__()
        scale = width ** -0.5
        self.ln = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        # Applies to the entire sequence (CLS + Patches)
        x = self.ln(x)
        x = x @ self.proj
        return x

class MultiLayerFusion(nn.Module):
    """Concatenates and projects features."""
    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__()
        self.proj = nn.Linear(input_dim * num_layers, output_dim)
        self.ln = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        # features: List of [B, Seq_Len, C]
        # Concatenate features from different layers
        x = torch.cat(features, dim=-1) # [B, Seq_Len, C*L]
        x = self.proj(x)
        x = self.ln(x)
        x = self.dropout(x)
        return x

class NormalityPromotion(nn.Module):
    def forward(self, visual_feat, text_live, text_spoof):
        # visual_feat: [B, N, C]
        alpha = torch.einsum('bnc,c->bn', visual_feat, text_live)
        beta = torch.einsum('bnc,c->bn', visual_feat, text_spoof)
        
        psi = 0.5 * (1 + torch.tanh(alpha - beta))
        
        norm = torch.norm(visual_feat, dim=-1, keepdim=True) + 1e-6
        lam = 1.0 / norm
        
        f_star = visual_feat + (lam * psi.unsqueeze(-1)) * visual_feat
        return f_star

class GatedMixtureOfExperts(nn.Module):
    def __init__(self, dim, num_experts=5, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        x_flat = x.view(-1, C)
        
        logits = self.router(x_flat)
        top_k_logits, indices = logits.topk(self.top_k, dim=1)
        weights = F.softmax(top_k_logits, dim=1)
        
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = indices[:, k]
            expert_weights = weights[:, k].unsqueeze(1)
            for e_idx in range(self.num_experts):
                mask = (expert_indices == e_idx)
                if mask.any():
                    output[mask] += self.experts[e_idx](x_flat[mask]) * expert_weights[mask]
                    
        return output.view(B, N, C), logits

# =================================================================================
# 2. Prompt Learner
# =================================================================================

class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.num_layers = len(cfg['feature_layers'])
        classnames = ["face", "spoof face"]
        n_ctx = cfg.get('n_ctx', 12)
        ctx_init = cfg.get('ctx_init', "")
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            ctx_vectors = ctx_vectors.unsqueeze(0).unsqueeze(0).expand(self.num_layers, len(classnames), -1, -1)
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(self.num_layers, len(classnames), n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'CNC-FAS: Initializing layer-specific prompts (n_ctx={n_ctx})')
        self.ctx = nn.Parameter(ctx_vectors) 
        
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :]) 
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        prefix = self.token_prefix.unsqueeze(0).expand(self.num_layers, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(self.num_layers, -1, -1, -1)
        return torch.cat([prefix, self.ctx, suffix], dim=2)

# =================================================================================
# 3. Main Model
# =================================================================================

class CNC_FAS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.device = cfg.get('device', 'cpu')
        
        print(f"Loading CLIP: {cfg['backbone']}")
        clip_model, _ = clip.load(cfg['backbone'], device='cpu')
        self.clip_visual = clip_model.visual.float()
        self.dtype = clip_model.dtype
        
        self.vit_width = self.clip_visual.transformer.width 
        self.embed_dim = self.clip_visual.output_dim
        self.feat_layers = cfg['feature_layers']
        
        # 1. Prompt Learner & Text Encoder
        self.prompt_learner = PromptLearner(cfg, clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.fnp = NormalityPromotion()
        
        # 2. Main Architecture (at ViT Width)
        self.fusion = MultiLayerFusion(self.vit_width, self.vit_width, len(self.feat_layers))
        self.moe = GatedMixtureOfExperts(self.vit_width, num_experts=5, top_k=2)
        self.decoder = Decoder(self.vit_width, num_layers=len(self.feat_layers))
        
        # 3. Separate Projectors for EACH layer in `feature_layers`
        # Using the simplified CLIP projector (LN + Proj)
        self.layer_projectors = nn.ModuleList([
            Projector(self.vit_width, self.embed_dim)
            for _ in range(len(self.feat_layers))
        ])

        # Freeze CLIP parameters
        for p in self.clip_visual.parameters(): p.requires_grad = False
        for p in self.text_encoder.parameters(): p.requires_grad = False
        
        self.hooks = {}
        for idx in self.feat_layers:
            self.clip_visual.transformer.resblocks[idx].register_forward_hook(
                lambda m, i, o, idx=idx: self.hooks.update({idx: o})
            )

    def forward(self, image):
        self.hooks = {}
        with torch.no_grad():
            _ = self.clip_visual(image)
        
        B = image.shape[0]
        # Collect sequences (CLS + Patches) from all hooked layers
        raw_sequences = [] 
        for idx in self.feat_layers:
            # hook output: [Seq_Len, B, Width] -> Permute to [B, Seq_Len, Width]
            seq = self.hooks[idx].permute(1, 0, 2)
            raw_sequences.append(seq)
            
        seq_len = raw_sequences[0].shape[1]
        grid_h = int((seq_len - 1)**0.5)

        # --- A. Text Features ---
        prompts = self.prompt_learner() 
        tokenized = self.prompt_learner.tokenized_prompts
        L = len(self.feat_layers)
        
        text_features = self.text_encoder(
            prompts.view(-1, prompts.shape[2], prompts.shape[3]), 
            tokenized.repeat(L, 1)
        )
        text_features = F.normalize(text_features, dim=-1).view(L, 2, -1)

        # --- B. Fusion & MoE ---
        # Fusing the ENTIRE sequence (CLS included)
        fused_seq = self.fusion(raw_sequences)
        moe_seq, router_logits = self.moe(fused_seq)
        
        # --- C. Reconstruction ---
        # Decoder produces 3 layers of reconstructed [CLS + Patches]
        rec_sequences = self.decoder(moe_seq)

        # --- D. Loss Components ---
        layer_maps = []
        layer_logits = []

        for i in range(L):
            # 1. Project the ENTIRE sequence using THIS layer's projector
            projector = self.layer_projectors[i]
            
            # Projects [B, N, 768] -> [B, N, 512]
            enc_feat_proj = projector(raw_sequences[i])
            rec_feat_proj = projector(rec_sequences[i])
            
            enc_feat = F.normalize(enc_feat_proj, dim=-1)
            rec_feat = F.normalize(rec_feat_proj, dim=-1)
            
            # 2. Split CLS (0) and Patches (1:)
            enc_cls = enc_feat[:, 0, :]
            enc_patches = enc_feat[:, 1:, :]
            
            rec_cls = rec_feat[:, 0, :]
            rec_patches = rec_feat[:, 1:, :]
            
            t_live = text_features[i, 0]
            t_spoof = text_features[i, 1]
            
            # 3. Normality Logits (For THIS layer's CLS token)
            logits_enc = torch.stack([
                torch.matmul(enc_cls, t_live),
                torch.matmul(enc_cls, t_spoof)
            ], dim=1) / 0.07
            
            logits_dec = torch.stack([
                torch.matmul(rec_cls, t_live),
                torch.matmul(rec_cls, t_spoof)
            ], dim=1) / 0.07
            
            layer_logits.append({"enc": logits_enc, "dec": logits_dec})
            
            # 4. Anomaly Map (FNP on Patches)
            target_star = self.fnp(enc_patches, t_live, t_spoof)
            pred_star = self.fnp(rec_patches, t_live, t_spoof)
            
            sim = F.cosine_similarity(target_star, pred_star, dim=-1)
            sim_map = sim.view(B, grid_h, grid_h)
            layer_maps.append(1.0 - sim_map)

        # Aggregation
        stacked_maps = torch.stack(layer_maps, dim=1)
        # Define spoof score as the max pixel score of the stacked maps summed over layers
        spoof_score = stacked_maps.view(B, L, -1).mean(dim=-1).max(dim=-1).values
        # avg_map = stacked_maps.mean(dim=1)
        # spoof_score = avg_map.view(B, -1).max(dim=-1).values

        final_map = F.interpolate(
            stacked_maps.mean(dim=1).unsqueeze(1), 
            size=image.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        # spoof_score = final_map.view(B, -1).mean(dim=1)
        
        return {
            "layer_logits": layer_logits,
            "anomaly_maps": stacked_maps,
            "spoof_score": spoof_score,
            "router_logits": router_logits,
            "final_map": final_map
        }