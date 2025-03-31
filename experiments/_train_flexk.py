#!/usr/bin/env python
import os
import sys
import time
import datetime
import random
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import argparse
import glob

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Configuration
@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384
    dropout: int = 0.2
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.0

# Helper Functions
def rms_norm(x, dim=-1, eps=1e-6):
    return x / torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=True) + eps)

def causal_mask(b, h, q_idx, k_idx):
    return q_idx >= k_idx

class Rotary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        t = torch.arange(x.size(2), device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

def apply_rotary_emb(x, cos, sin):
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# Model Components
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.head_dim = config.n_embed // config.n_head
        self.c_attention = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5))
        
    def to_float16(self):
        """Convert model parameters to float16"""
        self.c_attention.weight.data = self.c_attention.weight.data.half()
        if self.c_attention.bias is not None:
            self.c_attention.bias.data = self.c_attention.bias.data.half()
        self.c_proj.weight.data = self.c_proj.weight.data.half()
        if self.c_proj.bias is not None:
            self.c_proj.bias.data = self.c_proj.bias.data.half()

    def forward(self, x, v1=None, mask_mod=None):
        B, T, C = x.size()
        
        # Keep track of input dtype for autocast
        input_dtype = x.dtype
        
        # QKV projection
        qkv = self.c_attention(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        
        query = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        key = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        value = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        if v1 is None:
            v1 = value
        else:
            v1 = v1.view_as(value)
        value = (1 - self.lamb) * value + self.lamb * v1
        
        query = rms_norm(query, dim=-1)
        key = rms_norm(key, dim=-1)
        cos, sin = self.rotary(query)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        
        if mask_mod is None:
            mask_mod = causal_mask
        block_mask = create_block_mask(mask_mod, B, self.n_head, T, T, device=x.device)
        scale = 1 / (self.head_dim ** 0.5)
        
        # Ensure same dtype for flex_attention
        flex_dtype = torch.float16 if torch.is_autocast_enabled() else torch.float32
        query = query.to(flex_dtype)
        key = key.to(flex_dtype)
        value = value.to(flex_dtype)
        
        attn_output = flex_attention(query, key, value, block_mask=block_mask, scale=scale)
        
        # Convert back to input dtype
        attn_output = attn_output.to(input_dtype)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.c_proj(attn_output)
        
        return attn_output, value

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms_norm(x, dim=-1) * self.scale

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x).square()

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            ReLUSquared(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )
    def forward(self, x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.top_k
        self.topkroute_linear = nn.Linear(config.n_embed, config.num_experts)
        self.noise_linear = nn.Linear(config.n_embed, config.num_experts)
    def forward(self, x):
        logits = self.topkroute_linear(x)
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output

class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = NoisyTopkRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.num_experts = config.num_experts
    def forward(self, x):
        B, T, C = x.shape
        gating_output = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, C)
        flat_gating_output = gating_output.view(-1, self.num_experts)
        tokens_per_batch = B * T * self.top_k
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
        updates = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            expert_mask = (gating_output.argmax(dim=-1) == i)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            limited_indices = (selected_indices[:expert_capacity]
                               if selected_indices.numel() > expert_capacity
                               else selected_indices)
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)
        final_output += updates.view(B, T, C)
        return final_output

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.norm1 = RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embed)
        self.moe = SparseMoE(config)

    def forward(self, x, x0, v1=None, mask_mod=None):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        attn_in = rms_norm(x, dim=-1)
        attn_out, v_new = self.attn(attn_in, v1, mask_mod=mask_mod)
        x = x + attn_out
        x = x + self.moe(rms_norm(x, dim=-1))
        return x, v_new

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm_f = RMSNorm(config.n_embed),
        ))
        self.encoder_layers = config.n_layer // 2
        self.decoder_layers = config.n_layer - self.encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.decoder_layers))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()
        self.embed_norm = RMSNorm(config.n_embed)
        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def to(self, *args, **kwargs):
        # Override to method to ensure consistent dtype
        super().to(*args, **kwargs)
        if len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]
        elif 'dtype' in kwargs:
            dtype = kwargs['dtype']
        else:
            return self
            
        def convert_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.data = m.weight.data.to(dtype)
                if m.bias is not None:
                    m.bias.data = m.bias.data.to(dtype)
                    
        self.apply(convert_weights)
        return self

    def forward(self, idx, targets):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        def document_causal_mask(b, h, q_idx, k_idx):
            causal = q_idx >= k_idx
            window = q_idx - k_idx < 1024
            return causal & window
        
        x = self.transformer.wte(idx)
        x = self.embed_norm(x)
        x0 = x
        v1 = None
        skip_connections = []
        
        for i in range(self.encoder_layers):
            x, v1 = self.transformer.h[i](x, x0, v1, mask_mod=document_causal_mask)
            skip_connections.append(x)
        
        for i in range(self.decoder_layers):
            skip = skip_connections.pop()
            weighted_skip = self.skip_weights[i] * skip
            x, v1 = self.transformer.h[self.encoder_layers + i](x + weighted_skip, x0, v1, mask_mod=document_causal_mask)
        
        x = self.transformer.norm_f(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        self.eval()
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(idx.device.type=="cuda")):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size:]
                x = self.transformer.wte(idx_cond)
                x = self.embed_norm(x)
                x0 = x
                v1 = None
                skip_connections = []
                for i in range(self.encoder_layers):
                    x, v1 = self.transformer.h[i](x, x0, v1)
                    skip_connections.append(x)
                for i in range(self.decoder_layers):
                    skip = skip_connections.pop()
                    weighted_skip = self.skip_weights[i] * skip
                    x, v1 = self.transformer.h[self.encoder_layers + i](x + weighted_skip, x0, v1)
                x = self.transformer.norm_f(x)
                logits = self.lm_head(x)
                logits = 30 * torch.tanh(logits / 30)
                logits_last = logits[:, -1, :]
                probs = F.softmax(logits_last, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
        self.train()
        return idx

class ShardedDataLoader:
    def __init__(self, data_dir, pattern, batch_size, block_size, split="train"):
        self.batch_size = batch_size
        self.block_size = block_size
        self.files = glob.glob(os.path.join(data_dir, pattern))
        self.split = split

    def next_batch(self):
        return torch.randint(0, 50304, (self.batch_size, self.block_size)), \
               torch.randint(0, 50304, (self.batch_size, self.block_size))

def decode(tokens):
    enc = tiktoken.get_encoding("gpt2")
    return enc.decode(tokens)

# Test Functions for FlexAttention
def test_flex_attention_causal(device):
    B, T, n_head, head_dim = 1, 5, 1, 4  # Small sizes for testing
    query = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.float16)
    key = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.float16)
    value = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.float16)
    
    def mask_mod(b, h, q_idx, k_idx):
        return q_idx >= k_idx  # Causal mask
    
    block_mask = create_block_mask(mask_mod, B, n_head, T, T, device=device)
    scale = 1 / (head_dim ** 0.5)
    
    # FlexAttention
    attn_output_flex = flex_attention(query, key, value, block_mask=block_mask, scale=scale)
    
    # Standard attention
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    mask = torch.tril(torch.ones(T, T, device=device)).bool()
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output_standard = torch.matmul(attn_weights, value)
    
    # Check difference
    diff = torch.abs(attn_output_flex - attn_output_standard).max()
    print(f"[Test] Causal Mask - Max difference between FlexAttention and standard: {diff.item()}")
    if diff.item() < 1e-5:
        print("[Test] FlexAttention matches standard attention for causal mask.")
    else:
        print("[Test] Warning: FlexAttention does not match standard attention for causal mask.")

def test_flex_attention_windowed(device):
    B, T, n_head, head_dim = 1, 10, 1, 4  # Small sizes for testing
    query = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.float16)
    key = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.float16)
    value = torch.randn(B, n_head, T, head_dim, device=device, dtype=torch.float16)
    
    window_size = 3  # Small window for testing (model uses 1024)
    
    def mask_mod(b, h, q_idx, k_idx):
        causal = q_idx >= k_idx
        window = q_idx - k_idx < window_size
        return causal & window
    
    block_mask = create_block_mask(mask_mod, B, n_head, T, T, device=device)
    scale = 1 / (head_dim ** 0.5)
    
    # FlexAttention
    attn_output_flex = flex_attention(query, key, value, block_mask=block_mask, scale=scale)
    
    # Standard attention with window mask
    mask = torch.zeros(T, T, device=device).bool()
    for q in range(T):
        start = max(0, q - window_size + 1)
        mask[q, start:q + 1] = True
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output_standard = torch.matmul(attn_weights, value)
    
    # Check difference
    diff = torch.abs(attn_output_flex - attn_output_standard).max()
    print(f"[Test] Windowed Mask - Max difference between FlexAttention and standard: {diff.item()}")
    if diff.item() < 1e-5:
        print("[Test] FlexAttention matches standard attention for windowed mask.")
    else:
        print("[Test] Warning: FlexAttention does not match standard attention for windowed mask.")

# Main Training Script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Training with FlexAttention")
    parser.add_argument("--experiment_name", type=str, default="GPT_Training", help="Name of the experiment")
    args = parser.parse_args()
    experiment_name = args.experiment_name

    batch_size = 32
    max_iters = 250
    learning_rate = 6e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run FlexAttention tests
    print("[Verification] Testing FlexAttention...")
    test_flex_attention_causal(device)
    test_flex_attention_windowed(device)
    print("[Verification] Tests complete.\n")

    config = GPTConfig()
    model = GPT(config)
    model = model.to(device)
    model = model.to(torch.float32)  # Keep model in float32 for training
    scaler = torch.amp.GradScaler('cuda')
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    data_dir = "finewebedu10B"
    train_loader = ShardedDataLoader(data_dir, "finewebedu_train_*.bin", batch_size, config.block_size, split="train")
    valid_loader = ShardedDataLoader(data_dir, "finewebedu_train_*.bin", batch_size, config.block_size, split="test")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        print("[Training] Starting training loop...")
        start_time = time.time()

        for i in tqdm(range(max_iters), desc="Training"):
            iter_start = time.time()
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Use autocast for forward pass
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                logits, loss = model(x, y)
            
            # Scale loss and compute gradients
            scaler.scale(loss).backward()
            
            # Unscale gradients and step optimizer
            scaler.step(optimizer)
            scaler.update()
            
            iter_time = time.time() - iter_start
            if i % 10 == 0:
                model.eval()
                with torch.no_grad():
                    vx, vy = valid_loader.next_batch()
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                        _, v_loss = model(vx, vy)
                model.train()
                print(f"[Training] Step {i}: Loss={loss.item():.4f}, ValLoss={v_loss.item():.4f}, IterTime={iter_time*1000:.2f} ms")
                mlflow.log_metric("train_loss", loss.item(), step=i)
                mlflow.log_metric("val_loss", v_loss.item(), step=i)
            
            if i % 100 == 0:
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=50)
                gen_text = decode(generated[0].tolist())
                print(f"[Generation] Step {i}: {gen_text}")
                mlflow.log_param(f"gen_text_{i}", gen_text)

        total_time = time.time() - start_time
        print(f"[Training] Training complete in {total_time:.2f} seconds")
        mlflow.log_metric("total_training_time_s", total_time)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_save_path = os.path.join(checkpoint_dir, f"model_{timestamp}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"[Checkpoint] Model saved at {model_save_path}")
        mlflow.log_artifact(model_save_path)

    print("[Generation] Generating final text sample...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = model.generate(context, max_new_tokens=500)
    generated_text = decode(output[0].tolist())
    print("[Generation] Generated Text:")
    print(generated_text)
    with open("output.txt", "w", encoding='utf-8') as f:
        f.write(generated_text)