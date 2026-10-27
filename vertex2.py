#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VORTEX AI — PyTorch/Transformers Integration Layer
==================================================
Moteur d'intégration local-first pour l'entraînement et l'inférence.
Zéro télémétrie. Zéro vendor lock-in.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List, Dict, Any
import warnings

# Vérification de la disponibilité de Vortex (Fallback sécurisé)
vortex_available = False
try:
    import vortex
    from vortex.ai import FusedAttention, FusedLoRA, ModelConfig, LoRAConfig, ai_utils
    vortex_available = True
except ImportError:
    warnings.warn("Vortex AI core non disponible. Mode fallback PyTorch natif activé.")

# =============================================================================
# 1. VORTEX ATTENTION (FlashAttention optimisé)
# =============================================================================
class VortexAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, num_key_value_heads: int, 
                 attention_dropout: float = 0.0, max_position_embeddings: int = 4096, rope_theta: float = 10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        self._init_rope()

    def _init_rope(self):
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False, **kwargs) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._get_cos_sin(seq_length, query.device, query.dtype)
        query, key = self._apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)

        past_key_value = (key, value) if use_cache else None

        # Utilisation de l'attention PyTorch native optimisée (FlashAttention)
        try:
            from torch.nn.functional import scaled_dot_product_attention
            attn_output = scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=attention_mask is None
            )
        except ImportError:
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = torch.dropout(attn_weights, self.attention_dropout, train=self.training)
            attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        return self.o_proj(attn_output), past_key_value

    def _get_cos_sin(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        if position_ids is None:
            position_ids = torch.arange(q.size(2), device=q.device)
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

# =============================================================================
# 2. VORTEX MLP (SwiGLU)
# =============================================================================
class VortexMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# =============================================================================
# 3. VORTEX TRANSFORMER LAYER
# =============================================================================
class VortexTransformerLayer(nn.Module):
    def __init__(self, config: Dict[str, Any], layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = VortexAttention(
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
            attention_dropout=config.get("attention_dropout", 0.0),
            max_position_embeddings=config["max_position_embeddings"],
            rope_theta=config.get("rope_theta", 10000.0),
        )
        self.mlp = VortexMLP(config["hidden_size"], config["intermediate_size"])
        self.input_layernorm = nn.RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6))
        self.post_attention_layernorm = nn.RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6))
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.gradient_checkpointing and self.training:
            # Correction : utilisation de lambda pour contourner le problème des arguments nommés
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, attention_mask=attention_mask, position_ids=position_ids, 
                                  past_key_value=past_key_value, use_cache=use_cache)
                return custom_forward
            hidden_states, present_key_value = checkpoint(create_custom_forward(self.self_attn), hidden_states)
        else:
            hidden_states, present_key_value = self.self_attn(
                hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, use_cache=use_cache)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value

# =============================================================================
# 4. VORTEX MODEL (Le Cœur)
# =============================================================================
class VortexModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"], padding_idx=config.get("pad_token_id", None))
        self.layers = nn.ModuleList([VortexTransformerLayer(config, i) for i in range(config["num_hidden_layers"])])
        self.norm = nn.RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False, labels=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)

        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_key_value = layer(hidden_states, attention_mask, position_ids, past_key_value, use_cache)
            if use_cache:
                present_key_values.append(present_key_value)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config["vocab_size"]), shift_labels.view(-1))

        return {"loss": loss, "logits": logits, "past_key_values": present_key_values}

    def apply_lora(self, rank: int = 32, alpha: float = 16.0, **kwargs):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=rank, lora_alpha=alpha,
                target_modules=kwargs.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                lora_dropout=kwargs.get("dropout", 0.05), bias="none", task_type=TaskType.CAUSAL_LM
            )
            return get_peft_model(self, lora_config)
        except ImportError:
            warnings.warn("PEFT non disponible. LoRA non appliqué.")
            return self

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = cls(config.to_dict())
        pretrained = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        model.load_state_dict(pretrained.state_dict(), strict=False)
        return model

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=50, **kwargs):
        past_key_values = None
        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                past_key_values=past_key_values, use_cache=True
            )
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    sorted_probs[cumsum > top_p] = 0
                    probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if (next_token == self.config.get("eos_token_id", 2)).all():
                break
        return input_ids

# =============================================================================
# 5. VORTEX TRAINER (Entraînement)
# =============================================================================
class VortexTrainer:
    def __init__(self, model: VortexModel, optimizer: Optional[torch.optim.Optimizer] = None, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        self.scheduler = None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler: self.scheduler.step()
        return loss.item()

__all__ = ["VortexAttention", "VortexMLP", "VortexTransformerLayer", "VortexModel", "VortexTrainer"]
