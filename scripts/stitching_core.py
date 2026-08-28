"""Verified Qwen3 cut-boundary and cached-generation primitives.

The implementation deliberately follows Transformers 5.16.1 directly. It is
small research code, not a generic stitching framework.
"""

import hashlib
import re
import time

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

BASE = "Qwen/Qwen3-4B"
GUARD = "Qwen/Qwen3Guard-Gen-4B"
REVISIONS = {
    BASE: "1cfa9a7208912126459214e8b04321603b3df60c",
    GUARD: "6ec42827da0c1ff11e7a49dc269d2e810d27e108",
}
CUT = 18
HIDDEN_SIZE = 2560


def freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def render_guard_prompt(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def tokenize_guard_prompt(tokenizer, text, device=None):
    rendered = render_guard_prompt(tokenizer, text)
    batch = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    return batch.to(device) if device is not None else batch


def masks_and_rope(backbone, hidden_states, attention_mask, position_ids):
    kwargs = dict(
        config=backbone.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    masks = {"full_attention": create_causal_mask(**kwargs)}
    if backbone.has_sliding_layers:
        masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
    return masks, backbone.rotary_emb(hidden_states, position_ids)


def prefix_hidden_states(model, input_ids, attention_mask, cut=CUT):
    backbone = model.model
    hidden = backbone.embed_tokens(input_ids)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    masks, rope = masks_and_rope(backbone, hidden, attention_mask, positions)
    for index in range(cut):
        hidden = backbone.layers[index](
            hidden,
            attention_mask=masks[backbone.config.layer_types[index]],
            position_ids=positions,
            position_embeddings=rope,
            past_key_values=None,
            use_cache=False,
        )
    return hidden


def manual_self_logits(model, input_ids, attention_mask, cut=CUT):
    backbone = model.model
    hidden = backbone.embed_tokens(input_ids)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    masks, rope = masks_and_rope(backbone, hidden, attention_mask, positions)
    for index, layer in enumerate(backbone.layers):
        hidden = layer(
            hidden,
            attention_mask=masks[backbone.config.layer_types[index]],
            position_ids=positions,
            position_embeddings=rope,
            past_key_values=None,
            use_cache=False,
        )
        if index + 1 == cut:
            hidden = hidden  # explicit identity boundary
    return model.lm_head(backbone.norm(hidden))


def parse_guard_label(text):
    match = re.search(r"Safety\s*:\s*(Safe|Controversial|Unsafe)\b", text, re.I)
    return None if match is None else match.group(1).capitalize()


def eos_ids(model):
    value = model.generation_config.eos_token_id
    if value is None:
        return set()
    return set(value if isinstance(value, (list, tuple)) else [value])


def stitched_cached_step(donor, receiver, adapter, input_ids, attention_mask, cache, cut=CUT):
    """One prefill or incremental step; regression-tested against HF generation."""
    query_length = input_ids.shape[1]
    past_length = cache.get_seq_length(0)
    positions = torch.arange(
        past_length, past_length + query_length, device=input_ids.device
    ).unsqueeze(0)
    hidden = donor.model.embed_tokens(input_ids)
    donor_mask = create_causal_mask(
        config=donor.config, inputs_embeds=hidden, attention_mask=attention_mask,
        past_key_values=cache, position_ids=positions, layer_idx=0,
    )
    receiver_mask = create_causal_mask(
        config=receiver.config, inputs_embeds=hidden, attention_mask=attention_mask,
        past_key_values=cache, position_ids=positions, layer_idx=cut,
    )
    donor_rope = donor.model.rotary_emb(hidden, positions)
    for index in range(cut):
        hidden = donor.model.layers[index](
            hidden, attention_mask=donor_mask, position_ids=positions,
            position_embeddings=donor_rope, past_key_values=cache, use_cache=True,
        )
    hidden = adapter(hidden.float()).to(hidden.dtype) if isinstance(adapter, nn.Module) else adapter(hidden)
    receiver_rope = receiver.model.rotary_emb(hidden, positions)
    for index in range(cut, receiver.config.num_hidden_layers):
        hidden = receiver.model.layers[index](
            hidden, attention_mask=receiver_mask, position_ids=positions,
            position_embeddings=receiver_rope, past_key_values=cache, use_cache=True,
        )
    return receiver.lm_head(receiver.model.norm(hidden)[:, -1:, :])


def stitched_cached_greedy(donor, receiver, adapter, input_ids, attention_mask,
                            max_new_tokens=12, cut=CUT):
    cache = DynamicCache(config=receiver.config)
    generated = []
    started = time.perf_counter()
    with torch.inference_mode():
        logits = stitched_cached_step(
            donor, receiver, adapter, input_ids, attention_mask, cache, cut
        )
    next_token = logits[:, -1].argmax(-1, keepdim=True)
    generated.append(next_token.cpu())
    stop = eos_ids(receiver)
    for _ in range(1, max_new_tokens):
        if stop and int(next_token.item()) in stop:
            break
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, device=attention_mask.device)], dim=1
        )
        with torch.inference_mode():
            logits = stitched_cached_step(
                donor, receiver, adapter, next_token, attention_mask, cache, cut
            )
        next_token = logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_token.cpu())
    return torch.cat(generated, dim=1), time.perf_counter() - started
