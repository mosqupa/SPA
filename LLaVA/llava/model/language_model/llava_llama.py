#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

def _patched_forward(self, x, seq_len=None):
    """Return a RoPE cache covering the model's full configured context."""
    cache_len = max(seq_len or 0, self.max_position_embeddings)
    if cache_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=cache_len, device=x.device, dtype=x.dtype)

    return (
        self.cos_cached[:cache_len].to(dtype=x.dtype),
        self.sin_cached[:cache_len].to(dtype=x.dtype),
    )

if not getattr(LlamaRotaryEmbedding, "_llava_gap_patched", False):
    LlamaRotaryEmbedding._llava_original_forward = LlamaRotaryEmbedding.forward
    LlamaRotaryEmbedding.forward = _patched_forward
    LlamaRotaryEmbedding._llava_gap_patched = True


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        use_pos_adapter: bool = False,
        shuffle_coords: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                use_pos_adapter=use_pos_adapter,
                shuffle_coords=shuffle_coords
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        keep_ratio = kwargs.pop("keep_ratio", 1.0)
        use_2d_pe = kwargs.pop("use_2d_pe", False)
        pe_scale = kwargs.pop("pe_scale", 1.0)
        shuffle_pe = kwargs.pop("shuffle_pe", False)
        use_noise = kwargs.pop("use_noise", False)
        use_pos_adapter = kwargs.pop("use_pos_adapter", False)
        shuffle_coords = kwargs.pop("shuffle_coords", False)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                keep_ratio=keep_ratio,
                use_2d_pe=use_2d_pe,
                pe_scale=pe_scale,
                shuffle_pe=shuffle_pe,
                use_noise=use_noise,
                use_pos_adapter=use_pos_adapter,
                shuffle_coords=shuffle_coords
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)


        # Retain the fixed prompt coordinate system for optional attention
        # export. Keeping CPU copies avoids holding extra GPU memory.
        self._last_generation_prompt_position_ids = (
            position_ids.detach().cpu() if position_ids is not None else None
        )
        self._last_generation_prompt_attention_mask = (
            attention_mask.detach().cpu() if attention_mask is not None else None
        )

        return super().generate(
            position_ids=position_ids, # [batch, seq_len]
            attention_mask=attention_mask, # None
            inputs_embeds=inputs_embeds, # [batch, seq_len, embed_dim]
            **kwargs # ['do_sample', 'temperature', 'top_p', 'num_beams', 'max_new_tokens', 'use_cache']
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if past_key_values is not None:
            full_position_ids = kwargs.get("position_ids") # get the full position_ids from kwargs [batch, seq_len]
            if full_position_ids is not None:
                decode_input_ids = inputs.get("input_ids")
                decode_len = decode_input_ids.shape[-1]
                prompt_len = full_position_ids.shape[-1]
                attention_mask = inputs.get("attention_mask")

                if attention_mask is not None and attention_mask.shape[-1] >= prompt_len:
                    prompt_mask = attention_mask[:, :prompt_len].to(device=full_position_ids.device, dtype=torch.bool)
                    prompt_indices = torch.arange(prompt_len, device=full_position_ids.device,).unsqueeze(0).expand_as(prompt_mask)
                    last_prompt_indices = prompt_indices.masked_fill(~prompt_mask, -1).amax(dim=-1, keepdim=True).clamp_min(0)
                    last_prompt_position = full_position_ids.gather(1, last_prompt_indices)
                    generated_count = attention_mask.shape[-1] - prompt_len
                else:
                    last_prompt_position = full_position_ids[:, -1:]
                    if hasattr(past_key_values, "get_seq_length"):
                        past_length = past_key_values.get_seq_length()
                    else:
                        past_length = past_key_values[0][0].shape[-2]
                    generated_count = past_length - prompt_len + decode_len

                first_offset = generated_count - decode_len + 1
                decode_offsets = torch.arange(
                    first_offset,
                    generated_count + 1,
                    dtype=full_position_ids.dtype,
                    device=full_position_ids.device,
                ).unsqueeze(0)
                inputs["position_ids"] = last_prompt_position + decode_offsets
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
