from transformers import AutoModelForCausalLM
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from torch.nn import CrossEntropyLoss
from transformers.utils import logging
import numpy as np
from src_agent.training.qformer import QFormer
from transformers import AutoTokenizer
from transformers import AutoConfig
logger = logging.get_logger(__name__)
import os
from datetime import datetime
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLModel, Qwen3VLCausalLMOutputWithPast
from transformers.modeling_outputs import BaseModelOutputWithPast

current_dir = os.path.dirname(os.path.abspath(__file__))
inference_dir = os.path.join(current_dir, '../../../planner-matter-inference')
inference_dir = os.path.normpath(inference_dir)

# Debug file path
DEBUG_LOG_FILE = os.path.join(current_dir, 'gradient_debug_log.txt')

def debug_log(message):
    """Write debug message to file"""
    with open(DEBUG_LOG_FILE, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        f.write(f"[{timestamp}] {message}\n")
        f.flush()
        
class Qwen3VLForConditionalGeneration_new(Qwen3VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        # EDIT: Initialize custom models
        config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        self.model_inf = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.bfloat16,
            config=config,
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
        )
        # Get hidden dimension from config (Qwen3-VL uses text_config.hidden_size)
        hidden_dim = config.text_config.hidden_size
        self.knowledge_processor = QFormer(
            dim=hidden_dim,
            embedding_dim=hidden_dim
        )
        for param in self.knowledge_processor.parameters():
            param.requires_grad = True
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here
        self.knowledge_rope_deltas = None
        # Initialize weights and apply final processing
        self.post_init()
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
        This is a wrapper around the model's get_rope_index method.
        """
        return self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    # EDIT: forward function is modified to support knowledge inputs
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
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # Added for History 
        history_input_ids: torch.LongTensor = None,
        history_inputs_embeds: Optional[torch.FloatTensor] = None,
        history_attention_mask: Optional[torch.Tensor] = None,
        history_position_ids: Optional[torch.LongTensor] = None,
        history_pixel_values: Optional[torch.Tensor] = None,
        history_image_grid_thw: Optional[torch.LongTensor] = None,
        history_cache_position: Optional[torch.LongTensor] = None,
        history_past_key_values: Optional[List[torch.FloatTensor]] = None,
        # Added for Experience
        experience_input_ids: torch.LongTensor = None,
        experience_inputs_embeds: Optional[torch.FloatTensor] = None,
        experience_attention_mask: Optional[torch.Tensor] = None,
        experience_position_ids: Optional[torch.LongTensor] = None,
        experience_pixel_values: Optional[torch.Tensor] = None,
        experience_image_grid_thw: Optional[torch.LongTensor] = None,
        experience_cache_position: Optional[torch.LongTensor] = None,
        experience_past_key_values: Optional[List[torch.FloatTensor]] = None,
        file_id_list: Optional[List[str]] = None,
    ):
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:
            TODO: Add example
        """
        # Batch size is always 1
        attention_mask = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.visual.dtype)
                image_embeds, _ = self.model.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
                video_embeds, _ = self.model.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = self.model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        
        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            from transformers.utils import is_torchdynamo_compiling
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                # print('get_rope_index')
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                # print('get_rope_index else')
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        # print('inputs_embeds', inputs_embeds.shape)
        # # DEBUG: Log separator for new forward pass
        # debug_log("=" * 80)
        # debug_log("NEW FORWARD PASS")
        # debug_log("=" * 80)
        # # DEBUG: Check inputs_embeds gradients
        # debug_log(f"[DEBUG] inputs_embeds: requires_grad={inputs_embeds.requires_grad}, grad_fn={inputs_embeds.grad_fn}, shape={inputs_embeds.shape}")
        
        # INFERENCE: Prepare experience and history inputs if it's the 1st token generation
        if input_ids.shape[1] != 1:
            # EDIT: Prepare history and experience inputs  [padding and process in batch]
            concatenated_embeddings, final_position_ids = self.get_compress_history_and_experience(inputs_embeds, position_ids,
                                history_input_ids, history_inputs_embeds, history_attention_mask, history_position_ids, history_pixel_values, history_image_grid_thw, history_cache_position, history_past_key_values,
                                experience_input_ids, experience_inputs_embeds, experience_attention_mask, experience_position_ids, experience_pixel_values, experience_image_grid_thw, experience_cache_position, experience_past_key_values)
            self.shift_length = concatenated_embeddings.shape[1] - inputs_embeds.shape[1]
            print('shift_length', self.shift_length)
        else:
            concatenated_embeddings = inputs_embeds
            final_position_ids = position_ids + self.shift_length  ### remember to add knowledge length to the position ids
        concatenated_embeddings = concatenated_embeddings.to(self.model_inf.device)
        final_position_ids = final_position_ids.to(self.model_inf.device)
        
        # # DEBUG: Check concatenated_embeddings gradients
        # debug_log(f"[DEBUG] concatenated_embeddings: requires_grad={concatenated_embeddings.requires_grad}, grad_fn={concatenated_embeddings.grad_fn}, shape={concatenated_embeddings.shape}")
        
        # print('run inference model')
        # Ensure concatenated_embeddings maintains gradients
        concatenated_embeddings = concatenated_embeddings.requires_grad_(True)
        # debug_log(f"[DEBUG] After requires_grad_(True): requires_grad={concatenated_embeddings.requires_grad}, grad_fn={concatenated_embeddings.grad_fn}")
        
        # DEBUG: Check model_inf training state and parameter gradients
        # debug_log(f"[DEBUG] model_inf.training={self.model_inf.training}")
        # model_inf_has_grad_params = any(p.requires_grad for p in self.model_inf.parameters())
        # debug_log(f"[DEBUG] model_inf has params requiring grad: {model_inf_has_grad_params}")
        
        # Since model_inf already has requires_grad=True (set in train.py), 
        # we need to ensure the model is in train mode and gradients flow through.
        # The issue is that even with requires_grad=True, PyTorch might not build the graph
        # if it detects the computation doesn't need gradients. We need to explicitly
        # ensure gradient computation happens.
    
        # CRITICAL: Even though params have requires_grad=True, we need to ensure
        # the forward pass actually creates a computation graph. We'll call the
        # language model directly to get hidden_states with proper gradients.
        # First, call model to get the embeddings processed correctly
        # model_outputs = self.model_inf.model(
        #     input_ids=None,
        #     position_ids=final_position_ids,
        #     attention_mask=None,
        #     inputs_embeds=concatenated_embeddings,
        #     output_attentions=False,
        #     output_hidden_states=True,
        #     return_dict=True,
        #     cache_position=None,
        # )
        model_outputs = self.model_inf.model(
            input_ids=None,
            position_ids=final_position_ids,
            # attention_mask=attention_mask,
            attention_mask=None,
            past_key_values=past_key_values,
            inputs_embeds=concatenated_embeddings,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            cache_position=None,
        )
        # Get hidden_states from model outputs (these should have gradients if model params have requires_grad=True)
        hidden_states = model_outputs.last_hidden_state
        
        # Now compute logits using model_inf's lm_head (which has requires_grad=True)
        # This ensures the computation graph connects properly
        logits = self.model_inf.lm_head(hidden_states)
        
        # Create a dummy outputs object for compatibility with rest of code
        from transformers.modeling_outputs import BaseModelOutputWithPast
        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=model_outputs.past_key_values,
        )
        
        # DEBUG: Check outputs after calling model directly
        # debug_log(f"[DEBUG] After calling model_inf.model directly")
        # debug_log(f"[DEBUG] hidden_states from model: requires_grad={hidden_states.requires_grad}, grad_fn={hidden_states.grad_fn}, shape={hidden_states.shape}")
        # debug_log(f"[DEBUG] logits from lm_head: requires_grad={logits.requires_grad}, grad_fn={logits.grad_fn}, shape={logits.shape}")
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits_sequence_length = logits.shape[1]
            labels_sequence_length = labels.shape[1]
            remove_length = logits_sequence_length - labels_sequence_length
            # debug_log(f"[DEBUG] remove_length: {remove_length}")
            logits = logits.float()
            logits = logits[:, remove_length:, :] #QEDIT: remove the first 24 q-former tokens ##NOTE: change to 80 if 10
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            
            # DEBUG: Check before loss computation
            # debug_log(f"[DEBUG] shift_logits: requires_grad={shift_logits.requires_grad}, grad_fn={shift_logits.grad_fn}, shape={shift_logits.shape}")
            
            loss = loss_fct(shift_logits, shift_labels)
            # debug_log(f"[DEBUG] loss: requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}, value={loss.item() if loss.numel() == 1 else 'multi-element'}")

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            rope_deltas=self.rope_deltas,
        )

    def get_compress_history_and_experience(self, 
                               inputs_embeds, position_ids,
                               history_input_ids, history_inputs_embeds, history_attention_mask, history_position_ids, history_pixel_values, history_image_grid_thw, history_cache_position, history_past_key_values,
                               experience_input_ids, experience_inputs_embeds, experience_attention_mask, experience_position_ids, experience_pixel_values, experience_image_grid_thw, experience_cache_position, experience_past_key_values):        
        
        # Initialize compressed embeddings list
        compressed_embeddings_list = []
        
        # # Process History inputs if provided
        # if history_input_ids is not None and len(history_input_ids) > 0:
        #     print("Processing history inputs...")
        #     compressed_history = self._process_and_compress_inputs(
        #         history_input_ids, history_inputs_embeds, history_attention_mask, 
        #         history_position_ids, history_pixel_values, history_image_grid_thw, 
        #         history_cache_position, history_past_key_values, "history"
        #     )
        #     if compressed_history is not None:
        #         compressed_embeddings_list.append(compressed_history)
        
        # Process Experience inputs if provided
        if experience_input_ids is not None and len(experience_input_ids) > 0:
            print("Processing experience inputs...")
            print(f"experience_input_ids: {len(experience_input_ids)}, {experience_input_ids[0].shape}")
            compressed_experience = self._process_and_compress_inputs(
                experience_input_ids, experience_inputs_embeds, experience_attention_mask, 
                experience_position_ids, experience_pixel_values, experience_image_grid_thw, 
                experience_cache_position, experience_past_key_values, "experience"
            )
            if compressed_experience is not None:
                compressed_embeddings_list.append(compressed_experience)
        
        # If no compressed embeddings, return original inputs
        if not compressed_embeddings_list:
            return inputs_embeds, position_ids
        
        # Concatenate all compressed embeddings
        # DEBUG: Check compressed embeddings before concatenation
        # for i, comp_emb in enumerate(compressed_embeddings_list):
            # debug_log(f"[DEBUG] compressed_embeddings_list[{i}]: requires_grad={comp_emb.requires_grad}, grad_fn={comp_emb.grad_fn}, shape={comp_emb.shape}")
        
        all_compressed_embeddings = torch.cat(compressed_embeddings_list, dim=1)
        # debug_log(f"[DEBUG] all_compressed_embeddings (after cat): requires_grad={all_compressed_embeddings.requires_grad}, grad_fn={all_compressed_embeddings.grad_fn}, shape={all_compressed_embeddings.shape}")
        
        # if all_compressed_embeddings.shape[1] > 24:
        #     all_compressed_embeddings = all_compressed_embeddings[:, :24, :]
        # elif all_compressed_embeddings.shape[1] < 24:
        #     all_compressed_embeddings = torch.cat([all_compressed_embeddings, all_compressed_embeddings[:, :24 - all_compressed_embeddings.shape[1], :]], dim=1)
        # debug_log(f"[DEBUG] all_compressed_embeddings (after padding/truncating): requires_grad={all_compressed_embeddings.requires_grad}, grad_fn={all_compressed_embeddings.grad_fn}, shape={all_compressed_embeddings.shape}")
            
        print(f"All compressed embeddings shape: {all_compressed_embeddings.shape}")
        
        # Calculate the position id for compressed_embeddings + raw_embedding, and concatenate
        concatenated_embeddings, final_position_ids = get_qformer_position_id(all_compressed_embeddings, inputs_embeds, position_ids)
        # debug_log(f"[DEBUG] concatenated_embeddings (from get_qformer_position_id): requires_grad={concatenated_embeddings.requires_grad}, grad_fn={concatenated_embeddings.grad_fn}, shape={concatenated_embeddings.shape}")
        
        return concatenated_embeddings, final_position_ids
    
    def _process_and_compress_inputs(self, input_ids, inputs_embeds, attention_mask, 
                                   position_ids, pixel_values, image_grid_thw, 
                                   cache_position, past_key_values, input_type):
        """Helper function to process and compress a specific type of inputs (history/experience)"""
        compressed_list = []
        for k_input_id, k_pixel_value, k_image_grid_thw in zip(input_ids, pixel_values, image_grid_thw):
            k_input_ids = k_input_id.unsqueeze(0).to(self.model.device)#
            k_pixel_values = k_pixel_value.unsqueeze(0).to(self.model.device)#
            k_image_grid_thw = k_image_grid_thw.to(self.model.device)#.unsqueeze(0)
            # print(f"k_input_ids: {k_input_ids.shape}, k_pixel_values: {k_pixel_values.shape}, k_image_grid_thw: {k_image_grid_thw.shape}")
            if k_input_ids.dim() == 1:
                k_input_ids = k_input_ids.unsqueeze(0)
                k_pixel_values = k_pixel_values.unsqueeze(0)
                # k_image_grid_thw = k_image_grid_thw.unsqueeze(0)
            if attention_mask is not None:
                k_attention_mask = attention_mask[0].to(self.model.device)
            else:
                k_attention_mask = torch.ones_like(k_input_ids, device=self.model.device)
            # print(self.tokenizer.decode(k_input_ids[0], skip_special_tokens=True))
            # Get embeddings
            try:
                k_inputs_embeds = self.model.get_input_embeddings()(k_input_ids)
            except Exception as e:
                k_inputs_embeds = self.model.get_input_embeddings()(k_input_ids.to(torch.long))
            
            # Process pixel values if provided
            if k_pixel_values is not None:
                k_pixel_values = k_pixel_values.type(self.model.visual.dtype)
                k_image_embeds, _ = self.model.get_image_features(k_pixel_values, k_image_grid_thw)
                k_image_embeds = torch.cat(k_image_embeds, dim=0).to(k_inputs_embeds.device, k_inputs_embeds.dtype)
                k_image_mask, _ = self.model.get_placeholder_mask(
                    k_input_ids, inputs_embeds=k_inputs_embeds, image_features=k_image_embeds
                )
                k_inputs_embeds = k_inputs_embeds.masked_scatter(k_image_mask, k_image_embeds)
            
            if k_attention_mask is not None:
                k_attention_mask = k_attention_mask.to(self.model.device)
                
            # Get Position IDs
            k_position_ids = None
            from transformers.utils import is_torchdynamo_compiling
            if position_ids is None and (k_attention_mask is None or k_attention_mask.ndim == 2):
                # calculate RoPE index once per generation in the pre-fill stage only
                prefill_compiled_stage = is_torchdynamo_compiling() and (
                    (k_input_ids is not None and k_input_ids.shape[0] != 1) if k_input_ids is not None else False
                    or (k_inputs_embeds is not None and k_inputs_embeds.shape[1] != 1)
                )
                prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                    (cache_position is not None and cache_position[0] == 0)
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                )
                if (prefill_compiled_stage or prefill_noncompiled_stage) or self.knowledge_rope_deltas is None:
                    k_position_ids, knowledge_rope_deltas = self.get_rope_index(
                        k_input_ids,
                        k_image_grid_thw,
                        None,
                        k_attention_mask,
                    )
                    self.knowledge_rope_deltas = knowledge_rope_deltas
                # then use the prev pre-calculated rope-deltas to get the correct position ids
                else:
                    batch_size, seq_length, _ = k_inputs_embeds.shape
                    delta = (
                        (cache_position[0] + self.knowledge_rope_deltas).to(k_inputs_embeds.device)
                        if cache_position is not None
                        else 0
                    )
                    k_position_ids = torch.arange(seq_length, device=k_inputs_embeds.device)
                    k_position_ids = k_position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:  # otherwise `deltas` is an int `0`
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    k_position_ids = k_position_ids.add(delta)
                    k_position_ids = k_position_ids.unsqueeze(0).expand(3, -1, -1)   
            
            # DEBUG: Check inputs before model forward
            # debug_log(f"[DEBUG] _process_and_compress_inputs ({input_type}) - k_inputs_embeds: requires_grad={k_inputs_embeds.requires_grad}, grad_fn={k_inputs_embeds.grad_fn}, shape={k_inputs_embeds.shape}")
            
            # Run through model to get hidden states
            self.model.config.use_cache = True
            k_outputs = self.model(
                input_ids=None,
                position_ids=k_position_ids,
                attention_mask=k_attention_mask,
                inputs_embeds=k_inputs_embeds,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                cache_position=None,
            )
            # Use last_hidden_state (equivalent to hidden_states[-1] when available)
            single_hidden_state = k_outputs.last_hidden_state
            single_attention_mask = k_attention_mask # Shape: [1, seq_len]
            compressed = self.knowledge_processor(single_hidden_state, single_attention_mask)
            compressed_list.append(compressed.to(k_input_id.device))
        
        # Concatenate along sequence length dimension (dim=1)
        compressed_inputs_embeds = torch.cat(compressed_list, dim=1)
        # debug_log(f"[DEBUG] _process_and_compress_inputs ({input_type}) - compressed_inputs_embeds (after cat): requires_grad={compressed_inputs_embeds.requires_grad}, grad_fn={compressed_inputs_embeds.grad_fn}, shape={compressed_inputs_embeds.shape}")
        print(f"compressed_{input_type}_inputs_embeds: {compressed_inputs_embeds.shape}")
        
        return compressed_inputs_embeds

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )
        # Additional inputs for history
        model_inputs["history_input_ids"] = kwargs.get("history_input_ids", None)
        model_inputs["history_inputs_embeds"] = kwargs.get("history_inputs_embeds", None)
        model_inputs["history_attention_mask"] = kwargs.get("history_attention_mask", None)
        model_inputs["history_pixel_values"] = kwargs.get("history_pixel_values", None)
        model_inputs["history_image_grid_thw"] = kwargs.get("history_image_grid_thw", None)
        model_inputs["history_position_ids"] = kwargs.get("history_position_ids", None)
        model_inputs["history_cache_position"] = kwargs.get("history_cache_position", None)
        model_inputs["history_past_key_values"] = kwargs.get("history_past_key_values", None)
        
        # Additional inputs for experience
        model_inputs["experience_input_ids"] = kwargs.get("experience_input_ids", None)
        model_inputs["experience_inputs_embeds"] = kwargs.get("experience_inputs_embeds", None)
        model_inputs["experience_attention_mask"] = kwargs.get("experience_attention_mask", None)
        model_inputs["experience_pixel_values"] = kwargs.get("experience_pixel_values", None)
        model_inputs["experience_image_grid_thw"] = kwargs.get("experience_image_grid_thw", None)
        model_inputs["experience_position_ids"] = kwargs.get("experience_position_ids", None)
        model_inputs["experience_cache_position"] = kwargs.get("experience_cache_position", None)
        model_inputs["experience_past_key_values"] = kwargs.get("experience_past_key_values", None)
        
        # Qwen3VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        return self.model._get_image_nums_and_video_nums(input_ids, inputs_embeds)

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(
                            f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead."
                        )
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(tensor, lengths=lengths, repeat_times=expand_size)
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # input_ids is required for expanding visual inputs
        # If input_ids is unavailable, visual inputs will not be used; therefore, there is no need to expand visual inputs.
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


def get_qformer_position_id(q_former_list, raw_embeddings, raw_position_ids):
    q_former_outputs = q_former_list
    # 1. Get dimensions
    batch_size = q_former_outputs.shape[0]
    q_former_total_len = q_former_outputs.shape[1]
    raw_len = raw_embeddings.shape[1]
    total_len = q_former_total_len + raw_len
    device = q_former_outputs.device
    dtype = raw_embeddings.dtype

    # 2. Create sequential position IDs for Q-former outputs
    q_former_position_ids = torch.zeros(
        3,  # 3 dimensions for Qwen3VL
        batch_size,
        q_former_total_len,
        dtype=raw_embeddings.dtype,
        device=raw_embeddings.device
    )

    # Fill sequentially (all 3 dimensions get same values for Q-former outputs)
    for dim in range(3):
        q_former_position_ids[dim] = torch.arange(
            q_former_total_len, 
            device=device
        ).unsqueeze(0).expand(batch_size, -1)

    # 3. Ensure position_ids_4 starts after Q-former position IDs
    raw_position_ids = raw_position_ids.to(raw_embeddings.device)
    max_q_former_pos = q_former_position_ids.max().to(raw_position_ids.device)
    min_raw_pos = raw_position_ids.min()

    # Apply offset if needed
    raw_position_ids = raw_position_ids.clone().to(raw_embeddings.device)
    if min_raw_pos <= max_q_former_pos:
        offset = max_q_former_pos + 1 - min_raw_pos
        offset = offset.to(dtype=raw_position_ids.dtype)
        raw_position_ids += offset

    # 4. Concatenate position IDs
    final_position_ids = torch.cat([q_former_position_ids, raw_position_ids], dim=2)

    # 5. Concatenate embeddings
    q_former_outputs = q_former_outputs.to(raw_embeddings.device)
    concatenated_embeddings = torch.cat([
        q_former_outputs,
        raw_embeddings
    ], dim=1)

    # 6. Calculate final mrope_position_deltas
    # This is typically the max position ID + 1 - total sequence length
    max_position = final_position_ids.max()
    final_mrope_deltas = (max_position + 1 - total_len).unsqueeze(0).expand(batch_size, 1).to(q_former_outputs.device)
    
    return concatenated_embeddings, final_position_ids

