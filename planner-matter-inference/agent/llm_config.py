"""LLM configuration for different model types."""
import argparse
import base64
import os
import sys
from io import BytesIO
from typing import Dict, List

import torch
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from PIL import Image
from transformers import AutoProcessor

class VLLMModel:
    """vLLM model wrapper that call API directly"""
    
    def __init__(self, model_name: str, server_url: str, api_key: str = "EMPTY", **kwargs):
        self.model_name = model_name
        self.server_url = server_url
        self.api_key = api_key
        self.client = OpenAI(
            base_url=server_url,
            api_key=api_key
        )
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_tokens = kwargs.get('max_tokens', 2048)
    
    def chat(self, messages: List[Dict], stream: bool = False, **kwargs):
        """Chat with the model using simplified message format"""
        # Prepare function calling parameters
        call_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get('temperature', self.temperature),
            "top_p": kwargs.get('top_p', self.top_p),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "n": kwargs.get('n', 1),
        }
        
        # Call the model
        response = self.client.chat.completions.create(**call_params)
        
        if stream:
            return response, None, None
        else:
            return response.choices[0].message, None, None

class CoMEMModel:
    """CoMEM model wrapper that load checkpoint and support continuous memory"""
    
    def __init__(self, model_name: str, checkpoint_path: str, **kwargs):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.temperature = kwargs.get('temperature', 0.1)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_tokens = 10**4
        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        
        # Import the custom model class
        train_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "planner-matter-sft"))
        if not os.path.isdir(train_repo_root):
            raise FileNotFoundError(
                f"Could not find planner-matter-sft at {train_repo_root}. "
                "Expected this repo layout: <root>/planner-matter-inference and <root>/planner-matter-sft."
            )
        if train_repo_root not in sys.path:
            sys.path.insert(0, train_repo_root)
        if 'qwen3' in self.checkpoint_path:
            # Qwen3 VL with continuous memory
            print('Using 3 VL model')
            print(f"Loading model from checkpoint: {self.checkpoint_path}")
            from src_agent.training.qwen3VL_compressor import Qwen3VLForConditionalGeneration_new
            self.model = Qwen3VLForConditionalGeneration_new.from_pretrained(
                self.checkpoint_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Qwen2.5 VL with continuous memory
            if 'full-sft' in self.model_name or 'rl' in self.model_name:
                print('Using 2_5 full-sft model')
                # class that is used to load the finetuned model with prepared continuous memory embedding, and without on-the-fly compression module
                from src_agent.training.qwenVL_inference_full_sft import Qwen2_5_VLForConditionalGeneration_new
            else:
                # class that is used to load the normal model with continuous memory compression module
                print('Using 2_5 normal model')
                from src_agent.training.qwenVL_inference import Qwen2_5_VLForConditionalGeneration_new
            self.model = Qwen2_5_VLForConditionalGeneration_new.from_pretrained(
                self.checkpoint_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                low_cpu_mem_usage=True
            )
        # # Load model_inf weights from checkpoint with key remapping
        # # Checkpoint has old structure: model_inf.model.layers.* and model_inf.visual.*
        # # Current model expects: model_inf.model.language_model.layers.* and model_inf.model.visual.*
        # self._load_model_inf_from_checkpoint()
    
    def _load_model_inf_from_checkpoint(self):
        """Load model_inf and knowledge_processor weights from checkpoint with key remapping.
        
        The checkpoint uses old key structure:
        - model_inf.model.layers.* -> model_inf.model.language_model.layers.*
        - model_inf.model.embed_tokens.* -> model_inf.model.language_model.embed_tokens.*
        - model_inf.model.norm.* -> model_inf.model.language_model.norm.*
        - model_inf.visual.* -> model_inf.model.visual.*
        - knowledge_processor.* -> knowledge_processor.* (Q-Former, no remapping needed)
        """
        from safetensors import safe_open

        safetensor_files = [f for f in os.listdir(self.checkpoint_path) if f.endswith(".safetensors")]
        if not safetensor_files:
            print("No safetensor files found in checkpoint, skipping weight loading")
            return
        
        # Collect model_inf and knowledge_processor weights from checkpoint
        checkpoint_model_inf_weights = {}
        checkpoint_knowledge_processor_weights = {}
        for sf_file in safetensor_files:
            
            sf_path = os.path.join(self.checkpoint_path, sf_file)
            with safe_open(sf_path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    if key.startswith("model_inf."):
                        checkpoint_model_inf_weights[key] = f.get_tensor(key)
                    elif key.startswith("knowledge_processor."):
                        checkpoint_knowledge_processor_weights[key] = f.get_tensor(key)
        
        print(f"Found {len(checkpoint_model_inf_weights)} model_inf weights and {len(checkpoint_knowledge_processor_weights)} knowledge_processor weights in checkpoint")
        
        if not checkpoint_model_inf_weights and not checkpoint_knowledge_processor_weights:
            print("No model_inf or knowledge_processor weights found in checkpoint")
            return
        
        # Remap model_inf keys from old structure to new structure
        remapped_weights = {}
        for old_key, tensor in checkpoint_model_inf_weights.items():
            new_key = old_key
            if old_key.startswith('model_inf.model.layers.'):
                new_key = old_key.replace('model_inf.model.layers.', 'model_inf.model.language_model.layers.')
            elif old_key.startswith('model_inf.model.embed_tokens'):
                new_key = old_key.replace('model_inf.model.embed_tokens', 'model_inf.model.language_model.embed_tokens')
            elif old_key.startswith('model_inf.model.norm'):
                new_key = old_key.replace('model_inf.model.norm', 'model_inf.model.language_model.norm')
            elif old_key.startswith('model_inf.visual.'):
                new_key = old_key.replace('model_inf.visual.', 'model_inf.model.visual.')
            # model_inf.lm_head stays the same
            remapped_weights[new_key] = tensor
        
        # Add knowledge_processor weights (no remapping needed)
        for key, tensor in checkpoint_knowledge_processor_weights.items():
            remapped_weights[key] = tensor
        
        # Load remapped weights into model
        model_state_dict = self.model.state_dict()
        loaded_count = 0
        kp_loaded_count = 0
        missing_keys = []
        
        # Debug: Check if knowledge_processor keys exist in model
        kp_keys_in_model = [k for k in model_state_dict.keys() if 'knowledge_processor' in k]
        print(f"[Debug] Model has {len(kp_keys_in_model)} knowledge_processor keys")
        if kp_keys_in_model:
            print(f"[Debug] First 3 model kp keys: {kp_keys_in_model[:3]}")
        
        for key, tensor in remapped_weights.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == tensor.shape:
                    model_state_dict[key] = tensor.to(model_state_dict[key].dtype)
                    loaded_count += 1
                    if key.startswith('knowledge_processor.'):
                        kp_loaded_count += 1
                        print(f"[Debug] Loaded Q-Former key: {key}")
                else:
                    print(f"Shape mismatch for {key}: checkpoint {tensor.shape} vs model {model_state_dict[key].shape}")
            else:
                missing_keys.append(key)
                if key.startswith('knowledge_processor.'):
                    print(f"[Debug] Q-Former key NOT FOUND in model: {key}")
        
        # Load the updated state dict
        self.model.load_state_dict(model_state_dict, strict=False)
        print(f"Loaded {loaded_count} weights from checkpoint (including {kp_loaded_count} Q-Former weights)")
        if missing_keys:
            print(f"Warning: {len(missing_keys)} keys not found in model (first 5): {missing_keys[:5]}")
        
    def process_vision_info(self, conversation):
        """Process vision information from conversation."""
        image_inputs = []
        for message in conversation:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if item['type'] == 'image_url':
                        image_url = item['image_url']['url']
                        image_bytes = base64.b64decode(image_url.split(',')[1])
                        image = Image.open(BytesIO(image_bytes))
                        image_inputs.append(image)
        
        return image_inputs

    def knowledge_processor_vlm(self, processor, inputs, texts=None, images=None, tokenizer=None, formatted_prompt=None):
        """Process experience information for VLM"""
        # Default tokens for image processing
        DEFAULT_IM_START_TOKEN = "<|im_start|>"
        DEFAULT_IM_END_TOKEN = "<|im_end|>"
        DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
        VISION_START_TOKEN = "<|vision_start|>"
        VISION_END_TOKEN = "<|vision_end|>"
        
        all_experience_input_ids = [] 
        all_experience_pixel_values = []
        all_experience_image_grid_thw = []
        for trajectory_actions, trajectory_images in zip(texts, images):
            trajectory_text = ""
            trajectory_image = []
            for action, image_base64 in zip(trajectory_actions, trajectory_images):
                if isinstance(image_base64, dict) and image_base64.get('url', '').startswith('data:image/png;base64,'):
                    image_bytes = base64.b64decode(image_base64.get('url', '').split(',')[1])
                elif isinstance(image_base64, str) and image_base64.startswith('data:image/png;base64,'):
                    image_bytes = base64.b64decode(image_base64.split(',')[1])
                else:
                    image_bytes = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_bytes))
                trajectory_image.append(image)
                trajectory_text += f"{DEFAULT_IM_START_TOKEN}user\n{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}{action}{DEFAULT_IM_END_TOKEN}\n"
            if trajectory_image:
                e_inputs = processor(text=[trajectory_text], images=trajectory_image, padding=False, return_tensors='pt')
                e_input_ids = e_inputs['input_ids'].squeeze(0)
                e_pixel_values = e_inputs['pixel_values']
                e_image_grid_thw = e_inputs['image_grid_thw']
                all_experience_pixel_values.append(e_pixel_values)
                all_experience_image_grid_thw.append(e_image_grid_thw)
            else:
                e_input_ids = processor.tokenizer(trajectory_text, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
            
            all_experience_input_ids.append(e_input_ids)

        
        inputs['experience_input_ids'] = all_experience_input_ids
        inputs['experience_pixel_values'] = all_experience_pixel_values
        inputs['experience_image_grid_thw'] = all_experience_image_grid_thw
        
        return inputs
    
    def generate_response_with_experience(self, image=None, prompt=None, experience_texts=None, experience_images=None, file_id_list=None, conversation=None, experience_embedding=None):
        """Generate response with experience texts and images"""
        
        if not conversation:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ],
                }
            ]
        
        formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # print('formatted_prompt:', formatted_prompt)
        image_inputs = self.process_vision_info(conversation)
        # print('image_number:', len(image_inputs))
        
        device = next(self.model.parameters()).device
        inputs = self.processor(
            text=[formatted_prompt],
            images=image_inputs,
            return_tensors="pt",
        ).to(device)
        
        if file_id_list is not None:
            inputs["file_id_list"] = file_id_list
            inputs_with_experience = inputs
        else:
            inputs_with_experience = self.knowledge_processor_vlm(
                processor=self.processor,
                inputs=inputs,
                texts=experience_texts,
                images=experience_images,
                tokenizer=self.tokenizer,
                formatted_prompt=formatted_prompt,
            ).to(device)
        
        generated_ids = self.model.generate(
            **inputs_with_experience, 
            max_new_tokens=self.max_tokens,
            use_cache=True, 
            temperature=self.temperature,
            top_p=self.top_p,
        )
        
        input_ids = inputs_with_experience["input_ids"]
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0]
        return output_text
    
    def chat(self, messages: List[Dict], stream: bool = False, 
             experience_texts=None, experience_images=None, file_id_list=None):
        """Chat with the model using transformers with experience support"""
        if stream:
            raise NotImplementedError("Streaming not yet implemented for transformers models")
        # Check if experience data is provided
        has_experience = False
        if experience_texts is not None:
            # Check if any experience text is not empty
            has_experience = any(len(text_list) > 0 for text_list in experience_texts)
        if experience_images is not None:
            # Check if any experience image list is not empty
            has_experience = any(len(img_list) > 0 for img_list in experience_images)
        if file_id_list is not None:
            has_experience = True

        if not has_experience:
            print("No experience data provided, falling back to VLLM API...")
            vllm_model = VLLMModel(
                model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                server_url="http://localhost:8000/v1",
                api_key="EMPTY",
                temperature=0.2,
                top_p=0.9,
                max_tokens=self.max_tokens,
            )
            return vllm_model.chat(messages, stream=False)

        else:
            print("Generating response with experience...")
            print('file_id_list:', file_id_list)
            # Generate response with experience
            response_text = self.generate_response_with_experience(
                experience_texts=experience_texts,
                experience_images=experience_images,
                file_id_list=file_id_list,
                conversation=messages
            )
            
            # Create OpenAI-style response
            return ChatCompletionMessage(
                role="assistant",
                content=response_text,
                function_call=None,
                tool_calls=None
            ), None, None

class TransformersModel:
    """Transformers model wrapper that loads checkpoint and without continuous memory"""
    
    def __init__(self, checkpoint_path: str, **kwargs):
        """
        Initialize Transformers model from checkpoint.
        
        Args:
            checkpoint_path: Path to the HuggingFace checkpoint directory
            **kwargs: Additional arguments (temperature, top_p, max_tokens, etc.)
        """
        self.checkpoint_path = checkpoint_path
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_tokens = kwargs.get('max_tokens', 500)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Transformers model from checkpoint: {checkpoint_path}")
        # Load model
        print("Loading model weights...")
        if 'qwen3' in checkpoint_path:
            print("Using Qwen3 VL model")
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                attn_implementation="flash_attention_2",
                device_map="auto" if self.device == 'cuda' else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            self.tokenizer = self.processor.tokenizer
        else:
            print("Using Qwen2.5 VL model")
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                attn_implementation="flash_attention_2",
                device_map="auto" if self.device == 'cuda' else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            self.tokenizer = self.processor.tokenizer
        self.model.eval()
        
        print("✓ Model loaded successfully!")
    
    
    def process_vision_info(self, conversation):
        """Process vision information from conversation."""
        image_inputs = []
        for message in conversation:
            if isinstance(message['content'], list):
                for item in message['content']:
                    if item['type'] == 'image_url':
                        image_url = item['image_url']['url']
                        image_bytes = base64.b64decode(image_url.split(',')[1])
                        image = Image.open(BytesIO(image_bytes))
                        image_inputs.append(image)
        
        return image_inputs

    def chat(self, messages: List[Dict], stream: bool = False, **kwargs):
        """
        Chat with the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream responses (not yet implemented)
            **kwargs: Additional generation parameters (temperature, top_p, max_tokens)
        
        Returns:
            Tuple of (ChatCompletionMessage, None, None) to match other model interfaces
        """
        # Apply chat template
        formatted_prompt = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False
        )
        # Extract images from messages
        image_inputs = self.process_vision_info(messages)
        if image_inputs:
            inputs = self.processor(
                text=[formatted_prompt],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[formatted_prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

        # Generation parameters
        generation_kwargs = {
            'max_new_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'top_p': kwargs.get('top_p', self.top_p),
            'do_sample': kwargs.get('temperature', self.temperature) > 0,
            'use_cache': True,
        }
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        print('output_text', output_text)
        # Return in OpenAI-style format
        return ChatCompletionMessage(
            role="assistant",
            content=output_text,
            function_call=None,
            tool_calls=None
        ), None, None


def create_transformers_model(checkpoint_path: str, **kwargs) -> TransformersModel:
    """
    Create a Transformers model instance from checkpoint.
    
    Args:
        checkpoint_path: Path to the HuggingFace checkpoint directory
        **kwargs: Additional arguments (temperature, top_p, max_tokens, device)
    
    Returns:
        TransformersModel instance
    """
    return TransformersModel(checkpoint_path=checkpoint_path, **kwargs)

def create_vllm_model(args: argparse.Namespace, model_name: str = None) -> VLLMModel:
    """Create a vLLM model instance"""
    if model_name is None:
        model_name = args.model
    
    model_name_map = {
        'qwen2.5-vl': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'qwen2-vl': 'Qwen/Qwen2-VL-7B-Instruct',
        'qwen3-vl':  'Qwen/Qwen3-VL-8B-Instruct',
        'qwen2.5-vl-3b': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'ui-tars': 'ByteDance-Seed/UI-TARS-1.5-7B',
        'ui-ins-7b': 'Tongyi-MiA/UI-Ins-7B',
        'ui-ins-32b': 'Tongyi-MiA/UI-Ins-32B',
        'websight': 'WenyiWU0111/websight-7B_combined',
        'cogagent': 'zai-org/cogagent-9b-20241220',
        'qwen2.5-vl-32b': 'qwen/qwen2.5-vl-32b-instruct',
        'glm': 'thudm/glm-4.1v-9b-thinking',
        'gemini': 'google/gemini-2.5-pro',
        'claude': 'anthropic/claude-sonnet-4',
        'gpt-4o': 'openai/gpt-4o',
    }
    model_server_map = {
        'qwen2.5-vl': 'http://localhost:8000/v1',
        'ui-tars': 'http://localhost:8001/v1',
        'qwen2-vl': 'http://localhost:8002/v1',
        'websight': 'http://localhost:8003/v1',
        'qwen2.5-vl-3b': 'http://localhost:8004/v1',
        'cogagent': 'http://localhost:8005/v1',
        'ui-ins-7b': 'http://localhost:8006/v1',
        'qwen3-vl': 'http://localhost:8007/v1',
        'qwen2.5-vl-32b': 'https://openrouter.ai/api/v1',
        'glm': 'https://openrouter.ai/api/v1',
        'gemini': 'https://openrouter.ai/api/v1',
        'claude': 'https://openrouter.ai/api/v1',
        'gpt-4o': 'https://openrouter.ai/api/v1',
    }

    model_name_ = model_name_map.get(model_name, model_name)
    server_url = model_server_map.get(model_name, 'http://localhost:8000/v1')
    api_key = os.environ.get('OPEN_ROUTER_API_KEY') or os.environ.get('OPENROUTER_API_KEY') or 'EMPTY'
    print('model_name', model_name_)
    print('server_url', server_url)
    if api_key and api_key != 'EMPTY':
        print('api_key', f"{api_key[:6]}...{api_key[-4:]}")
    else:
        print('api_key', api_key)
    
    return VLLMModel(
        model_name=model_name_,
        server_url=server_url,
        api_key=api_key,
        temperature=0.2,
        top_p=0.9,
        max_tokens=256,
    )


def create_comem_model(args: argparse.Namespace) -> CoMEMModel:
    """Create a CoMEM model instance"""
    model_name = args.model
    checkpoint_path = getattr(args, "checkpoint_path", model_name)
    
    return CoMEMModel(model_name=model_name, checkpoint_path=checkpoint_path)

def create_model(args: argparse.Namespace):
    """Create a model instance based on model type"""
    if args.use_continuous_memory:
        return create_comem_model(args)
    else:
        if 'rl' in getattr(args, 'model', '') or 'sft' in getattr(args, 'model', ''):
            return create_transformers_model(getattr(args, 'checkpoint_path', ''))
        else:
            # Default to vLLM
            return create_vllm_model(args)


def load_grounding_model_vllm(args: argparse.Namespace):
    """
    Load grounding model using vLLM server with OpenAI client.
    
    Args:
        args: Arguments object
        
    Returns:
        Grounding model client
    """
    # Use grounding_model_name from args, default to ui-ins-7b
    grounding_model_name = getattr(args, 'grounding_model_name', 'ui-ins-7b')
    grounding_model = create_vllm_model(args, model_name=grounding_model_name)
    return grounding_model

def load_tool_llm(args: argparse.Namespace, model_name=None) -> VLLMModel:
    """Load tool LLM"""
    # Use tool_model_name from args if available, otherwise use provided model_name or default
    if model_name is None:
        model_name = getattr(args, 'tool_model_name', 'qwen2.5-vl')
    tool_model = create_vllm_model(args, model_name=model_name)
    return tool_model

