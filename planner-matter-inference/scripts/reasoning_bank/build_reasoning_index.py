#!/usr/bin/env python3
"""
Build FAISS index for Reasoning Bank (text-only or multimodal).

This script pre-builds the CLIP embeddings and FAISS index for the reasoning bank,
so that during inference the agent only needs to load the pre-built index without
loading CLIP models onto GPU.

Usage:
    # Text-only mode
    python scripts/build_reasoning_index.py \
        --bank_path memory/reasoning_bank_Amazon.jsonl \
        --index_base memory_index/reasoning_bank_text \
        --mode text

    # Multimodal mode (text + images)
    python scripts/build_reasoning_index.py \
        --bank_path memory/reasoning_bank_Amazon.jsonl \
        --index_base memory_index/reasoning_bank_mm \
        --mode multimodal \
        --device cuda
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.reasoning_bank import ReasoningBank


def main():
    parser = argparse.ArgumentParser(description='Build FAISS index for Reasoning Bank')
    parser.add_argument(
        '--bank_path',
        type=str,
        required=True,
        help='Path to reasoning bank JSONL file'
    )
    parser.add_argument(
        '--index_base',
        type=str,
        required=True,
        help='Base path for FAISS index (without extension)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['text', 'multimodal'],
        default='text',
        help='Index mode: text-only or multimodal (text + images)'
    )
    parser.add_argument(
        '--clip_model',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='CLIP model name for embeddings'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to use for CLIP model (cuda or cpu)'
    )
    parser.add_argument(
        '--force_rebuild',
        action='store_true',
        help='Force rebuild even if index exists'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.bank_path):
        print(f"Error: Bank file not found: {args.bank_path}")
        sys.exit(1)
    
    # Create index directory
    index_dir = os.path.dirname(args.index_base)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)
    
    use_multimodal = (args.mode == 'multimodal')
    
    # Check if index already exists
    index_path = f"{args.index_base}.faiss"
    meta_path = f"{args.index_base}.json"
    
    if os.path.exists(index_path) and os.path.exists(meta_path) and not args.force_rebuild:
        print(f"Index already exists at {index_path}")
        print("Use --force_rebuild to rebuild anyway")
        sys.exit(0)
    
    print("=" * 60)
    print("Reasoning Bank Index Builder")
    print("=" * 60)
    print(f"Bank path:    {args.bank_path}")
    print(f"Index base:   {args.index_base}")
    print(f"Mode:         {args.mode}")
    print(f"CLIP model:   {args.clip_model}")
    print(f"Device:       {args.device}")
    print("=" * 60)
    print()
    
    # Force device by temporarily modifying environment
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("Note: CUDA_VISIBLE_DEVICES set to '' to force CPU mode")
    
    try:
        print(f"Loading reasoning bank from {args.bank_path}...")
        
        # Initialize ReasoningBank (this will build the index)
        bank = ReasoningBank(
            bank_path=args.bank_path,
            index_base_path=args.index_base,
            clip_model_name=args.clip_model,
            use_multimodal=use_multimodal
        )
        
        print(f"✓ Successfully built index with {len(bank.items)} items")
        print(f"✓ Index saved to: {index_path}")
        print(f"✓ Metadata saved to: {meta_path}")
        
        # Verify the index
        if bank.index is not None:
            print(f"✓ Index dimension: {bank.index.d}")
            print(f"✓ Index size: {bank.index.ntotal}")
        else:
            print("✗ Warning: Index is None")
            sys.exit(1)
        
        print()
        print("=" * 60)
        print("Index building completed successfully!")
        print("=" * 60)
        
        # Show usage example
        print()
        print("You can now use this index in your evaluation:")
        print(f"  bash scripts/runners/run_agent.sh \\")
        print(f"    --eval_type webvoyager \\")
        print(f"    --domain Amazon \\")
        print(f"    --model qwen2.5-vl \\")
        print(f"    --use_reasoning_bank \\")
        if use_multimodal:
            print(f"    --reasoning_bank_multimodal \\")
        print(f"    --reasoning_bank_path {args.bank_path} \\")
        print(f"    --reasoning_index_base {args.index_base} \\")
        print(f"    --reasoning_top_k 2")
        
    except Exception as e:
        print(f"✗ Error building index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
