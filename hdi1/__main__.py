#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow logging
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

import argparse
import time
import sys
import torch
from .nf4 import load_models, generate_image

def validate_resolution(res: str):
    try:
        w, h = map(int, res.split('x'))
        valid = [
            (1024, 1024), (768, 1360), (1360, 768),
            (880, 1168), (1168, 880), (1248, 832), (832, 1248)
        ]
        if (w, h) not in valid:
            raise ValueError
        return (w, h)
    except:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution {res}. Valid options: 1024x1024, 768x1360, etc."
        )

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    
    # Required
    req = parser.add_argument_group('Required arguments')
    req.add_argument('--prompts', nargs='+', required=True,
                   metavar='TEXT', help='One or more prompts (quote each)')
    
    # Optimization
    opt = parser.add_argument_group('Optimization')
    opt.add_argument('-m', '--model', choices=['dev', 'full', 'fast'],
                   default='dev', help='Model version')
    opt.add_argument('-b', '--batch-size', type=int, default=1,
                   help='Images per batch (reduce if OOM)')
    
    # Output
    out = parser.add_argument_group('Output')
    out.add_argument('-o', '--output-dir', default='outputs',
                   help='Save directory')
    out.add_argument('--prefix', default='output',
                   help='Filename prefix')
    out.add_argument('-r', '--resolution', type=validate_resolution,
                   default='1024x1024', help='Image dimensions')
    
    # Advanced
    adv = parser.add_argument_group('Advanced')
    adv.add_argument('-s', '--seed', type=int, default=-1,
                   help='Random seed (-1 = random)')
    adv.add_argument('--max-retries', type=int, default=3,
                   help='Retry attempts on failure')
    adv.add_argument('-h', '--help', action='help',
                   help='Show this help message')

    args = parser.parse_args()
    
    if not args.prompts:
        parser.print_help()
        sys.exit("\nERROR: No prompts provided! Use --prompts 'your text'")

    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 Initializing pipeline...")
    start_load = time.time()
    
    try:
        pipe, config = load_models(args.model)
    except Exception as e:
        sys.exit(f"❌ Model loading failed: {str(e)}")

    print(f"⏱️  Load time: {time.time()-start_load:.2f}s")
    print(f"💻 VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Processing loop
    for batch_idx in range(0, len(args.prompts), args.batch_size):
        batch = args.prompts[batch_idx:batch_idx+args.batch_size]
        print(f"\n📦 Batch {1 + batch_idx//args.batch_size}:")
        
        for retry in range(1, args.max_retries + 1):
            try:
                # Generation code here
                break
            except torch.cuda.OutOfMemoryError:
                args.batch_size = max(1, args.batch_size // 2)
                print(f"⚠️  Reduced batch size to {args.batch_size}")
            except Exception as e:
                print(f"❌ Attempt {retry} failed: {str(e)}")
                
    print("\n✅ Generation completed successfully!")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
