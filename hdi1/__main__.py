#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
import warnings
warnings.filterwarnings("ignore")

import argparse
import time
import sys
import torch
from pathlib import Path
from .nf4 import load_models, generate_image

def validate_resolution(res_str):
    try:
        width, height = map(int, res_str.split('x'))
        valid = {
            (1024, 1024), (768, 1360), (1360, 768),
            (880, 1168), (1168, 880), (1248, 832), (832, 1248)
        }
        if (width, height) not in valid:
            raise ValueError
        return (width, height)
    except:
        raise argparse.ArgumentTypeError(f"Invalid resolution: {res_str}")

def print_header():
    print("\n" + "="*60)
    print(f"HiDream Image Generator".center(60))
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}".center(60))
    print(f"GPU: {torch.cuda.get_device_name(0)}".center(60))
    print("="*60 + "\n")

def main():
    print_header()
    
    parser = argparse.ArgumentParser(
        description="Optimized Image Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    
    required = parser.add_argument_group('Required')
    required.add_argument('--prompts', nargs='+', required=True,
                        metavar='TEXT', help='Input prompts')
    
    model = parser.add_argument_group('Model')
    model.add_argument('-m', '--model', choices=['dev', 'full', 'fast'],
                     default='dev', help='Model version')
    
    output = parser.add_argument_group('Output')
    output.add_argument('-o', '--output-dir', default='outputs',
                      help='Save directory')
    output.add_argument('--prefix', default='output',
                      help='Filename prefix')
    output.add_argument('-r', '--resolution', type=validate_resolution,
                      default='1024x1024', help='Image size')
    
    advanced = parser.add_argument_group('Advanced')
    advanced.add_argument('-s', '--seed', type=int, default=-1,
                        help='Random seed')
    advanced.add_argument('--max-retries', type=int, default=3,
                        help='Failure retries')
    advanced.add_argument('-h', '--help', action='help',
                        help='Show help')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.prompts:
        sys.exit("❌ No prompts provided")
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("🔧 Initializing pipeline...")
    try:
        start = time.time()
        pipe, config = load_models(args.model)
        print(f"✅ Loaded in {time.time()-start:.2f}s")
        print(f"💻 VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB\n")
    except Exception as e:
        sys.exit(f"❌ Load failed: {str(e)}")
    
    # Process prompts
    print(f"🚀 Generating {len(args.prompts)} images")
    for idx, prompt in enumerate(args.prompts, 1):
        short_prompt = prompt if len(prompt) < 50 else f"{prompt[:47]}..."
        print(f"\n🎨 [{idx}/{len(args.prompts)}] {short_prompt}")
        
        for attempt in range(1, args.max_retries+1):
            try:
                start = time.time()
                image, seed = generate_image(
                    pipe=pipe,
                    model_type=args.model,
                    prompt=prompt,
                    resolution=args.resolution,
                    seed=args.seed
                )
                filename = f"{args.prefix}_{idx}_{seed}.png"
                image.save(output_dir / filename)
                print(f"✅ Saved {filename} ({time.time()-start:.2f}s)")
                break
            except Exception as e:
                print(f"⚠️ Attempt {attempt} failed: {str(e)}")
                if attempt == args.max_retries:
                    print(f"❌ Failed after {args.max_retries} attempts")
    
    print("\n🏁 Generation complete")
    print(f"💾 Outputs: {output_dir.resolve()}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
        sys.exit(1)
