import argparse
import time
import os
import torch
from .nf4 import load_models, generate_image

def main():
    parser = argparse.ArgumentParser(description="Optimized Image Generation Pipeline")
    
    # Required
    parser.add_argument("--prompts", nargs="+", required=True,
                      help="List of prompts (enclose each in quotes)")
    
    # Optimization params
    parser.add_argument("-m", "--model", choices=["dev", "full", "fast"],
                      default="dev", help="Model version")
    parser.add_argument("-b", "--batch-size", type=int, default=1,
                      help="Images per batch (VRAM dependent)")
    parser.add_argument("-s", "--seed", type=int, default=-1,
                      help="Seed for reproducibility")
    parser.add_argument("-r", "--resolution", default="1024x1024",
                      choices=["1024x1024", "768x1360", "1360x768", 
                              "880x1168", "1168x880", "1248x832", "832x1248"],
                      help="Output resolution")
    parser.add_argument("-o", "--output-dir", default="outputs",
                      help="Directory to save images")
    parser.add_argument("--prefix", default="output",
                      help="Filename prefix")
    parser.add_argument("--max-retries", type=int, default=3,
                      help="Retry attempts on failure")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    resolution = tuple(map(int, args.resolution.split("x")))

    # Load model once
    print("🚀 Initializing pipeline...")
    start_load = time.time()
    pipe, config = load_models(args.model)
    print(f"⏱️  Model loaded in {time.time()-start_load:.2f}s")
    print(f"💻 VRAM Allocation: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Batch processing
    for batch_idx in range(0, len(args.prompts), args.batch_size):
        batch = args.prompts[batch_idx:batch_idx+args.batch_size]
        
        print(f"\n📦 Processing batch {batch_idx//args.batch_size+1}/"
             f"{len(args.prompts)//args.batch_size+1}")
        
        for retry in range(args.max_retries):
            try:
                start_batch = time.time()
                results = []
                
                for prompt in batch:
                    print(f"🎨 Generating: {prompt[:60]}...")
                    start_gen = time.time()
                    
                    image, seed = generate_image(
                        pipe=pipe,
                        model_type=args.model,
                        prompt=prompt,
                        resolution=resolution,
                        seed=args.seed
                    )
                    
                    filename = f"{args.prefix}_{seed}.png"
                    image.save(os.path.join(args.output_dir, filename))
                    results.append((filename, time.time()-start_gen))
                
                # Print batch summary
                print(f"\n✅ Batch completed in {time.time()-start_batch:.2f}s")
                for name, duration in results:
                    print(f"  - {name} ({duration:.2f}s)")
                break
                
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️  OOM error, reducing batch size from {args.batch_size} to {max(1, args.batch_size//2)}")
                args.batch_size = max(1, args.batch_size//2)
            except Exception as e:
                print(f"❌ Attempt {retry+1} failed: {str(e)}")
                if retry == args.max_retries-1:
                    print(f"🔥 Failed batch after {args.max_retries} attempts")
                
    print("\n📊 Final VRAM Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
