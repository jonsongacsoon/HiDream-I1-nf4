from .nf4 import *
import argparse
import time
import logging
import os

def main():
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("prompts", nargs="+", type=str, help="Prompts to generate images from")
    parser.add_argument("-m", "--model", type=str, default="dev",
                        help="Model to use",
                        choices=["dev", "full", "fast"])
    parser.add_argument("-s", "--seed", type=int, default=-1, 
                        help="Seed for generation")
    parser.add_argument("-r", "--res", type=str, default="1024x1024", 
                        help="Resolution for generation", 
                        choices=["1024x1024", "768x1360", "1360x768", "880x1168", "1168x880", "1248x832", "832x1248"])
    parser.add_argument("-o", "--output-dir", type=str, default="outputs",
                        help="Output directory for images")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once
    print(f"Loading model {args.model}...")
    pipe, _ = load_models(args.model)
    print("Model loaded successfully!")

    resolution = tuple(map(int, args.res.strip().split("x")))

    # Process all prompts
    for i, prompt in enumerate(args.prompts, 1):
        st = time.time()
        output_path = os.path.join(args.output_dir, f"output{i}.png")
        
        image, seed = generate_image(pipe, args.model, prompt, resolution, args.seed)
        image.save(output_path)

        print(f"Image {i} saved to {output_path}, elapsed time: {time.time() - st:.2f} seconds")
        print(f"Seed used: {seed}")

if __name__ == "__main__":
    main()
