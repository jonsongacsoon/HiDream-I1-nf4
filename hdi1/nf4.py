import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from accelerate import init_empty_weights
import warnings
from . import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from .schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler

MODEL_PREFIX = "azaneko"
LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler,
        "dtype": torch.float16,
        "quant": "nf4"
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler,
        "dtype": torch.bfloat16,
        "quant": "fp16"
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler,
        "dtype": torch.float16,
        "quant": "int4"
    }
}

def log_vram(msg: str):
    print(f"{msg} (used {torch.cuda.memory_allocated()/1e9:.2f}GB VRAM)")

def load_models(model_type: str):
    config = MODEL_CONFIGS[model_type]
    
    with init_empty_weights():
        # Load tokenizer with truncation
        tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL_NAME,
            truncation=True,
            model_max_length=77
        )
        log_vram("✅ Tokenizer initialized")

        # Quantized text encoder
        text_encoder = LlamaForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            device_map="auto",
            torch_dtype=config["dtype"],
            attn_implementation="flash_attention_2",
            load_in_4bit=(config["quant"] == "int4"),
            bnb_4bit_compute_dtype=config["dtype"]
        )
        log_vram("✅ Text encoder loaded")

        # Transformer with memory optimization
        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            config["path"],
            subfolder="transformer",
            torch_dtype=config["dtype"],
            device_map="sequential"
        )
        log_vram("✅ Transformer loaded")

        # Pipeline configuration
        pipe = HiDreamImagePipeline.from_pretrained(
            config["path"],
            scheduler=config["scheduler"](
                num_train_timesteps=1000,
                shift=config["shift"],
                use_dynamic_shifting=False
            ),
            tokenizer_4=tokenizer,
            text_encoder_4=text_encoder,
            torch_dtype=config["dtype"],
            variant=config["quant"]
        )
        pipe.transformer = transformer
        log_vram("✅ Pipeline initialized")

    # Memory optimization techniques
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    
    return pipe, config

@torch.inference_mode()
def generate_image(
    pipe: HiDreamImagePipeline,
    model_type: str,
    prompt: str,
    resolution: tuple[int, int],
    seed: int,
    negative_prompt: str = "text, watermark, deformed, blurry, low quality"
):
    config = MODEL_CONFIGS[model_type]
    width, height = resolution
    seed = seed if seed != -1 else torch.randint(0, 1000000, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Token truncation with proper padding
    inputs = pipe.tokenizer_4(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True
    )
    
    images = pipe(
        prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        guidance_scale=config["guidance_scale"],
        num_inference_steps=config["num_inference_steps"],
        num_images_per_prompt=1,
        generator=generator,
        prompt_embeds=inputs.input_ids.to(pipe.device),
        output_type="pil"
    ).images
    
    return images[0], seed
