import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    AutoModel
)
from accelerate import load_checkpoint_and_dispatch
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
        "quant": "nf4",
        "device_map": "balanced"
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler,
        "dtype": torch.bfloat16,
        "quant": "fp16",
        "device_map": "sequential"
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler,
        "dtype": torch.float16,
        "quant": "int4",
        "device_map": "auto"
    }
}

def log_vram(msg: str):
    print(f"{msg} (used {torch.cuda.memory_allocated()/1e9:.2f}GB VRAM)")

def load_models(model_type: str):
    config = MODEL_CONFIGS[model_type]
    
    # Configure 4-bit quantization
    bnb_config = None
    if config["quant"] == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config["dtype"],
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_NAME,
        truncation=True,
        model_max_length=77
    )
    log_vram("✅ Tokenizer initialized")

    # Load text encoder with proper device mapping
    text_encoder = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        device_map=config["device_map"],
        torch_dtype=config["dtype"],
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2"
    )
    log_vram("✅ Text encoder loaded")

    # Load transformer with sharding
    transformer = load_checkpoint_and_dispatch(
        HiDreamImageTransformer2DModel.from_pretrained(
            config["path"],
            subfolder="transformer",
            torch_dtype=config["dtype"],
            device_map=config["device_map"],
            no_split_module_classes=["TransformerBlock"]
        ),
        checkpoint=config["path"],
        device_map=config["device_map"]
    )
    log_vram("✅ Transformer loaded")

    # Initialize pipeline
    pipe = HiDreamImagePipeline.from_pretrained(
        config["path"],
        scheduler=config["scheduler"](
            num_train_timesteps=1000,
            shift=config["shift"],
            use_dynamic_shifting=False
        ),
        tokenizer_4=tokenizer,
        text_encoder_4=text_encoder,
        transformer=transformer,
        torch_dtype=config["dtype"],
        device_map=config["device_map"]
    )
    log_vram("✅ Pipeline initialized")

    # Memory optimization
    pipe.enable_sequential_cpu_offload()
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
    
    # Process prompt with proper truncation
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
