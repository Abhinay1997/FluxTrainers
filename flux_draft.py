## IMPORTS
import math
import wandb
import torch
import numpy as np
import pandas as pd
import bitsandbytes as bnb
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from torch.utils.data import Dataset, DataLoader
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers.models import AutoencoderKL
from diffusers.models import FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from imscore.preference.model import SiglipPreferenceScorer
from imscore.pickscore.model import PickScorer

## CONFIG
# MODEL_PATH = 'black-forest-labs/FLUX.1-schnell'
MODEL_PATH = 'black-forest-labs/FLUX.1-dev'
DTYPE = torch.bfloat16
LORA_RANK = 32
DEVICE = "cuda"
NUM_INFERENCE_STEPS = 28
target_modules = {}
LORA_ALPHA = 0.50 * LORA_RANK
LORA_DROPOUT = 0.0
TARGET_MODULES = ["to_q", "to_k", "to_v"]
PARQUET_FILE = "table.parquet"  # Replace with your file path
COLUMN_NAME = "prompt"  # Replace with the name of the column
BATCH_SIZE = 1  # Adjust batch size as needed
LR = 1e-4
NUM_EPOCHS = 2
EVAL_STEPS = 50
SAVE_STEPS = 100
LORA_SAVE_PATH = './'
EVAL_IMAGES_SAVE_PATH = './'
DRAFT_LV = True
## Expt notes about changes to be logged to wandb
NOTES = "Test run"

transformer_lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    init_lora_weights="gaussian",
    target_modules=TARGET_MODULES,
)

import logging
def get_logger(name): 
    logger = logging.getLogger(name)
    if not logger.handlers: 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("train.log"), logging.StreamHandler()])
    return logger
logger = get_logger("train")

## MODEL LOAD
tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH, subfolder='tokenizer', torch_dtype=DTYPE)
text_encoder = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder='text_encoder', torch_dtype=DTYPE).to(DEVICE)
tokenizer_2 = T5TokenizerFast.from_pretrained(MODEL_PATH, subfolder='tokenizer_2', torch_dtype=DTYPE)
text_encoder_2 = T5EncoderModel.from_pretrained(MODEL_PATH, subfolder='text_encoder_2', torch_dtype=DTYPE).to(DEVICE)
vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder='vae', torch_dtype=DTYPE).to(DEVICE)
# rm = SiglipPreferenceScorer.from_pretrained("RE-N-Y/pickscore-siglip").to(DEVICE)
rm = PickScorer("yuvalkirstain/PickScore_v1").to(DEVICE)
rm.eval()  # Set reward model to evaluation mode
for param in rm.parameters():  # Disable gradients for reward model
    param.requires_grad = False
    
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) #8
# Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
# by the patch size. So the vae scale factor is multiplied by the patch size to account for this
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

# TRANSFORMER LOAD
transformer = FluxTransformer2DModel.from_pretrained(MODEL_PATH, subfolder='transformer', torch_dtype=DTYPE).to(DEVICE)

assert not text_encoder.training
assert not text_encoder_2.training
assert not transformer.training
assert not vae.training

transformer.add_adapter(transformer_lora_config)

print(f"Trainable params count", sum(p.numel() for p in transformer.parameters() if p.requires_grad))

## DATASET LOAD
class ParquetDataset(Dataset):
    def __init__(self, parquet_file, column_name):
        # Load the Parquet file
        df = pd.read_parquet(parquet_file)
        # Extract the specified column as a numpy array
        self.data = df[column_name].values
          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = ParquetDataset(PARQUET_FILE, COLUMN_NAME)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

## UTILITIES
@torch.no_grad()
def t5_encode(text_encoder, tokenizer, device, prompt):
    max_sequence_length = 512
    prompt = [prompt] if isinstance(prompt, str) else prompt

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds

@torch.no_grad()
def clip_encode(text_encoder, tokenizer, device, prompt):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    return prompt_embeds

def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents
     
def decode_latent(vae, vae_scale_factor, height, width, latents):
    latents = _unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    latents = vae.decode(latents, return_dict=False)[0]
    return latents

@torch.no_grad()
def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

@torch.no_grad()
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

@torch.no_grad()
def prepare_latents(vae_scale_factor, num_channels_latents, device, dtype, generator, height, width, batch_size):
    lh = 2 * (int(height) // (vae_scale_factor * 2))
    lw = 2 * (int(width) // (vae_scale_factor * 2))

    shape = (batch_size, num_channels_latents, lh, lw)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    latents = _pack_latents(latents, batch_size, num_channels_latents, lh, lw)
    latent_image_ids = _prepare_latent_image_ids(batch_size, lh // 2, lw // 2, device, dtype)
    return latents, latent_image_ids

@torch.no_grad()
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

@torch.no_grad()
def prepare_data(transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, device, dtype, vae_scale_factor, batch_size, num_inference_steps, prompts):
    num_channels_latents = transformer.config.in_channels // 4
    pooled_prompt_embeds = clip_encode(text_encoder, tokenizer, device, prompts)
    prompt_embeds = t5_encode(text_encoder_2, tokenizer_2, device, prompts)
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    height, width = (512, 512)
    sigmas = torch.tensor(np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)).to(device)
    latents, latent_img_ids = prepare_latents(vae_scale_factor=vae_scale_factor, num_channels_latents=num_channels_latents, device=device, dtype=dtype, generator=None, height=height, width=width, batch_size=batch_size)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
    )
    shifted_sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1) ** 1.0)
    shifted_sigmas = torch.cat([shifted_sigmas, torch.zeros(1, device=sigmas.device)])
    timesteps = sigmas * 1000
    guidance_scale = 3.5
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
    guidance = guidance.expand(latents.shape[0])
    # guidance=None
    data = {
        "prompts": prompts,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
        "image_ids": latent_img_ids,
        "latents": latents,
        "latent_image_ids": latent_img_ids,
        "guidance": guidance,
        "sigmas": shifted_sigmas,
        "timesteps": timesteps,
        "height": height,
        "width":width,
    }
    return data

def train_step(transformer, reward_model, optimizer, vae, vae_scale_factor, data):
    timesteps = data["timesteps"]
    sigmas = data["sigmas"]
    latents = data["latents"]
    prompts = data["prompts"]
    for i,t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        ## If not last timestep, don't accumulate grad
        if i != len(timesteps) - 1:
            with torch.no_grad():
                noise_pred = transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=data["guidance"],
                        pooled_projections=data["pooled_prompt_embeds"],
                        encoder_hidden_states=data["prompt_embeds"],
                        txt_ids=data["text_ids"],
                        img_ids=data["latent_image_ids"],
                        joint_attention_kwargs={},
                        return_dict=False,
                    )[0]
                latents = latents + (sigmas[i+1] - sigmas[i]) * noise_pred
        else:
            with torch.enable_grad():
                noise_pred = transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=data["guidance"],
                        pooled_projections=data["pooled_prompt_embeds"],
                        encoder_hidden_states=data["prompt_embeds"],
                        txt_ids=data["text_ids"],
                        img_ids=data["latent_image_ids"],
                        joint_attention_kwargs={},
                        return_dict=False,
                    )[0]
                latents = latents + (sigmas[i+1] - sigmas[i]) * noise_pred
                latents = decode_latent(vae=vae, vae_scale_factor=vae_scale_factor, height=data["height"], width=data["width"], latents=latents)
                latents = image_processor.denormalize(latents)
                ## check latents are in range 0-1
                print(f"Latents range: min={latents.min().item():.4f}, max={latents.max().item():.4f}")
                loss = -(reward_model.score(latents, prompts).mean())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(compiled_transformer.parameters(), 5.0)
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in transformer.parameters() if p.grad is not None]))
                optimizer.step()
    
                # optimizer.zero_grad()  # Zero gradients once at the start
                # loss = -reward_model.score(latents, prompts).mean()
                # loss.backward()
                # if DRAFT_LV:
                #     for i in range(2):
                #         epsilon = torch.randn_like(latents)
                #         new_latents = vae.normalize(latents.clone().detach())
                #         new_latents = vae.encode(new_latents) # [0, 1] to [-1, 1] and then to latent space
                #         x_1 = 0.9 * new_latents + 0.1 * epsilon ## add noise
                #         x_0_hat = x_1 - 0.1 * noise_pred ## recalculate the new latents
                          ## decode latents and denormalize
                #         loss_i = -reward_model.score(x_0_hat, prompts).mean() ## find the new reward loss
                #         loss_i.backward()  # Accumulate gradients
                #         torch.nn.utils.clip_grad_norm_(transformer.parameters(), 5.0)
                # grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in transformer.parameters() if p.grad is not None]))
                # optimizer.step()
                                
                    
    
    return loss, grad_norm

@torch.no_grad()
def eval_step(transformer, vae, vae_scale_factor, image_processor, data):
    timesteps = data["timesteps"]
    sigmas = data["sigmas"]
    latents = data["latents"]
    for i,t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        with torch.no_grad():
            noise_pred = transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=data["guidance"],
                    pooled_projections=data["pooled_prompt_embeds"],
                    encoder_hidden_states=data["prompt_embeds"],
                    txt_ids=data["text_ids"],
                    img_ids=data["latent_image_ids"],
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]
            latents = latents + (sigmas[i+1] - sigmas[i]) * noise_pred
    latents = decode_latent(vae=vae, vae_scale_factor=vae_scale_factor, height=data["height"], width=data["width"], latents=latents)
    return image_processor.postprocess(latents, output_type="pil")


## TRAINING LOOP
optimizer = bnb.optim.Adam8bit(transformer.parameters(),lr=LR)

run = wandb.init(
    entity="devarintinagasaiabhinay-self",
    project="Flux DRaFT LoRA",
    config={
        "learning_rate": LR,
        "architecture": MODEL_PATH,
        "dataset": "NagaSaiAbhinay/MJ-Prompts",
        "epochs": NUM_EPOCHS,
        "lora_config": transformer_lora_config,
        "batch_size" : BATCH_SIZE,
        "reward_model": type(rm),
        "notes" : NOTES,
        "optimizer": type(optimizer),
    },
)

if torch.cuda.is_available():
    # Get the currently allocated memory
    allocated_memory_bytes = torch.cuda.memory_allocated()
    allocated_memory_gb = allocated_memory_bytes / (1024**3)

    # Get the memory reserved by the caching allocator
    reserved_memory_bytes = torch.cuda.memory_reserved()
    reserved_memory_gb = reserved_memory_bytes / (1024**3)

    print(f"Allocated GPU Memory: {allocated_memory_gb:.2f} GB")
    print(f"Reserved GPU Memory: {reserved_memory_gb:.2f} GB")
else:
    print("CUDA is not available. No GPU detected or configured.")


# Prepare one fixed batch for overfitting
fixed_prompts = next(iter(dataloader))  # Get the first batch
data = prepare_data(
    transformer=transformer,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    device=DEVICE,
    dtype=DTYPE,
    vae_scale_factor=vae_scale_factor,
    batch_size=BATCH_SIZE,
    num_inference_steps=NUM_INFERENCE_STEPS,
    prompts=fixed_prompts
)
eval_data = prepare_data(
    transformer=transformer,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    device=DEVICE,
    dtype=DTYPE,
    vae_scale_factor=vae_scale_factor,
    batch_size=BATCH_SIZE,
    num_inference_steps=NUM_INFERENCE_STEPS,
    prompts=["People waiting in line at a checkout counter in America"]
)

compiled_transformer = torch.compile(transformer, dynamic=False, fullgraph=True)

for epoch in range(NUM_EPOCHS):
    transformer.train()
    transformer.enable_gradient_checkpointing()

    for step in range(10000):  # Or however many steps you want
        if step in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]:
            transformer.eval()
            with torch.no_grad():
                images = eval_step(
                    transformer=compiled_transformer,
                    vae=vae,
                    vae_scale_factor=vae_scale_factor,
                    image_processor=image_processor,
                    data=eval_data
                )
                run.log({"images": [wandb.Image(img) for img in images]})
                images[0].save(f"image_{step}.jpg")
            transformer.train()

        loss, grad_norm = train_step(
            transformer=compiled_transformer,
            reward_model=rm,
            optimizer=optimizer,
            vae=vae,
            vae_scale_factor=vae_scale_factor,
            data=eval_data
        )

        run.log({"loss": loss, "grad_norm": grad_norm})

run.finish()


## TODO: Uncomment later
# for epoch in range(NUM_EPOCHS):
#     transformer.train()
#     transformer.enable_gradient_checkpointing()
#     compiled_transformer = torch.compile(transformer, dynamic=False, fullgraph=True)
#     for step, batch in enumerate(dataloader):
#         data = prepare_data(transformer=transformer, text_encoder=text_encoder, text_encoder_2=text_encoder_2, tokenizer=tokenizer, tokenizer_2=tokenizer_2, device=DEVICE, dtype=DTYPE, vae_scale_factor=vae_scale_factor, batch_size=BATCH_SIZE, num_inference_steps=NUM_INFERENCE_STEPS, prompts=batch)
#         if step == 0 or (step > EVAL_STEPS and step % EVAL_STEPS) == 0:
#             transformer.eval()
#             images = eval_step(transformer=compiled_transformer, vae=vae, vae_scale_factor=vae_scale_factor, image_processor=image_processor, data=data)
#             run.log({"images": [wandb.Image(img) for img in images]})
#             images[0].save(f"image_{step}.jpg")
#             transformer.train()
    
#         if (step > SAVE_STEPS) and (step % SAVE_STEPS == 0):
#             FluxPipeline.save_lora_weights(
#                output_dir=OUTPUT_DIR,
#                transformer_lora_layers_to_save = get_peft_model_state_dict(transformer),
#                weight_name=WEIGHT_NAME_PREFIX + f'_{step}.safetensors',
#                safe_serialization=True
#             )

#         loss, grad_norm = train_step(transformer=compiled_transformer, reward_model=rm, optimizer=optimizer, vae=vae, vae_scale_factor=vae_scale_factor, data=data)
#         run.log({"loss": loss, "grad_norm": grad_norm})
# run.finish()


## See if you can use batch size 4 and 1024 x 1024 on a A100 80G
## Add draft-lv
## Check different reward models



# 1. Setup accelerate and save correctly
# 2. Setup eval batch properly
# 3. Run it on a large batch on an A100 and monitor the loss till it hits
# 4. Add DRaFT-LV
