
# Librerías generales 

import matplotlib.pyplot as plt

import os
import json

import numpy as np

# Para SDXL-Lightning

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# for cuda availability 
from torch import cuda


# Para  Stable Diffusion 2.1

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler



pipe = None

# PARA EL MODELO SDXL-Lightning
def load_model_SDXL(ckpt_number = 4, use_cuda:bool=True):

    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{ckpt_number}step_unet.safetensors" # Use the correct ckpt for your step setting!

    # we check for cuda availability
    if cuda.is_available() and use_cuda:
        device_ = "cuda"
    else:
        device_ = "cpu"
        # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device_, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device_))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device_)

        # Ensure sampler uses "trailing" timesteps.
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe


def generate_image_SDXL(prompt, save_name, ckpt_number = 4):

    #pipe = load_model_SDXL(ckpt_number)

    fig, ax = plt.subplots()
    # generated_image = np.array(pipe(prompt, num_inference_steps=ckpt_number, guidance_scale=0).images[0])
    img = ax.imshow(np.array(pipe(prompt, num_inference_steps=ckpt_number, guidance_scale=0).images[0]))
    # title =  f"{prompt[75:]}[...]"
    ax.set_title(f"{prompt[75:]}[...]")
    ax.axis("off")
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# PARA EL MODELO Stable Diffusion 2.1
def load_model_stable_diffusion(ckpt_number = 4, use_cuda:int=True):
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if cuda.is_available() and use_cuda:
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    return pipe

def generate_image_stable_diffusion(prompt, save_name, ckpt_number = 4):
    fig, ax = plt.subplots()
    # generated_image = np.array(pipe(prompt, num_inference_steps=ckpt_number, guidance_scale=0).images[0])
    img = ax.imshow(np.array(pipe(prompt, num_inference_steps=ckpt_number, guidance_scale=0).images[0]))
    # title =  f"{prompt[75:]}[...]"
    ax.set_title(f"{prompt[75:]}[...]")
    ax.axis("off")
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


"""
Ejemplo del formato para el JSON en la generación de imagen :

### Formato JSON ---> prompts.json
{
    concepto_1: [prompt1, prompt2, ... , promptN-1, promptN],
    concepto_2: [prompt1, prompt2, ... , promptN-1, promptN],
    ...
    concepto_N: [prompt1, prompt2, ... , promptN-1, promptN]

}

"""

def read_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    # regresamos una lista con cada concepto y la lista de sus prompts 
    return [(concepto, prompts) for concepto, prompts in data.items()]

def generate_images(json_file, model, LLM_model, ckpt_number = 4):
    data = read_json(json_file)
    total_images_generated_counter = 0
    for concepto, prompts in data:
        for i, prompt in enumerate(prompts):
            save_name = os.path.join("SPI_2025_CIDESI", "experiments", f"experimento_{LLM_model}_{model}", "imagenes", f"{concepto}_{prompt}_{i+1}.png" )
            if model == "SDXL":
                if pipe is None:
                    load_model_SDXL(ckpt_number)
                generate_image_SDXL(prompt, save_name, ckpt_number)
                total_images_generated_counter += 1
            elif model == "SD2":
                if pipe is None:
                    load_model_stable_diffusion(ckpt_number)
                generate_image_stable_diffusion(prompt, save_name, ckpt_number)
                total_images_generated_counter += 1
            else:
                print("Modelo no reconocido")
    print(f"¡Proceso terminado! \n Número total de imágenes generadas: {total_images_generated_counter}")