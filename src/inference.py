"""
Author: Gabe Grand

Tools for running inference of a pretrained ControlNet model.
Adapted from gradio_scribble2image.py from the original authors.

"""

import sys

sys.path.append("..")
from share import *

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from PIL import Image
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import os
import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

from cldm.model import create_model, load_state_dict
from tutorial_dataset import MyDataset
from share import *

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=1
device = 'cuda:0'

A_PROMPT_DEFAULT = "best quality, extremely detailed"
N_PROMPT_DEFAULT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"


def run_sampler(
    model,
    input_image: np.ndarray,
    prompt: str,
    num_samples: int = 1,
    image_resolution: int = 256,
    seed: int = -1,
    a_prompt: str = A_PROMPT_DEFAULT,
    n_prompt: str = N_PROMPT_DEFAULT,
    guess_mode=False,
    strength=1.0,
    ddim_steps=20,
    eta=0.0,
    scale=9.0,
    show_progress: bool = True,
):
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.to(device)

        ddim_sampler = DDIMSampler(model)

        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = img.copy()

        control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            show_progress=show_progress,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]

        return results
    


if __name__ == '__main__':
    # prompt = 'The light source in the image is the sun, which is shining through the window and creating a shadow on the wall. The shadow is placed on the wall, and the sunlight is reflecting off the ceiling, creating a visually appealing scene. The light direction is coming from the window, and the reflections on the ceiling add depth and interest to the room. The overall atmosphere of the image is bright and inviting, with the sunlight creating a warm and welcoming ambiance.'
    # prompt = 'The light source in the image is a lamp, which is placed on the bedside table. The lamp is turned on, and it is casting a shadow on the wall. The light is shining on the bed, creating a cold atmosphere. The reflections on the wall indicate that the light is coming from a window, which is located on the right side of the room. The window allows natural light to enter the room, creating a bright and well-lit environment.'
    # prompt = 'The light source in the image is a lamp, which is placed on the bedside table, but the lamp is turned off, creating a totally dark room.'
    prompt = '\"Light Source\": Natural light from a window on the right side of the image.\n\n\"Light Direction\": The light is coming from the right side, illuminating the bed on the right and casting light across the room.\n\n\"Shadows\": Shadows are present on the left side of the beds, indicating the light is stronger on the right.\n\n\"Reflections\": There are no significant reflections visible in the image.'
    sketch = cv2.imread('../stylitgan_test_new/alb/0_0.jpg')
    # sketch = cv2.imread('../stylitgan_train_new/alb/14_4.jpg')
    sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)

    model = create_model('./models/cldm_v21.yaml').cpu()

    model.load_state_dict(load_state_dict(".//lightning_logs/version_1/checkpoints/epoch=86-step=173826.ckpt", location=device))
    model = model.to(device)

    results = run_sampler(model, sketch, prompt)

    img = Image.fromarray(results[0], "RGB")
    img.save('test_unseen.jpg')