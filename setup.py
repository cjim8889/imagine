import os
current_file_path = os.path.abspath(__file__)
current_file_directory = os.path.dirname(current_file_path)
import torch
from diffusers import DiffusionPipeline
from components.codeformer import setup_codeformer
import torch
import logging

logging.basicConfig(level=logging.INFO)

upsampler, codeformer_net = setup_codeformer()
model = DiffusionPipeline.from_pretrained(
    "cjim8889/AllysMix3",
    cache_dir = os.path.join(current_file_directory, "model"),
    custom_pipeline="lpw_stable_diffusion",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    requires_safety_checker=False, 
    safety_checker=lambda images, **kwargs: [images, [False] * len(images)],
)

logging.info("Setup model")