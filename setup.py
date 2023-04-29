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
    "model",
    local_files_only=True,
    custom_pipeline="lpw_stable_diffusion",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

logging.info("Setup model")