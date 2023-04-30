import os
current_file_path = os.path.abspath(__file__)
current_file_directory = os.path.dirname(current_file_path)
import runpod, torch
import numpy as np
import cv2
from PIL import Image
from diffusers import DiffusionPipeline
from data import TextToImageParams
from components.codeformer import setup_codeformer, codeformer_inference
import torch
import time
from io import BytesIO
import boto3
import PIL
import logging

logging.basicConfig(level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    endpoint_url=os.environ['S3_ENDPOINT_URL']
)

upsampler, codeformer_net = setup_codeformer()
model = DiffusionPipeline.from_pretrained(
    os.path.join(current_file_directory, "model"),
    custom_pipeline="lpw_stable_diffusion",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)
available_schedulers = {v.__name__: v for v in model.scheduler.compatibles}
current_scheduler = model.scheduler.__class__.__name__

def upload_to_s3(bucket, key, image):
    # Save PIL image to in-memory file
    in_mem_file = BytesIO()
    image.save(in_mem_file, format="JPEG")
    in_mem_file.seek(0)

    # Upload the in-memory file to S3
    s3_client.upload_fileobj(in_mem_file, bucket, key, ExtraArgs={'ContentType': 'image/jpeg', 'ACL': 'public-read'})

def generate_public_url(bucket, key):
    return f"{os.environ['S3_ENDPOINT_URL']}/{bucket}/{key}"

def inference(
        prompt: str, 
        negative_prompt: str, 
        num_inference_steps=3, 
        width=512, 
        height=512,
        guidance_scale=8,
        generator=None,
        face_restore=False,
        upsample=False,
        codeformer_fidelity=0.5,
        ) -> PIL.Image.Image:
    image = model(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    if face_restore:
        image, _ = codeformer_inference(
            upsampler=upsampler,
            codeformer_net=codeformer_net,
            image=image,
            background_enhance=True,
            face_upsample=True,
            upscale=4,
            codeformer_fidelity=codeformer_fidelity,
        )

        return image
    
    if upsample:
        img = np.array(image)
        # Convert the image from RGB to BGR format (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = upsampler.enhance(img, outscale=4)[0]
        restored_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the numpy.ndarray image to a PIL.Image.Image object
        image = Image.fromarray(restored_img)
    
    return image
                                                                                                                                                                                                           
def switch_scheduler(scheduler_name):
    global current_scheduler
    if scheduler_name == current_scheduler:
        return
    
    is_karras = False
    if "Karras" in scheduler_name:
        scheduler_name = scheduler_name.replace("Karras", "")
        is_karras = True

    if scheduler_name in available_schedulers:
        model.scheduler = available_schedulers[scheduler_name].from_config(model.scheduler.config)
        if is_karras:
            model.scheduler.use_karras_sigmas = True

        current_scheduler = scheduler_name
        logging.info(f"Switched to {scheduler_name}")
        return
    else:
        logging.info(f"Error: {scheduler_name} is not a valid scheduler")
        return

def handler(event):
    job_input = event["input"]
    job_input = {k: v for k, v in job_input.items() if v is not None}

    job_input = TextToImageParams(**job_input)
    switch_scheduler(job_input.scheduler)
    
    logging.info(f"Starting inference for job {job_input['id']}")
    start_time = time.time()

    image = inference(
        prompt=job_input.prompt,
        negative_prompt=job_input.negative_prompt,
        num_inference_steps=job_input.num_inference_steps,
        width=job_input.width,
        height=job_input.height,
        guidance_scale=job_input.guidance_scale,
        generator=torch.cuda.manual_seed(int(job_input.seed)) if job_input.seed is not None else None,
        face_restore=job_input.face_restore,
        upsample=job_input.upsample,
        codeformer_fidelity=job_input.codeformer_fidelity,
    )

    elapsed_time = (time.time() - start_time) * 1000
    logging.info(f"Inference time: {elapsed_time} milliseconds")

    # After generating the image, upload it to the S3 bucket
    bucket_name = 'ImagineSDBot'
    key = f"{job_input['id']}.jpg"
    upload_to_s3(bucket_name, key, image)
    logging.info(f"Image uploaded to S3 with key {key}")

    # Generate a presigned URL for the uploaded image
    presigned_url = generate_public_url(bucket_name, key)
    logging.info(f"Generated presigned URL: {presigned_url}")

    return {
        "image_url": presigned_url,
        "elapsed_time": elapsed_time
    }


runpod.serverless.start({
    "handler": handler
})

# handler(
#     {
#     "input": {
#         "id": "test99915",
#         "prompt": "masterpiece, best quality, highres, best shadow, best illumination, finely detail, squatting, riding dildo, dildo, contrapposto, exposed breasts, impossible clothes, (pussy:1.2), tight (stockings:1.4) with (skindentation:1.3), (high heels:1.4), (off shoulder dress:1.2), pink hair, long hair, pleasure, orgasm face, pussy juice, (exposed:1.4) breasts, exposed breasts, naked breasts, puffy nipples, sci-fi spaceship, metal, steel, wires, cyberpunk,bound, tied, sex robot, sex machine, by lee jeffries nikon d850 film stock photograph 4 kodak portra 400 camera f1.6 lens rich colors hyper realistic lifelike texture dramatic lighting unreal engine trending on artstation cinestill 800",
#         "scheduler": "DPMSolverMultistepSchedulerKarras",
#         "guidance_scale": 7.5,
#         "seed": 9121234,
#         "upsample": True,
#         "face_restore": True,
#     }
#     }
# )