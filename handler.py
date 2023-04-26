import runpod
from diffusers import DiffusionPipeline
import torch
import os, time
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

## load your model(s) into vram here
default_negative_prompt = '''
    canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render
'''
model = DiffusionPipeline.from_pretrained(
    "/model",
    custom_pipeline="lpw_stable_diffusion",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    requires_safety_checker=False, 
    safety_checker=lambda images, **kwargs: [images, [False] * len(images)],
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
        guidance_scale=8
        ) -> PIL.Image.Image:
    image = model(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
    ).images[0]

    return image

def switch_scheduler(scheduler_name):
    global current_scheduler
    if scheduler_name == current_scheduler:
        return
    
    if scheduler_name in available_schedulers:
        model.scheduler = available_schedulers[scheduler_name].from_config(model.scheduler.config)
        current_scheduler = scheduler_name
        logging.info(f"Switched to {scheduler_name}")
        return
    else:
        logging.info(f"Error: {scheduler_name} is not a valid scheduler")
        return

def handler(event):
    job_input = event["input"]
    if "id" not in job_input:
        return { "error": "no id provided"}
    
    if "prompt" not in job_input:
        return { "error": "no prompt provided"}
    
    if "scheduler" in job_input:
        switch_scheduler(job_input["scheduler"])
    
    start_time = time.time()

    logging.info(f"Starting inference for job {job_input['id']}")
    image = inference(
        prompt=job_input["prompt"],
        negative_prompt=job_input.get("negative_prompt", default_negative_prompt),
        num_inference_steps=job_input.get("num_inference_steps", 3),
        width=job_input.get("width", 512),
        height=job_input.get("height", 512),
        guidance_scale=job_input.get("guidance_scale", 8)
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
        "status": "completed",
        "url": presigned_url,
        "elapsed_time": elapsed_time
    }


runpod.serverless.start({
    "handler": handler
})

# handler(
#     {
#     "input": {
#         "id": "test1",
#         "prompt": "a cat",
#         "scheduler": "DPMSolverMultistepScheduler",
#     }
#     }
# )