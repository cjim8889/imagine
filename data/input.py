from typing import Optional
from pydantic import create_model

## load your model(s) into vram here
default_negative_prompt = '''
    canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render
'''

# Define a list of acceptable parameters
txt2img_job_params = [
    ('job_id', str, ...),
    ('prompt', str, ...),
    ('model_name', str, "cjim8889/AllysMix3"),
    ('negative_prompt', Optional[str], default_negative_prompt),
    ('scheduler', Optional[str], "DPMSolverMultistepScheduler"),
    ('num_inference_steps', Optional[int], 30),
    ('guidance_scale', Optional[float], 8.0),
    ('codeformer_fidelity', Optional[float], 0.6),
    ('seed', Optional[int], None),
    ('width', Optional[int], 512),
    ('height', Optional[int], 512),
    ('upsample', Optional[bool], False),
    ('face_restore', Optional[bool], False),
]

TextToImageParams = create_model(
    "TextToImageParams",
    **{name: (type_, default) for name, type_, default in txt2img_job_params},
)
