from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-1-base"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                               scheduler=scheduler,
                                               torch_dtype=torch.bfloat16,
                                               safety_checker=None,
                                               requires_safety_checker=False,
                                               feature_extractor=None
                                               )

# Pipeline (including the final vae decoding and image saving process) should fit in 4gb vram with this setting enabled
pipe.enable_model_cpu_offload()

def callback(iter, t, latents):
  # convert latents to image
  with torch.no_grad():
      latents = 1 / 0.18215 * latents
      
      image = pipe.vae.decode(latents).sample

      image = (image / 2 + 0.5).clamp(0, 1)

      # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
      image = image.cpu().permute(0, 2, 3, 1).float().numpy()

      # convert to PIL Images
      image = pipe.numpy_to_pil(image)

      # do something with the Images
      for i, img in enumerate(image):
          img.save(f"iter_{iter}_img{i}.png")

prompt = "primary color wheel, alphonse mucha style"
neg_prompt = 'ugly, watermark, signature, people, humans, woman, man'
image = pipe(prompt, negative_prompt=neg_prompt, num_inference_steps=30, callback=callback, callback_steps=5).images[0]

image.save("final-img.png")
