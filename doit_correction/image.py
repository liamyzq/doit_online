"""Frozen SD v1.5 image experiments with explicit Euler Gaussian kernels."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionPipeline
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .estimators import MomentEstimator


PROMPTS = {
    "bernese": "portrait of a bernese mountain dog in golden-hour light, warm glow, gentle bokeh, masterpiece, high quality",
    "formula1": "a formula one race car on a racetrack, dynamic angle, masterpiece, professional photography, high quality",
    "airplane": "a commercial airplane flying above the clouds, vibrant colors, dramatic lighting, masterpiece, high detail, professional photography, high quality",
    "labrador": "a labrador retriever playing in a sunny park, joyful expression, vibrant colors, masterpiece, high quality, professional photography",
    "lighthouse": "a white lighthouse on a rocky coast at sunset, ocean waves, detailed professional landscape photography",
    "cafe": "a cozy street cafe in Paris on a rainy evening, warm window light, reflections, professional photography",
    "orchid": "a purple orchid with dew drops, macro photography, soft natural light, detailed petals",
    "mountain": "a mountain lake reflecting snow covered peaks at sunrise, clear water, professional landscape photography",
}
NEGATIVE_PROMPT = (
    "blurry, low resolution, bad lighting, unclear background, grainy texture, "
    "messy background, slightly out-of-focus background, simple framing, soft "
    "background blur, unnatural colors, overexposed, underexposed, harsh shadows, "
    "excessive noise, distorted features, unrealistic proportions, cartoonish "
    "style, poor composition, awkward angles, cluttered background, distracting "
    "elements, lack of detail, flat lighting, unflattering perspective"
)


@dataclass
class Transition:
    x: torch.Tensor
    mean: torch.Tensor
    endpoint_mean: torch.Tensor
    endpoint_std: float
    noise_std: float
    correction_gain: float
    index: int


def euler_coefficients(sigma, sigma_next):
    sigma, sigma_next = float(sigma), float(sigma_next)
    if sigma_next == 0:
        return 0.0, 0.0, -sigma
    up = math.sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
    down = math.sqrt(max(0, sigma_next ** 2 - up ** 2))
    return up, down, down - sigma


class ImageEngine:
    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.steps = int(config.get("steps", 20))
        self.size = int(config.get("size", 512))
        self.chunk = int(config.get("reward_batch", 4))
        self.guidance_scale = float(config.get("guidance_scale", 4.5))
        self.proxy = config.get("proxy", "euler")
        self.score_forwards = 0
        self.reward_queries = 0
        self.reward_seconds = 0.0
        self.vjp_seconds = 0.0
        model_id = config.get("model", "sd-legacy/stable-diffusion-v1-5")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype, local_files_only=True,
            safety_checker=None, requires_safety_checker=False)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        if self.pipe.scheduler.config.prediction_type != "epsilon":
            raise ValueError("This study specifies SD v1.5 epsilon prediction")
        self.pipe = self.pipe.to(self.device)
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.scheduler.set_timesteps(self.steps, device=self.device)
        self.sigmas = self.pipe.scheduler.sigmas.float().cpu().tolist()
        self.timesteps = self.pipe.scheduler.timesteps
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", local_files_only=True).to(self.device).eval()
        self.clip.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
        self.head = torch.nn.Linear(768, 1).to(self.device).eval()
        head_path = config.get("aesthetic_head", "/home/mlw0719/doit_correction_storage/models/sa_0_4_vit_l_14_linear.pth")
        self.head.load_state_dict(torch.load(head_path, map_location=self.device, weights_only=True))
        self.head.requires_grad_(False)
        self.prompt_cache = {}
        self.current_prompt = None
        self.embeddings = None

    @property
    def dimension(self):
        return 4 * (self.size // 8) ** 2

    def use_prompt(self, name):
        text = PROMPTS.get(name, name)
        if text not in self.prompt_cache:
            with torch.no_grad():
                positive, negative = self.pipe.encode_prompt(
                    text, self.device, 1, True, negative_prompt=NEGATIVE_PROMPT)
            self.prompt_cache[text] = (positive, negative)
        self.current_prompt = text
        self.embeddings = self.prompt_cache[text]

    def predict_epsilon(self, x, index):
        positive, negative = self.embeddings
        b = len(x)
        inputs = torch.cat([x, x]) / math.sqrt(self.sigmas[index] ** 2 + 1)
        embeds = torch.cat([negative.expand(b, -1, -1), positive.expand(b, -1, -1)])
        pred = self.pipe.unet(inputs.to(self.dtype), self.timesteps[index],
                              encoder_hidden_states=embeds).sample.float()
        uncond, cond = pred.chunk(2)
        self.score_forwards += b
        return uncond + self.guidance_scale * (cond - uncond)

    def transition(self, latent, index, with_grad=True):
        x = latent.detach().float().requires_grad_(with_grad)
        with torch.set_grad_enabled(with_grad):
            prediction = self.predict_epsilon(x, index)
            sigma, sigma_next = self.sigmas[index:index + 2]
            up, _, dt = euler_coefficients(sigma, sigma_next)
            mean = x + dt * prediction
        if self.proxy == "euler":
            endpoint_mean = mean.detach() - sigma_next * prediction.detach()
            endpoint_std = up
            gain = -dt * sigma
        elif self.proxy == "source_vp":
            alpha = self.pipe.scheduler.alphas_cumprod
            current_t = int(self.timesteps[index])
            next_t = int(self.timesteps[min(index + 1, self.steps - 1)])
            a, ap = float(alpha[current_t]), float(alpha[next_t])
            endpoint_mean = (mean.detach() - math.sqrt(1 - ap) * prediction.detach()) / math.sqrt(ap)
            endpoint_std = up / math.sqrt(ap)
            gain = -dt * math.sqrt(1 - a)
        else:
            raise ValueError(self.proxy)
        return Transition(x, mean, endpoint_mean.flatten(), endpoint_std, up, gain, index)

    def correction(self, transition, ratio, retain_graph=False):
        started = time.perf_counter()
        vector = ratio.reshape_as(transition.mean) / transition.noise_std
        result = torch.autograd.grad(transition.mean, transition.x, vector,
                                     retain_graph=retain_graph, create_graph=False)[0]
        torch.cuda.synchronize(self.device)
        self.vjp_seconds += time.perf_counter() - started
        return result.detach()

    @torch.no_grad()
    def decode(self, flat_latents):
        latents = flat_latents.reshape(-1, 4, self.size // 8, self.size // 8)
        values = self.pipe.vae.decode((latents / self.pipe.vae.config.scaling_factor).to(self.dtype)).sample
        values = (values.float() / 2 + 0.5).clamp(0, 1)
        arrays = (values.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        return [Image.fromarray(array) for array in arrays]

    @torch.no_grad()
    def score_images(self, images, return_embeddings=False):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        embeds = F.normalize(self.clip.get_image_features(**inputs), dim=-1)
        scores = self.head(embeds).flatten()
        if return_embeddings:
            text = self.processor(text=[self.current_prompt], return_tensors="pt", padding=True).to(self.device)
            text_emb = F.normalize(self.clip.get_text_features(**text), dim=-1)
            return scores, embeds, (embeds @ text_emb.T).flatten()
        return scores

    @torch.no_grad()
    def reward(self, flat_latents):
        started = time.perf_counter()
        results = []
        for batch in flat_latents.split(self.chunk):
            results.append(self.score_images(self.decode(batch)))
        self.reward_queries += len(flat_latents)
        torch.cuda.synchronize(self.device)
        self.reward_seconds += time.perf_counter() - started
        return torch.cat(results)

    @torch.no_grad()
    def future_reward(self, transition, eps, seed):
        """One full future rollout for each first-transition candidate.

        The first transition uses supplied eps. Future noises are independent;
        no guidance is applied while estimating the base conditional reward.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        rewards = []
        mean = transition.mean.detach()
        shape = mean.shape[1:]
        for batch in eps.split(int(self.config.get("future_batch", 2))):
            x = mean + transition.noise_std * batch.reshape(-1, *shape)
            for index in range(transition.index + 1, self.steps):
                prediction = self.predict_epsilon(x, index)
                up, _, dt = euler_coefficients(*self.sigmas[index:index + 2])
                noise = torch.randn(x.shape, device=self.device, generator=generator)
                x = x + dt * prediction + up * noise
            rewards.append(self.reward(x.flatten(1)))
        return torch.cat(rewards)

    def generate(self, prompt, seed, estimator_config, output, capture_callback=None):
        self.use_prompt(prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        estimator = MomentEstimator(estimator_config, seed + 1000003, self.device)
        latent = torch.randn((1, 4, self.size // 8, self.size // 8), device=self.device,
                             generator=generator) * self.sigmas[0]
        gamma = float(estimator_config.get("gamma", 4.0))
        start_index = self.steps - int(self.config.get("guided_steps", 10))
        diagnostics = []
        previous_guidance = None
        counts_before = self.reward_queries, self.score_forwards
        started = time.perf_counter()
        for index in range(self.steps):
            guided = index >= start_index and index < self.steps - 1 and gamma != 0
            transition = self.transition(latent, index, with_grad=guided or capture_callback is not None)
            if capture_callback is not None and index >= start_index and index < self.steps - 1:
                capture_callback(transition, prompt, seed)
            if guided:
                estimate = estimator.estimate(transition.endpoint_mean, transition.endpoint_std, self.reward)
                correction = self.correction(transition, estimate.ratio)
                if estimator.kind == "guidance_ema" and previous_guidance is not None:
                    beta = float(estimator_config.get("ema_beta", 0.5))
                    correction = beta * correction + (1 - beta) * previous_guidance
                previous_guidance = correction.detach().clone()
                if not torch.isfinite(correction).all():
                    raise FloatingPointError("Nonfinite VJP guidance")
                diagnostics.append(dict(index=index, correction_norm=float(correction.norm()), **estimate.diagnostics))
                mean = transition.mean.detach() + gamma * transition.correction_gain * correction
            else:
                mean = transition.mean.detach()
            noise = torch.randn(latent.shape, device=self.device, generator=generator)
            latent = mean + transition.noise_std * noise
            del transition
        images = self.decode(latent.flatten(1))
        scores, embeddings, alignment = self.score_images(images, return_embeddings=True)
        self.reward_queries += 1
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        images[0].save(output / "image.png")
        np.save(output / "embedding.npy", embeddings[0].cpu().numpy())
        return {
            "prompt": prompt, "seed": seed, "reward": float(scores[0]),
            "clip_alignment": float(alignment[0]), "seconds": elapsed,
            "guidance_queries": estimator.queries, "evaluation_queries": 1,
            "all_queries": self.reward_queries - counts_before[0],
            "score_forwards": self.score_forwards - counts_before[1],
            "denominator_events": sum(d["denominator_floored"] for d in diagnostics),
            "steps": diagnostics,
        }
