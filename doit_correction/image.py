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
from .execution import categorical_transition, gaussian_displacement


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
        self.dtype = {"float16": torch.float16, "float32": torch.float32}[config.get("dtype", "float16")]
        if self.dtype == torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.steps = int(config.get("steps", 20))
        self.size = int(config.get("size", 512))
        self.chunk = int(config.get("reward_batch", 4))
        self.guidance_scale = float(config.get("guidance_scale", 4.5))
        self.proxy = config.get("proxy", "euler")
        self.score_forwards = 0
        self.reward_queries = 0
        self.reward_seconds = 0.0
        self.vjp_seconds = 0.0
        self.vjp_calls = 0
        self.sensitivity_seconds = 0.0
        model_id = config.get("model", "sd-legacy/stable-diffusion-v1-5")
        model_revision = config.get("model_revision", "451f4fe16113bff5a5d2269ed5ad43b0592e9a14")
        clip_revision = "32bd64288804d66eefd0ccbe215aa642df71cc41"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, revision=model_revision, torch_dtype=self.dtype, local_files_only=True,
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
            "openai/clip-vit-large-patch14", revision=clip_revision, local_files_only=True).to(self.device).eval()
        self.clip.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", revision=clip_revision, local_files_only=True)
        self.head = torch.nn.Linear(768, 1).to(self.device).eval()
        head_path = config.get("aesthetic_head", "/home/mlw0719/doit_correction_storage/models/sa_0_4_vit_l_14_linear.pth")
        self.head.load_state_dict(torch.load(head_path, map_location=self.device, weights_only=True))
        self.head.requires_grad_(False)
        self.provenance = {"model": model_id, "model_revision": model_revision,
                           "clip_model": "openai/clip-vit-large-patch14", "clip_revision": clip_revision,
                           "aesthetic_head": str(head_path), "dtype": str(self.dtype),
                           "scheduler": type(self.pipe.scheduler).__name__, "sigmas": self.sigmas}
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
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        vector = ratio.reshape_as(transition.mean) / transition.noise_std
        result = torch.autograd.grad(transition.mean, transition.x, vector,
                                     retain_graph=retain_graph, create_graph=False)[0]
        torch.cuda.synchronize(self.device)
        self.vjp_seconds += time.perf_counter() - started
        self.vjp_calls += 1
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
            self.reward_queries += len(batch)
        torch.cuda.synchronize(self.device)
        self.reward_seconds += time.perf_counter() - started
        return torch.cat(results)

    @torch.no_grad()
    def future_reward(self, transition, eps, seed, paired_future=False):
        """One full future rollout for each first-transition candidate.

        The first transition uses supplied eps. Optionally share all later
        noises within each antithetic pair; each marginal is still the base
        conditional rollout. No guidance is applied to these futures.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        rewards = []
        mean = transition.mean.detach()
        shape = mean.shape[1:]
        chunk = int(self.config.get("future_batch", 2))
        if paired_future:
            if len(eps) % 2:
                raise ValueError("Paired futures require an even antithetic batch")
            half = len(eps) // 2
            eps = torch.stack([eps[:half], eps[half:]], dim=1).flatten(0, 1)
            chunk = 2 * max(1, chunk // 2)
        for batch in eps.split(chunk):
            x = mean + transition.noise_std * batch.reshape(-1, *shape)
            for index in range(transition.index + 1, self.steps):
                prediction = self.predict_epsilon(x, index)
                up, _, dt = euler_coefficients(*self.sigmas[index:index + 2])
                noise_shape = (len(x) // 2, *shape) if paired_future else x.shape
                noise = torch.randn(noise_shape, device=self.device, generator=generator)
                if paired_future:
                    noise = noise.repeat_interleave(2, dim=0)
                x = x + dt * prediction + up * noise
            rewards.append(self.reward(x.flatten(1)))
        rewards = torch.cat(rewards)
        return rewards.reshape(-1, 2).T.flatten() if paired_future else rewards

    def generate(self, prompt, seed, estimator_config, output, capture_callback=None):
        self.use_prompt(prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        selection_generator = torch.Generator(device=self.device).manual_seed(seed + 2000003)
        estimator = MomentEstimator(estimator_config, seed + 1000003, self.device)
        latent = torch.randn((1, 4, self.size // 8, self.size // 8), device=self.device,
                             generator=generator) * self.sigmas[0]
        gamma = float(estimator_config.get("gamma", 4.0))
        execution = estimator_config.get("execution", "gradient")
        if execution not in {"gradient", "resample"}:
            raise ValueError(execution)
        if execution == "resample" and (estimator_config.get("oracle") != "future" or estimator.kind != "mc"):
            raise ValueError("Categorical execution requires fresh MC full-future candidates")
        start_index = self.steps - int(self.config.get("guided_steps", 10))
        diagnostics = []
        previous_guidance = None
        counts_before = self.reward_queries, self.score_forwards
        times_before = self.reward_seconds, self.vjp_seconds
        sensitivity_before = self.sensitivity_seconds
        vjps_before = self.vjp_calls
        started = time.perf_counter()
        for index in range(self.steps):
            guided = index >= start_index and index < self.steps - 1 and (gamma != 0 or execution == "resample")
            transition = self.transition(latent, index,
                                         with_grad=(guided and execution == "gradient") or capture_callback is not None)
            if capture_callback is not None and index >= start_index and index < self.steps - 1:
                capture_callback(transition, prompt, seed)
            if index in self.config.get("trace_indices", []):
                trace_dir = Path(output) / "states"
                trace_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"latent": transition.x.detach().cpu(), "index": index,
                            "prompt": prompt, "seed": seed, "image_config": self.config,
                            "estimator_config": estimator_config}, trace_dir / f"step{index:02d}.pt")
            if guided:
                oracle = estimator_config.get("oracle", "proxy")
                if oracle == "proxy":
                    estimate = estimator.estimate(transition.endpoint_mean, transition.endpoint_std, self.reward)
                elif oracle in {"future", "multifidelity"}:
                    from .fidelity import estimate_fidelity
                    future_seed = seed * 100003 + index * 997 + 15000001
                    future_reward = lambda eps: self.future_reward(
                        transition, eps, future_seed, paired_future=estimator_config.get("paired_future", True))
                    estimate = estimate_fidelity(estimator, transition.endpoint_mean, transition.endpoint_std,
                                                 self.reward, future_reward)
                else:
                    raise ValueError(oracle)
                if execution == "resample":
                    selected_latent, execution_diagnostics = categorical_transition(
                        transition.mean.detach(), transition.noise_std,
                        estimate.eps, estimate.log_weights, selection_generator)
                else:
                    if "sensitivity" in estimator_config:
                        from .reliability import sensitivity_correction
                        correction = sensitivity_correction(
                            self, transition, estimate.ratio, estimator_config["sensitivity"],
                            seed * 100003 + index * 997 + 43000003)
                    else:
                        correction = self.correction(transition, estimate.ratio)
                    raw_correction_norm = float(correction.norm())
                    if estimator.kind == "guidance_ema" and previous_guidance is not None:
                        beta = float(estimator_config.get("ema_beta", 0.5))
                        correction = beta * correction + (1 - beta) * previous_guidance
                    previous_guidance = correction.detach().clone()
                    if not torch.isfinite(correction).all():
                        raise FloatingPointError(
                            f"Nonfinite VJP guidance at index {index}: {estimate.diagnostics}")
                    displacement, execution_diagnostics = gaussian_displacement(
                        correction, transition.correction_gain, gamma, transition.noise_std,
                        estimator_config.get("execution_kl_limit"))
                    execution_diagnostics.update(correction_norm=float(correction.norm()),
                                                 raw_correction_norm=raw_correction_norm)
                    mean = transition.mean.detach() + displacement
                diagnostics.append(dict(index=index, execution=execution,
                                        sigma=self.sigmas[index], sigma_next=self.sigmas[index + 1],
                                        noise_std=transition.noise_std,
                                        correction_gain=transition.correction_gain,
                                        **execution_diagnostics, **estimate.diagnostics))
            else:
                mean = transition.mean.detach()
            noise = torch.randn(latent.shape, device=self.device, generator=generator)
            # Advance the base RNG once per step for aligned streams. A selected
            # candidate already contains its transition noise and gets no extra noise.
            latent = selected_latent if guided and execution == "resample" else mean + transition.noise_std * noise
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
            "reward_seconds": self.reward_seconds - times_before[0],
            "vjp_seconds": self.vjp_seconds - times_before[1],
            "vjp_calls": self.vjp_calls - vjps_before,
            "sensitivity_seconds": self.sensitivity_seconds - sensitivity_before,
            "oracle_seconds": sum(d.get("oracle_seconds", 0.0) for d in diagnostics),
            "estimator_seconds": sum(d.get("estimator_seconds", 0.0) for d in diagnostics),
            "update_seconds": sum(d.get("update_seconds", 0.0) for d in diagnostics),
            "denominator_events": sum(d["denominator_floored"] for d in diagnostics),
            "steps": diagnostics,
        }
