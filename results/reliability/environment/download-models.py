import fnmatch
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

storage = Path('/home/mlw0719/doit_reliability_storage')
api = HfApi()
records = []
models = [
    ('sd-legacy/stable-diffusion-v1-5', '451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
     ['model_index.json', 'scheduler/*', 'tokenizer/*', 'feature_extractor/*', 'text_encoder/config.json', 'unet/config.json', 'vae/config.json'],
     [('text_encoder/model.safetensors', 'text_encoder/pytorch_model.bin'),
      ('unet/diffusion_pytorch_model.safetensors', 'unet/diffusion_pytorch_model.bin'),
      ('vae/diffusion_pytorch_model.safetensors', 'vae/diffusion_pytorch_model.bin')]),
    ('openai/clip-vit-large-patch14', '32bd64288804d66eefd0ccbe215aa642df71cc41',
     ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.json', 'merges.txt', 'vocab.json'],
     [('model.safetensors', 'pytorch_model.bin')]),
]
for model, revision, patterns, weight_choices in models:
    names = api.list_repo_files(model, revision=revision)
    selected = [name for name in names if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)]
    for choices in weight_choices:
        selected.append(next(name for name in choices if name in names))
    print(json.dumps({'model': model, 'revision': revision, 'files': selected}), flush=True)
    path = snapshot_download(model, revision=revision, cache_dir=str(storage / 'huggingface' / 'hub'), allow_patterns=selected)
    records.append({'model': model, 'revision': revision, 'snapshot': path, 'files': selected})
    (storage / 'model-acquisition.json').write_text(json.dumps(records, indent=2) + '\n')
    print('Downloaded:', path, flush=True)
