"""Build the locked exp-046 contact sheets from retrieved local image artifacts."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


root = Path.cwd()
manifest = json.loads(
    (root / "configs/analysis-exp-046-sensitivity-endpoints.json").read_text()
)
methods = list(manifest["groups"])
font = ImageFont.load_default(size=17)
small_font = ImageFont.load_default(size=14)

for prompt in manifest["prompts"]:
    sheet = Image.new("RGB", (320 * len(methods), 370 * len(manifest["seeds"])),
                      (248, 248, 246))
    draw = ImageDraw.Draw(sheet)
    for column, method in enumerate(methods):
        for row, seed in enumerate(manifest["seeds"]):
            run = root / f"artifacts/exp-046-sensitivity-endpoints-{method}-s{seed}"
            record = next(
                item for item in json.loads((run / "images.json").read_text())
                if item["prompt"] == prompt and item["seed"] == seed
            )
            picture = Image.open(run / prompt / str(seed) / "image.png")
            picture = picture.convert("RGB").resize((320, 320))
            x, y = column * 320, row * 370
            draw.text((x + 6, y + 3), f"{method} | seed {seed}",
                      fill="black", font=font)
            caption = (f"R {record['reward']:.3f} | "
                       f"A {record['clip_alignment']:.3f} | "
                       f"{record['seconds']:.0f}s")
            draw.text((x + 6, y + 25), caption, fill="black", font=small_font)
            sheet.paste(picture, (x, y + 48))
    output = root / "artifacts" / f"exp-046-contact-{prompt}.jpg"
    sheet.save(output, quality=92)
    print(output)
