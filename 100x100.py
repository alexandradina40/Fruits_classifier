import os
from pathlib import Path
from PIL import Image

# ─────────────────────────────
# CONFIGURARE
# ─────────────────────────────
INPUT_DIR = r"D:\Master\ACABI\Clasificare_fructe\output_segmented"     # folderul cu subfoldere
OUTPUT_DIR = r"output_100x100"             # unde salvezi imaginile noi

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# ─────────────────────────────
# PROCESARE
# ─────────────────────────────
input_path = Path(INPUT_DIR)
output_path = Path(OUTPUT_DIR)

for root, dirs, files in os.walk(input_path):
    for file in files:
        ext = Path(file).suffix.lower()

        if ext in SUPPORTED_EXT:
            input_file = Path(root) / file

            # creează structura de foldere în output
            relative_path = input_file.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                with Image.open(input_file) as img:
                    img = img.convert("RGB")  # evită probleme cu PNG/alpha
                    img_resized = img.resize((100, 100), Image.LANCZOS)
                    img_resized.save(output_file)

                print(f"[OK] {input_file}")

            except Exception as e:
                print(f"[EROARE] {input_file} -> {e}")