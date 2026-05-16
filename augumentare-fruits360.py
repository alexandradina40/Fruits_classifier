"""
Augmentare Training + Test cu ținte fixe per clasă
---------------------------------------------------
Training → 520 imagini per clasă
Test     → 174 imagini per clasă

Structura așteptată în INPUT_DIR:
    INPUT_DIR/
        Training/
            Apple Granny Smith/
            Apple Red/
            ...
        Test/
            Apple Granny Smith/
            Apple Red/
            ...
"""

import os
import random
import math
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURARE
# ─────────────────────────────────────────────

INPUT_DIR  = r"D:\Master\ACABI\Clasificare_fructe\fruits-360_6_clase"
OUTPUT_DIR = r"D:\Master\ACABI\Clasificare_fructe\fruits-360_6_clase-augmented"

TARGET_TRAIN = 520   # imagini per clasă în Training
TARGET_TEST  = 174   # imagini per clasă în Test

INCLUDE_ORIGINALS = True   # copiază și originalele în output
RANDOM_SEED       = 42
SUPPORTED_EXT     = {".jpg", ".jpeg", ".png", ".bmp",
                     ".JPG", ".JPEG", ".PNG", ".BMP"}

# ─────────────────────────────────────────────
# AUGMENTĂRI (aceleași ca înainte)
# ─────────────────────────────────────────────

def get_border_color(img):
    arr = np.array(img)
    border = np.concatenate([
        arr[0, :].reshape(-1, 3),
        arr[-1, :].reshape(-1, 3),
        arr[:, 0].reshape(-1, 3),
        arr[:, -1].reshape(-1, 3),
    ])
    return tuple(np.mean(border, axis=0).astype(int).tolist())

def flip_horizontal(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def rotate(img, angle=None):
    if angle is None:
        angle = random.choice([-30, -20, -15, -10, 10, 15, 20, 30])
    fill = get_border_color(img)
    return img.rotate(angle, expand=False, fillcolor=fill)

def adjust_brightness(img, factor=None):
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Brightness(img).enhance(factor)

def adjust_contrast(img, factor=None):
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Contrast(img).enhance(factor)

def adjust_saturation(img, factor=None):
    if factor is None:
        factor = random.uniform(0.6, 1.5)
    return ImageEnhance.Color(img).enhance(factor)

def adjust_sharpness(img, factor=None):
    if factor is None:
        factor = random.uniform(0.5, 2.0)
    return ImageEnhance.Sharpness(img).enhance(factor)

def gaussian_blur(img, radius=None):
    if radius is None:
        radius = random.uniform(0.3, 1.2)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def add_noise(img, intensity=None):
    if intensity is None:
        intensity = random.uniform(3, 15)
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def zoom_crop(img, zoom_factor=None):
    if zoom_factor is None:
        zoom_factor = random.uniform(1.05, 1.20)
    w, h = img.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top  = (new_h - h) // 2
    return img_resized.crop((left, top, left + w, top + h))

def shear(img, shear_factor=None):
    if shear_factor is None:
        shear_factor = random.uniform(-0.12, 0.12)
    w, h = img.size
    fill = get_border_color(img)
    coeffs = (1, shear_factor, 0, 0, 1, 0)
    return img.transform(
        (w, h), Image.AFFINE, coeffs,
        resample=Image.BICUBIC, fillcolor=fill
    )

def random_shift(img, max_shift=None):
    if max_shift is None:
        max_shift = 0.08
    fill = get_border_color(img)
    w, h = img.size
    dx = int(random.uniform(-max_shift, max_shift) * w)
    dy = int(random.uniform(-max_shift, max_shift) * h)
    arr = np.array(img)
    result = np.full_like(arr, fill)
    src_x1 = max(0, -dx); src_x2 = min(w, w - dx)
    src_y1 = max(0, -dy); src_y2 = min(h, h - dy)
    dst_x1 = max(0,  dx); dst_x2 = min(w, w + dx)
    dst_y1 = max(0,  dy); dst_y2 = min(h, h + dy)
    result[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]
    return Image.fromarray(result)

ALL_AUGMENTATIONS = [
    ("flip_h",     flip_horizontal),
    ("rotate",     rotate),
    ("brightness", adjust_brightness),
    ("contrast",   adjust_contrast),
    ("saturation", adjust_saturation),
    ("sharpness",  adjust_sharpness),
    ("blur",       gaussian_blur),
    ("noise",      add_noise),
    ("zoom",       zoom_crop),
    ("shear",      shear),
    ("shift",      random_shift),
]

def augment_image(img):
    """Aplică 1–2 augmentări aleatoare combinate."""
    n = random.randint(1, 2)
    chosen = random.sample(ALL_AUGMENTATIONS, n)
    for _, fn in chosen:
        img = fn(img)
    return img

# ─────────────────────────────────────────────
# PROCESARE PER CLASĂ
# ─────────────────────────────────────────────

def get_image_files(folder: Path):
    return [f for f in folder.iterdir()
            if f.is_file() and f.suffix in SUPPORTED_EXT]

def process_class(class_name, src_folder, dst_folder, target):
    src_files = get_image_files(src_folder)
    n_existing = len(src_files)

    if n_existing == 0:
        print(f"    [SKIP] {class_name} – nicio imagine găsită.")
        return 0

    dst_folder.mkdir(parents=True, exist_ok=True)
    saved = 0

    # 1. Copiază originalele
    if INCLUDE_ORIGINALS:
        for f in src_files:
            with Image.open(f) as img:
                img.convert("RGB").save(dst_folder / f.name)
                saved += 1

    # 2. Generează augmentări până la target
    n_to_generate = max(0, target - saved)
    if n_to_generate == 0:
        print(f"    {class_name:<25} deja {saved} imagini (>= {target}), skip augmentare.")
        return saved

    # Repetă lista sursă de câte ori e nevoie
    src_cycle = src_files * math.ceil((n_to_generate + 1) / max(n_existing, 1))
    random.shuffle(src_cycle)

    for i in range(n_to_generate):
        src_file = src_cycle[i % len(src_cycle)]
        try:
            with Image.open(src_file) as img:
                aug_img = augment_image(img.convert("RGB"))
                aug_name = f"aug_{i:05d}_{src_file.stem}{src_file.suffix}"
                aug_img.save(dst_folder / aug_name)
                saved += 1
        except Exception as e:
            print(f"      [EROARE] {src_file.name}: {e}")

    return saved

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def process_split(split_name, target_per_class):
    """Procesează un split (Training sau Test)."""
    src_split = Path(INPUT_DIR)  / split_name
    dst_split = Path(OUTPUT_DIR) / split_name

    if not src_split.exists():
        print(f"  [SKIP] {split_name} nu există în {INPUT_DIR}")
        return

    class_dirs = sorted([d for d in src_split.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"  [EROARE] Nicio clasă găsită în {src_split}")
        return

    print(f"\n{'─'*55}")
    print(f"  {split_name.upper()}  →  țintă {target_per_class} imagini/clasă")
    print(f"{'─'*55}")

    total = 0
    for cls_dir in class_dirs:
        dst_dir = dst_split / cls_dir.name
        n_orig  = len(get_image_files(cls_dir))
        saved   = process_class(cls_dir.name, cls_dir, dst_dir, target_per_class)
        print(f"    {cls_dir.name:<28}  {n_orig:>4} orig → {saved:>4} final")
        total += saved

    print(f"\n  Total {split_name}: {total} imagini salvate în {dst_split}")

    # Statistici finale
    print(f"\n  Distribuție finală {split_name}:")
    for cls_dir in sorted(dst_split.iterdir()):
        if cls_dir.is_dir():
            n = len(get_image_files(cls_dir))
            bar = "█" * (n // 10)
            print(f"    {cls_dir.name:<28}  {n:>4}  {bar}")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("  AUGMENTARE TRAINING + TEST")
    print("=" * 55)
    print(f"  Input  : {INPUT_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Țintă Training : {TARGET_TRAIN} / clasă")
    print(f"  Țintă Test     : {TARGET_TEST} / clasă")

    process_split("Training", TARGET_TRAIN)
    process_split("Test",     TARGET_TEST)

    print("\n" + "=" * 55)
    print("  FINALIZAT!")
    print("=" * 55)


if __name__ == "__main__":
    main()