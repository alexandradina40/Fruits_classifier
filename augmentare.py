import os
import random
import math
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURARE
# ─────────────────────────────────────────────

INPUT_DIR  = r"D:\Master\ACABI\Clasificare_fructe\output_100x100\Test"
OUTPUT_DIR = r"D:\Master\ACABI\Clasificare_fructe\output_100x100_augmented\Test"

# Numărul țintă de imagini per clasă după augmentare
# Pune None pentru a dubla pur și simplu fiecare clasă
TARGET_PER_CLASS = 80

# Dacă True, copiază și imaginile originale în OUTPUT_DIR
INCLUDE_ORIGINALS = True

# Seed pentru reproducibilitate
RANDOM_SEED = 42

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def get_border_color(img):
    """Culoarea medie a marginilor imaginii – pentru fill natural la rotație/shear."""
    arr = np.array(img)
    border = np.concatenate([
        arr[0, :].reshape(-1, 3),
        arr[-1, :].reshape(-1, 3),
        arr[:, 0].reshape(-1, 3),
        arr[:, -1].reshape(-1, 3),
    ])
    return tuple(np.mean(border, axis=0).astype(int).tolist())

# ─────────────────────────────────────────────
# TRANSFORMĂRI DE AUGMENTARE
# ─────────────────────────────────────────────

def flip_horizontal(img):
    """Oglindire orizontală – utilă și naturală pentru fructe."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)

# flip_vertical eliminat – fructele nu apar cu susul în jos

def rotate(img, angle=None):
    """Rotații mici (±30°) cu fundal bazat pe culoarea marginilor imaginii."""
    if angle is None:
        angle = random.choice([-30, -20, -15, -10, 10, 15, 20, 30])
    fill = get_border_color(img)
    return img.rotate(angle, expand=False, fillcolor=fill)

def adjust_brightness(img, factor=None):
    """Luminozitate moderată: ±30% față de original."""
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Brightness(img).enhance(factor)

def adjust_contrast(img, factor=None):
    """Contrast moderat: ±30%."""
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Contrast(img).enhance(factor)

def adjust_saturation(img, factor=None):
    """Saturație ușor modificată."""
    if factor is None:
        factor = random.uniform(0.6, 1.5)
    return ImageEnhance.Color(img).enhance(factor)

def adjust_sharpness(img, factor=None):
    """Claritate ușor variată."""
    if factor is None:
        factor = random.uniform(0.5, 2.0)
    return ImageEnhance.Sharpness(img).enhance(factor)

def gaussian_blur(img, radius=None):
    """Blur ușor – simulează defocus."""
    if radius is None:
        radius = random.uniform(0.3, 1.2)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def add_noise(img, intensity=None):
    """Zgomot Gaussian subtil."""
    if intensity is None:
        intensity = random.uniform(3, 15)
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def zoom_crop(img, zoom_factor=None):
    """Zoom in ușor + crop centrat – obiectul rămâne întreg."""
    if zoom_factor is None:
        zoom_factor = random.uniform(1.05, 1.20)
    w, h = img.size
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top  = (new_h - h) // 2
    return img_resized.crop((left, top, left + w, top + h))

def shear(img, shear_factor=None):
    """Shear orizontal mic cu fill natural."""
    if shear_factor is None:
        shear_factor = random.uniform(-0.12, 0.12)
    w, h = img.size
    fill = get_border_color(img)
    coeffs = (1, shear_factor, 0, 0, 1, 0)
    return img.transform(
        (w, h), Image.AFFINE, coeffs,
        resample=Image.BICUBIC,
        fillcolor=fill
    )

def random_shift(img, max_shift=None):
    """Translație mică cu fill natural."""
    if max_shift is None:
        max_shift = 0.08
    fill = get_border_color(img)
    w, h = img.size
    dx = int(random.uniform(-max_shift, max_shift) * w)
    dy = int(random.uniform(-max_shift, max_shift) * h)
    arr = np.array(img)
    fill_arr = np.array(fill, dtype=np.uint8)
    result = np.full_like(arr, 0)
    result[:, :] = fill_arr  # fundal natural
    src_x1 = max(0, -dx); src_x2 = min(w, w - dx)
    src_y1 = max(0, -dy); src_y2 = min(h, h - dy)
    dst_x1 = max(0,  dx); dst_x2 = min(w, w + dx)
    dst_y1 = max(0,  dy); dst_y2 = min(h, h + dy)
    result[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]
    return Image.fromarray(result)

# ─────────────────────────────────────────────
# LISTA AUGMENTĂRILOR
# ─────────────────────────────────────────────

ALL_AUGMENTATIONS = [
    ("flip_h",      flip_horizontal),
    ("rotate",      rotate),
    ("brightness",  adjust_brightness),
    ("contrast",    adjust_contrast),
    ("saturation",  adjust_saturation),
    ("sharpness",   adjust_sharpness),
    ("blur",        gaussian_blur),
    ("noise",       add_noise),
    ("zoom",        zoom_crop),
    ("shear",       shear),
    ("shift",       random_shift),
]

def augment_image(img):
    """Aplică 1–2 augmentări aleatoare combinate (mai puțin agresiv)."""
    n = random.randint(1, 2)
    chosen = random.sample(ALL_AUGMENTATIONS, n)
    for name, fn in chosen:
        img = fn(img)
    return img

# ─────────────────────────────────────────────
# PROCESARE PRINCIPALĂ
# ─────────────────────────────────────────────

def get_image_files(folder):
    files = []
    for f in Path(folder).iterdir():
        if f.suffix.lower() in SUPPORTED_EXT:
            files.append(f)
    return files

def process_class(class_name, src_folder, dst_folder, target):
    src_files = get_image_files(src_folder)
    n_existing = len(src_files)

    if n_existing == 0:
        print(f"  [SKIP] {class_name} – nicio imagine găsită.")
        return 0

    dst_folder.mkdir(parents=True, exist_ok=True)
    saved = 0

    # 1. Copiază originalele dacă e setat
    if INCLUDE_ORIGINALS:
        for f in src_files:
            dst_path = dst_folder / f.name
            with Image.open(f) as img:
                img = img.convert("RGB")
                img.save(dst_path)
                saved += 1

    # 2. Calculează câte imagini augmentate mai trebuie generate
    n_to_generate = max(0, target - saved)

    src_cycle = src_files * math.ceil((n_to_generate + 1) / max(n_existing, 1))
    random.shuffle(src_cycle)

    for i in range(n_to_generate):
        src_file = src_cycle[i % len(src_cycle)]
        try:
            with Image.open(src_file) as img:
                img = img.convert("RGB")
                aug_img = augment_image(img)
                stem = src_file.stem
                aug_name = f"aug_{i:05d}_{stem}{src_file.suffix}"
                aug_img.save(dst_folder / aug_name)
                saved += 1
        except Exception as e:
            print(f"    [EROARE] {src_file.name}: {e}")

    return saved


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    if not input_path.exists():
        print(f"[EROARE] Folderul de intrare nu există: {input_path}")
        return

    classes = {}
    for cls_dir in sorted(input_path.iterdir()):
        if cls_dir.is_dir():
            files = get_image_files(cls_dir)
            classes[cls_dir.name] = (cls_dir, len(files))

    if not classes:
        print("[EROARE] Nu s-au găsit subfoldere cu imagini.")
        return

    print("=" * 60)
    print("AUGMENTARE BAZĂ DE DATE – CLASIFICARE FRUCTE")
    print("=" * 60)
    print(f"\nFolder intrare : {input_path}")
    print(f"Folder ieșire  : {output_path}")
    print(f"Target/clasă   : {TARGET_PER_CLASS} imagini")
    print(f"Include orig.  : {'Da' if INCLUDE_ORIGINALS else 'Nu'}")
    print()

    print("Distribuție inițială:")
    for cls_name, (cls_dir, n) in classes.items():
        print(f"  {cls_name:<25} {n:>5} imagini")

    print("\nÎncep augmentarea...\n")

    total_nou = 0
    for cls_name, (cls_dir, n_orig) in classes.items():
        target = TARGET_PER_CLASS if TARGET_PER_CLASS else n_orig * 2
        dst_dir = output_path / cls_name
        print(f"  [{cls_name}]  {n_orig} → {target} imagini")
        saved = process_class(cls_name, cls_dir, dst_dir, target)
        print(f"    ✓ Salvate: {saved} imagini în {dst_dir}")
        total_nou += saved

    print("\n" + "=" * 60)
    print(f"FINALIZAT! Total imagini generate: {total_nou}")
    print(f"Folder output: {output_path}")
    print("=" * 60)

    print("\nDistribuție finală:")
    for cls_name in classes:
        dst_dir = output_path / cls_name
        if dst_dir.exists():
            n_final = len(get_image_files(dst_dir))
            print(f"  {cls_name:<25} {n_final:>5} imagini")


if __name__ == "__main__":
    main()