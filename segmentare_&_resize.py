"""
Pipeline complet: Segmentare + Redimensionare 100x100
------------------------------------------------------
Pasul 1 – Segmentare: elimină fundalul, curăță bățul, pune fructul pe fundal alb.
Pasul 2 – Resize: redimensionează imaginea segmentată la 100x100 px.

Rezultatul final se salvează direct în OUTPUT_FOLDER (nu mai e nevoie de un script separat).

Instalare dependențe:
    pip install rembg pillow onnxruntime opencv-python numpy
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from rembg import remove

# ──────────────────────────────────────────────
# CONFIGURARE
# ──────────────────────────────────────────────

INPUT_FOLDER  = r"D:\Master\ACABI\Clasificare_fructe\Imagini_fructe"
OUTPUT_FOLDER = r"D:\Master\ACABI\Clasificare_fructe\output_segmented_100x100"

EXTENSION     = {".JPG", ".jpg", ".jpeg", ".png", ".bmp"}
OUTPUT_SIZE   = (100, 100)   # dimensiunea finală dorită

# ──────────────────────────────────────────────
# FUNCȚII SEGMENTARE
# ──────────────────────────────────────────────

def composite_on_white(rgba_img: Image.Image) -> Image.Image:
    """Lipește fructul decupat pe un fundal complet alb."""
    background = Image.new("RGB", rgba_img.size, (255, 255, 255))
    background.paste(rgba_img, mask=rgba_img.split()[3])
    return background


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """Păstrează doar cea mai mare insulă de pixeli (fructul), eliminând zgomotul."""
    binary = (binary_mask > 128).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return binary

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(binary)
    out[labels == largest_label] = 255
    return out


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """Umple găurile din interiorul fructului (ex: reflexii)."""
    binary = (binary_mask > 128).astype(np.uint8) * 255
    h, w = binary.shape
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return binary | flood_inv


def remove_stick_universal(binary_mask: np.ndarray) -> np.ndarray:
    """Elimină bățul de susținere, indiferent de unghi."""
    _, img_w = binary_mask.shape

    kernel_size = int(img_w * 0.055)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    core_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    removed_parts = cv2.subtract(binary_mask, core_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(removed_parts, connectivity=8)
    final_mask = core_mask.copy()

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 100:
            final_mask[labels == label] = 255
            continue

        comp_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]

        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio < 2.2:
            final_mask[labels == label] = 255

    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_smooth)
    return final_mask


def apply_segmentation_pipeline(img: Image.Image) -> Image.Image:
    """Pipeline complet de segmentare: rembg → curățare băț → umplere găuri."""
    # Pas 1: Eliminare fundal cu rembg
    rgba_img = remove(img)
    alpha = np.array(rgba_img.split()[3])
    binary_mask = (alpha > 128).astype(np.uint8) * 255

    # Pas 2: Eliminare băț
    clean_mask = remove_stick_universal(binary_mask)

    # Pas 3: Curățare finală
    clean_mask = keep_largest_component(clean_mask)
    clean_mask = fill_holes(clean_mask)

    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=2)
    clean_mask = keep_largest_component(clean_mask)

    # Aplicare mască pe imaginea originală
    r, g, b = img.convert("RGB").split()
    final_rgba = Image.merge("RGBA", (r, g, b, Image.fromarray(clean_mask, mode="L")))

    return composite_on_white(final_rgba)


# ──────────────────────────────────────────────
# FUNCȚIE RESIZE
# ──────────────────────────────────────────────

def resize_to_target(img: Image.Image, size: tuple) -> Image.Image:
    """Redimensionează imaginea la dimensiunea dorită cu interpolarea LANCZOS."""
    return img.resize(size, Image.LANCZOS)


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def process_image(input_path: Path, output_folder: Path) -> None:
    print(f"  {input_path.name} ...", end=" ", flush=True)

    # Pas 1: Deschide imaginea
    pil_img = Image.open(input_path).convert("RGB")

    # Pas 2: Segmentare
    segmented = apply_segmentation_pipeline(pil_img)

    # Pas 3: Resize la 100x100
    final = resize_to_target(segmented, OUTPUT_SIZE)

    # Salvare
    out_name = f"{input_path.stem}_segmented.jpg"
    out_path = output_folder / out_name
    final.save(out_path, "JPEG", quality=95)

    print(f"✓  → {out_name}")


def main():
    input_folder  = Path(INPUT_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    images = [p for p in input_folder.iterdir()
              if p.suffix in EXTENSION or p.suffix.upper() in EXTENSION]

    if not images:
        print(f"[EROARE] Nicio imagine găsită în '{input_folder}'.")
        sys.exit(1)

    print(f"{'='*55}")
    print(f"  SEGMENTARE + RESIZE {OUTPUT_SIZE[0]}×{OUTPUT_SIZE[1]}px")
    print(f"{'='*55}")
    print(f"  Input : {input_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Imagini găsite: {len(images)}\n")

    ok, err = 0, 0
    for img_path in sorted(images):
        try:
            process_image(img_path, output_folder)
            ok += 1
        except Exception as e:
            print(f"  [EROARE] {img_path.name}: {e}")
            err += 1

    print(f"\n{'='*55}")
    print(f"  FINALIZAT: {ok} procesate, {err} erori")
    print(f"  Folder output: {output_folder}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
