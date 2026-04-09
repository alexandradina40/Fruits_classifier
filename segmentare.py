import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from rembg import remove
#   pip install rembg pillow onnxruntime opencv-python numpy

# ──────────────────────────────────────────────
# CONFIGURARE
# ──────────────────────────────────────────────

INPUT_FOLDER = r"D:\Master\ACABI\Clasificare_fructe\Imagini_fructe"
OUTPUT_FOLDER = "output_segmented"
EXTENSION = {".JPG"}
# ──────────────────────────────────────────────

def composite_on_white(rgba_img) -> Image.Image:
    """Lipește fructul decupat pe un fundal complet alb."""
    background = Image.new("RGB", rgba_img.size, (255, 255, 255))
    background.paste(rgba_img, mask=rgba_img.split()[3])
    return background

def keep_largest_component(binary_mask) -> np.ndarray:
    """Păstrează doar cea mai mare insulă de pixeli (fructul), eliminând zgomotul."""
    binary = (binary_mask > 128).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return binary

    # stats[1:] ignoră fundalul (label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(binary)
    out[labels == largest_label] = 255
    return out

def fill_holes(binary_mask) -> np.ndarray:
    """Umple eventualele găuri apărute în interiorul fructului (ex: reflexii)."""
    binary = (binary_mask > 128).astype(np.uint8) * 255
    h, w = binary.shape
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return binary | flood_inv

# ══════════════════════════════════════════════════════════════════
# ELIMINARE BĂȚ - METODA UNIVERSALĂ (INVARIANTĂ LA ROTAȚIE)
# ══════════════════════════════════════════════════════════════════

def remove_stick_universal(binary_mask) -> np.ndarray:

    _, img_w = binary_mask.shape

    # 1. Setăm "radiera" (kernel-ul) să fie un pic mai groasă decât bățul.
    # Estimăm grosimea bățului la aproximativ 5% din lățimea imaginii.
    kernel_size = int(img_w * 0.055)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Dimensiunea trebuie să fie număr impar (ex: 45x45, 55x55)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 2. Tăierea brutală (MORPH_OPEN)
    # Erodează masca suficient cât să dispară bețele complet, apoi o dilată la loc.
    # core_mask va conține doar bucata "grasă" a imaginii (fructul).
    core_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # 3. Ce anume a fost tăiat?
    # Scădem din masca originală ce a rămas, pentru a izola bățul
    # și eventualele frunze/colțuri subțiri tăiate accidental.
    removed_parts = cv2.subtract(binary_mask, core_mask)

    # 4. Sortăm ce a fost tăiat: aruncăm bețele, punem înapoi colțurile fructului
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(removed_parts, connectivity=8)

    final_mask = core_mask.copy()

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 100:
            # Firimituri mici, le punem la loc
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

        # Regula: Dacă partea tăiată este lungă (băț), o ignorăm (rămâne ștearsă).
        # Dacă este o margine rotunjită tăiată din greșeală (ex: marginea mărului), o punem la loc.
        if aspect_ratio < 2.2:
            final_mask[labels == label] = 255

    # Curățăm și netezim marginile finale ca să arate natural
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_smooth)

    return final_mask
# ══════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL & SEGMENTARE
# ══════════════════════════════════════════════════════════════════

def apply_segmentation_pipeline(img) -> Image.Image:
    # Pasul 1: Extragerea brută a fundalului
    rgba_img = remove(img)
    alpha = np.array(rgba_img.split()[3])
    binary_mask = (alpha > 128).astype(np.uint8) * 255

    # Pasul 2: Eliminarea inteligentă a bețelor (funcționează la orice unghi)
    clean_mask = remove_stick_universal(binary_mask)

    # Pasul 3: Curățare finală (păstrăm doar fructul, închidem marginile tăiate)
    clean_mask = keep_largest_component(clean_mask)
    clean_mask = fill_holes(clean_mask)

    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=2)
    clean_mask = keep_largest_component(clean_mask)

    # Aplicăm masca finală pe imaginea originală
    r, g, b = img.convert("RGB").split()
    final_rgba = Image.merge("RGBA", (r, g, b, Image.fromarray(clean_mask, mode="L")))

    return composite_on_white(final_rgba)

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def process_image(input_path, output_folder) -> None:
    print(f"  Procesez: {input_path.name} ...", end=" ", flush=True)

    pil_img = Image.open(input_path).convert("RGB")
    result = apply_segmentation_pipeline(pil_img)

    out_path = output_folder / f"{input_path.stem}_segmented.jpg"
    result.save(out_path, "JPEG", quality=95)
    print(f"✓  → {out_path.name}")

def main():
    input_folder = Path(INPUT_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(exist_ok=True)

    images = [p for p in input_folder.iterdir() if p.suffix.upper() in EXTENSION]

    if not images:
        print(f"Nicio imagine găsită în '{input_folder}'.")
        sys.exit(1)

    print(f"Imagini găsite: {len(images)}\n")
    for img_path in sorted(images):
        process_image(img_path, output_folder)

    print(f"\n Fructele au fost decupate pe fundal alb.")

if __name__ == "__main__":
    main()