"""
Resize 100x100 – Baza de date augmentată
-----------------------------------------
Parcurge toate subfoldere (clasele) din INPUT_DIR
și redimensionează fiecare imagine la 100x100 px.
"""

import sys
from pathlib import Path
from PIL import Image

# ──────────────────────────────────────────────
# CONFIGURARE
# ──────────────────────────────────────────────

INPUT_DIR  = r"D:\Master\ACABI\Clasificare_fructe\baza de date augmented"
OUTPUT_DIR = r"D:\Master\ACABI\Clasificare_fructe\baza de date augmented 100x100"

OUTPUT_SIZE = (100, 100)
EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

# ──────────────────────────────────────────────
# PROCESARE
# ──────────────────────────────────────────────

def main():
    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    if not input_path.exists():
        print(f"[EROARE] Folderul nu există: {input_path}")
        sys.exit(1)

    # Colectează toate imaginile din toate subfoldere (clasele)
    all_images = []
    for class_dir in sorted(input_path.iterdir()):
        if class_dir.is_dir():
            for img_file in class_dir.iterdir():
                if img_file.suffix in EXTENSIONS:
                    all_images.append((class_dir.name, img_file))

    if not all_images:
        print("[EROARE] Nicio imagine găsită.")
        sys.exit(1)

    print(f"{'='*55}")
    print(f"  RESIZE → {OUTPUT_SIZE[0]}×{OUTPUT_SIZE[1]}px")
    print(f"{'='*55}")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Total imagini: {len(all_images)}\n")

    ok, err = 0, 0

    for class_name, img_path in all_images:
        # Creează subfolderul clasei în output
        out_class_dir = output_path / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(img_path) as img:
                resized = img.convert("RGB").resize(OUTPUT_SIZE, Image.LANCZOS)
                out_path = out_class_dir / img_path.name
                resized.save(out_path, "JPEG", quality=95)
            ok += 1
        except Exception as e:
            print(f"  [EROARE] {class_name}/{img_path.name}: {e}")
            err += 1

    print(f"{'='*55}")
    print(f"  FINALIZAT: {ok} procesate, {err} erori")
    print(f"  Folder output: {output_path}")
    print(f"{'='*55}")

    # Statistici per clasă
    print("\nDistribuție finală per clasă:")
    for class_dir in sorted(output_path.iterdir()):
        if class_dir.is_dir():
            n = len([f for f in class_dir.iterdir() if f.suffix in EXTENSIONS])
            print(f"  {class_dir.name:<25} {n:>5} imagini")


if __name__ == "__main__":
    main()