import random
import shutil
from pathlib import Path

# =========================
# CONFIGURARE
# =========================
BASE_DIR = Path(r"D:\Master\ACABI\Clasificare_fructe\output_100x100")
TRAIN_DIR = BASE_DIR / "Training"
TEST_DIR = BASE_DIR / "Test"

# extensii acceptate
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# seed pentru rezultate reproductibile
random.seed(42)

# =========================
# CREARE FOLDERE OUTPUT
# =========================
TRAIN_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

# =========================
# PARCURGERE FOLDERE FRUCTE
# =========================
for fruit_folder in BASE_DIR.iterdir():
    if not fruit_folder.is_dir():
        continue

    # ignoram folderele deja existente de output
    if fruit_folder.name in {"Training", "Test"}:
        continue

    # luam toate imaginile din folderul curent
    images = [
        f for f in fruit_folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print(f"[SKIP] Folder gol sau fara imagini: {fruit_folder.name}")
        continue

    # amestecare aleatoare
    random.shuffle(images)

    # impartire 75% / 25%
    split_index = int(len(images) * 0.75)

    train_images = images[:split_index]
    test_images = images[split_index:]

    # creare subfoldere cu acelasi nume
    train_fruit_dir = TRAIN_DIR / fruit_folder.name
    test_fruit_dir = TEST_DIR / fruit_folder.name

    train_fruit_dir.mkdir(parents=True, exist_ok=True)
    test_fruit_dir.mkdir(parents=True, exist_ok=True)

    # copiere imagini
    for img_path in train_images:
        shutil.copy2(img_path, train_fruit_dir / img_path.name)

    for img_path in test_images:
        shutil.copy2(img_path, test_fruit_dir / img_path.name)

    print(
        f"[OK] {fruit_folder.name}: total={len(images)}, "
        f"training={len(train_images)}, test={len(test_images)}"
    )

print("\nGata. Imaginile au fost impartite in folderele Training si Test.")