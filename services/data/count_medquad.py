import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from app.infrastructure.loaders.medquad_loader import MedQuADLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
medquad_dir = PROJECT_ROOT / "data" / "raw" / "medquad"

print(f"Counting MedQuAD with all {sum(1 for _ in medquad_dir.rglob('*.xml'))} XML files...")

loader = MedQuADLoader(str(medquad_dir), log_every=2000)

total = 0
for doc in loader.load():
    total += 1

print(f"\n{'='*50}")
print(f"[RESULT] Total valid MedQuAD documents (with answer): {total}")
print(f"{'='*50}")
