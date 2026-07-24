"""Merge all 5 attack types into one combined training set, preserving
which attack each image came from (for later per-attack evaluation).

Creates:
    data/adversarial_all_v2/fgsm/<class>/...
    data/adversarial_all_v2/pgd/<class>/...
    data/adversarial_all_v2/bim/<class>/...
    data/adversarial_all_v2/cw/<class>/...
    data/adversarial_all_v2/deepfool/<class>/...

Uses copy2 (not move) so the original per-attack folders stay intact.
"""

import shutil
from pathlib import Path

OUTPUT_DIR = Path("data/adversarial_all_v2")

# (attack_name, source_dir, filename_prefix_to_filter_by_or_None)
SOURCES = [
    ("fgsm", Path("data/adversarial_v2"), "fgsm_"),
    ("pgd", Path("data/adversarial_v2"), "pgd_"),
    ("bim", Path("data/bim_adversarial_v2"), None),
    ("cw", Path("data/cw_adversarial_v2"), None),
    ("deepfool", Path("data/deepfool_adversarial_v2"), None),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for attack_name, source_dir, prefix in SOURCES:
        if not source_dir.exists():
            print(f"WARNING: {source_dir} does not exist, skipping {attack_name}")
            summary[attack_name] = 0
            continue

        dest_dir = OUTPUT_DIR / attack_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for src_path in source_dir.rglob("*"):
            if not src_path.is_file():
                continue
            if src_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if prefix is not None and not src_path.name.startswith(prefix):
                continue

            rel_path = src_path.relative_to(source_dir)
            dest_path = dest_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            count += 1

        summary[attack_name] = count
        print(f"{attack_name:10s}: copied {count} files -> {dest_dir}")

    grand_total = sum(summary.values())
    print(f"\nGrand total: {grand_total} files in {OUTPUT_DIR}")
    if grand_total != 6250:
        print(f"WARNING: expected 6250 (1250 x 5), got {grand_total}")


if __name__ == "__main__":
    main()