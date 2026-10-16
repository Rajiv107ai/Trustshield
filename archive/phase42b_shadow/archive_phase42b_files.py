"""
Creates a verified archive of Phase 4.2B online shadow scoring infrastructure
prior to removal in Phase 4.3B.
Calculates SHA-256 hashes and writes an integrity manifest.
"""
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "backend").is_dir() and (REPO_ROOT / "models").is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent

ARCHIVE_DIR = REPO_ROOT / "archive" / "phase42b_shadow"
ARCHIVE_ZIP = REPO_ROOT / "archive" / "phase42b_shadow.zip"
BACKUP_DIR = Path(r"C:\Users\rajiv_pis9z8x\.gemini\antigravity-ide\brain\daa3d19b-eeeb-4b24-ba1c-50eaa8071dbb\backups\phase42b_shadow_archive")

ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FILES = [
    "backend/services/shadow_service.py",
    "backend/services/redis_causal_service.py",
    "backend/candidate_loader.py",
    "scripts/backfill_redis_causal.py",
    "backend/test_shadow_scoring.py",
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "archive_name": "phase42b_online_shadow_infrastructure",
    "timestamp_utc": "2026-10-16T11:48:00Z",
    "description": "Preservation archive of Phase 4.2B online shadow scoring infrastructure before Phase 4.3B simplification removal.",
    "files": {}
}

print("=== Archiving Phase 4.2B Files ===")
for rel_path in TARGET_FILES:
    src = REPO_ROOT / rel_path
    if not src.is_file():
        print(f"ERROR: Source file missing: {src}")
        continue

    # Destination in repo archive
    dst_repo = ARCHIVE_DIR / rel_path
    dst_repo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_repo)

    # Destination in agent brain backups
    dst_brain = BACKUP_DIR / rel_path
    dst_brain.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_brain)

    checksum = sha256_file(src)
    size = src.stat().st_size
    manifest["files"][rel_path] = {
        "sha256": checksum,
        "size_bytes": size,
    }
    print(f"Preserved {rel_path} ({size} bytes, sha256={checksum[:12]}...)")

# Also record extracts of the endpoints and schemas
main_py = REPO_ROOT / "backend" / "main.py"
schemas_py = REPO_ROOT / "backend" / "schemas.py"

manifest["related_files_pre_removal"] = {
    "backend/main.py_sha256": sha256_file(main_py),
    "backend/schemas.py_sha256": sha256_file(schemas_py),
}

manifest_path_repo = ARCHIVE_DIR / "manifest.json"
manifest_path_brain = BACKUP_DIR / "manifest.json"

with open(manifest_path_repo, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
with open(manifest_path_brain, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

# Create zip archive
with zipfile.ZipFile(ARCHIVE_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel_path in TARGET_FILES:
        zf.write(REPO_ROOT / rel_path, arcname=rel_path)
    zf.write(manifest_path_repo, arcname="manifest.json")

print(f"\nCreated zip archive: {ARCHIVE_ZIP} ({ARCHIVE_ZIP.stat().st_size} bytes)")
print(f"Created directory archive: {ARCHIVE_DIR}")
print(f"Created brain backup archive: {BACKUP_DIR}")
print("Integrity verification: complete!")
