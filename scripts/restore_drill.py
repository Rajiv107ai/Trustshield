#!/usr/bin/env python3
"""Run a local durable-audit restore drill.

This script exercises the development SQLite path only. Production restore
drills must run against the managed PostgreSQL backup/restore process and are
not simulated by this command.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from backend.services.audit_service import AuditService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="SQLite audit database")
    parser.add_argument("--backup", type=Path, required=True, help="Backup copy path")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    backup = args.backup.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"source audit database does not exist: {source}")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, backup)
    restored = AuditService(db_path=backup, environment="development")
    try:
        report = restored.verify_integrity()
        if not report.valid:
            raise SystemExit(f"restore integrity check failed: {report.error}")
        print(f"restore drill passed: records={report.checked_records} backup={backup}")
    finally:
        restored.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
