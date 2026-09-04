#!/usr/bin/env python3
"""Seed Harborline Internal: tickets, ops account, payroll artifact."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from werkzeug.security import generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
APP_ROOT = BASE_DIR.parent
DATA_DIR = APP_ROOT / "data"
FILES_DIR = DATA_DIR / "files"
DB_PATH = DATA_DIR / "lab.db"
JOB_ID = "7f3c1e2a-9b44-4d21-a8c0-harbor4412"
PAYROLL_TOKEN = "PAYROLL_SECRET=harborline-hl4412-2026"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'guest'
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    author TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS file_jobs (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    path TEXT NOT NULL
);
"""

NEWS = [
    ("Q3 uptime", "Public status page is the source of truth. Internal runbooks are staff-only."),
    ("Helpdesk hours", "Open 9-5. File a ticket if VPN is down; do not email payroll."),
    ("Phishing drill", "IT will never send passwords in email. Reset via the portal only."),
]

TICKETS = [
    (1, "VPN timeout", "Reconnect and clear the cached profile. Assigned: netops.", "alice"),
    (2, "Badge printer", "Lobby printer jammed again.", "bob"),
    (3, "Calendar sync", "Works after re-login.", "alice"),
    (12, "Laptop imaging", "New hire image is 24h SLA.", "ithelp"),
    (18, "Wifi 4th floor", "AP-14 is flapping, ticket with facilities.", "bob"),
    (27, "Okta push delay", "Known IdP lag, wait 30s.", "alice"),
    (4412, "ops.monitor password rotation",
     "Quarterly rotation complete. Service account ops.monitor / Harborline!4412 "
     "is for the files job runner only. Do not paste this in Slack.",
     "it-sec"),
]


def seed() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.executescript(SCHEMA)

    if db.execute("SELECT COUNT(*) FROM posts").fetchone()[0] == 0:
        db.executemany("INSERT INTO posts (title, body) VALUES (?, ?)", NEWS)

    if db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0:
        db.executemany(
            "INSERT INTO tickets (id, title, body, author) VALUES (?, ?, ?, ?)",
            TICKETS,
        )

    if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        db.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'guest')",
            ("alice", generate_password_hash("alice2024")),
        )
        db.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'guest')",
            ("bob", generate_password_hash("sunshine1")),
        )
        db.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'ops')",
            ("ops.monitor", generate_password_hash("Harborline!4412")),
        )

    payroll = FILES_DIR / f"{JOB_ID}.csv"
    if not payroll.exists():
        rows = ["emp_id,name,dept,comp_code\n"]
        for i in range(1, 1800):
            rows.append(f"{10000+i},employee_{i},ops,HL-{i%90:03d}\n")
        rows.append(f"99999,payroll_export,{PAYROLL_TOKEN},CONFIDENTIAL\n")
        payroll.write_text("".join(rows), encoding="utf-8")

    if db.execute("SELECT COUNT(*) FROM file_jobs").fetchone()[0] == 0:
        db.execute(
            "INSERT INTO file_jobs (id, label, path) VALUES (?, ?, ?)",
            (JOB_ID, "payroll-export-weekly", str(payroll)),
        )

    db.commit()
    db.close()


if __name__ == "__main__":
    seed()
    print("seeded harborline db + payroll artifact", flush=True)
