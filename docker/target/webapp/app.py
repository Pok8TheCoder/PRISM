#!/usr/bin/env python3
"""Lab target web app.

A small, genuinely-vulnerable Flask app used as the attack surface for the
PRISM adversarial lab. It is not a toy label generator: every vulnerability
below is a real bug class (CWE) that a bot must actually exploit over HTTP
to succeed, and every "success" is independently re-verified server-side by
the caller (see src/adversarial/bots/lab_*.py) by re-reading real state
(the homepage row, the real per-boot secret file, the real users table).

Three objectives, three independent bugs, one common root cause: the
developer added an auth *check* but never a *role* check, and never
parameterized a query.

  1. Write access / defacement  -> CWE-862 Missing Authorization
     POST /admin/edit_page only checks "are you logged in", not "are you
     an admin". Any freshly self-registered guest can deface the homepage.

  2. Key theft (read)           -> CWE-22 Path Traversal
     GET /download?file=... is a legit "download your backup" feature.
     The filename is joined onto BACKUP_DIR with no sanitization, so
     "../../config/secret.key" escapes the backups folder and reaches the
     real per-boot Flask session-signing secret.

  3. Credential theft (read)    -> CWE-89 SQL Injection
     GET /search?q=... builds raw SQL with an f-string against the posts
     table. A UNION SELECT pulls username/password_hash pairs out of the
     real users table through the same 2-column result shape.
"""

import os
import secrets
import sqlite3
from pathlib import Path

from flask import Flask, request, session, redirect, url_for, g, abort, send_file
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).resolve().parent
APP_ROOT = BASE_DIR.parent
CONFIG_DIR = APP_ROOT / "config"
DATA_DIR = APP_ROOT / "data"
BACKUP_DIR = DATA_DIR / "backups"
DB_PATH = DATA_DIR / "lab.db"
SECRET_PATH = CONFIG_DIR / "secret.key"

for _d in (CONFIG_DIR, DATA_DIR, BACKUP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

if not SECRET_PATH.exists():
    SECRET_PATH.write_text(secrets.token_hex(32))

app = Flask(__name__)
app.secret_key = SECRET_PATH.read_text().strip()


def get_db():
    db = getattr(g, "_db", None)
    if db is None:
        db = g._db = sqlite3.connect(str(DB_PATH))
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_db(_exc):
    db = getattr(g, "_db", None)
    if db is not None:
        db.close()


def current_user():
    uid = session.get("user_id")
    if uid is None:
        return None
    row = get_db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    return row


PAGE_TEMPLATE = """<!doctype html>
<html><head><title>{title}</title></head>
<body>
{body}
</body></html>"""


def render(title, body):
    return PAGE_TEMPLATE.format(title=title, body=body)


@app.route("/")
def index():
    row = get_db().execute("SELECT content FROM homepage WHERE id = 1").fetchone()
    content = row["content"] if row else "<h1>Welcome</h1>"
    user = current_user()
    nav = '<p><a href="/register">register</a> | <a href="/login">login</a> | <a href="/search">search</a></p>'
    if user is not None:
        nav = f'<p>logged in as {user["username"]} | <a href="/admin/edit_page">edit page</a> | <a href="/logout">logout</a></p>'
    return render("PRISM Lab Co.", nav + content)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render(
            "Register",
            """<form method="post">
            <input name="username" placeholder="username"><br>
            <input name="password" type="password" placeholder="password"><br>
            <button type="submit">Register</button></form>""",
        )
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    if not username or not password:
        return render("Register", "<p>username and password required</p>"), 400
    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing is not None:
        return render("Register", "<p>username taken</p>"), 400
    # New self-registration always lands in the unprivileged "guest" role.
    db.execute(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'guest')",
        (username, generate_password_hash(password)),
    )
    db.commit()
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render(
            "Login",
            """<form method="post">
            <input name="username" placeholder="username"><br>
            <input name="password" type="password" placeholder="password"><br>
            <button type="submit">Login</button></form>""",
        )
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None or not check_password_hash(row["password_hash"], password):
        return render("Login", "<p>invalid credentials</p>"), 401
    session["user_id"] = row["id"]
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# --- CWE-862: Missing Authorization -----------------------------------
# Checks "is somebody logged in" but never checks role == "admin", so any
# self-registered guest account can deface the homepage.
@app.route("/admin/edit_page", methods=["GET", "POST"])
def admin_edit_page():
    user = current_user()
    if user is None:
        return redirect(url_for("login"))
    if request.method == "GET":
        return render(
            "Edit homepage",
            """<form method="post">
            <textarea name="content" rows="6" cols="60"></textarea><br>
            <button type="submit">Save</button></form>""",
        )
    content = request.form.get("content", "")
    db = get_db()
    db.execute("UPDATE homepage SET content = ? WHERE id = 1", (content,))
    db.commit()
    return redirect(url_for("index"))


# --- CWE-22: Path Traversal --------------------------------------------
# Legit "download your backup" feature for logged-in users. The filename
# is joined onto BACKUP_DIR without any sanitization/allowlist, so a
# "../../config/secret.key" style value escapes the backups folder.
@app.route("/download")
def download():
    user = current_user()
    if user is None:
        return redirect(url_for("login"))
    filename = request.args.get("file", "site_backup_2026.txt")
    path = os.path.join(str(BACKUP_DIR), filename)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path)


# --- CWE-89: SQL Injection ----------------------------------------------
# Public search over blog posts, built with a raw f-string against
# sqlite3. A UNION SELECT against the 2 text columns (title, body) can
# pull username/password_hash pairs out of the real users table.
@app.route("/search")
def search():
    q = request.args.get("q", "")
    db = get_db()
    rows = []
    error = None
    try:
        cur = db.execute(f"SELECT title, body FROM posts WHERE title LIKE '%{q}%' OR body LIKE '%{q}%'")
        rows = cur.fetchall()
    except sqlite3.Error as exc:
        error = str(exc)
    items = "".join(f"<li><b>{r['title']}</b>: {r['body']}</li>" for r in rows)
    body = f"""<form method="get">
    <input name="q" value="{q}" placeholder="search posts"><button type="submit">Search</button></form>
    <ul>{items}</ul>"""
    if error:
        body += f"<p>query error: {error}</p>"
    return render("Search", body)


@app.route("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
