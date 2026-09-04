#!/usr/bin/env python3
"""Harborline Internal — stepwise lab target (not one-shot SQLi).

Steal path (must be executed in order; none of this is on the homepage):

  1. Recon: many services listen besides HTTP (scan is visible in 5s flows).
  2. Enum: dir bust finds /internal/runbook (auth) and /tickets.
  3. Register/login as a guest, then IDOR sequential /tickets/<id> until 4412
     (CWE-639). That ticket has the ops.monitor password.
  4. Re-login as ops.monitor.
  5. /files/jobs lists a backup job UUID; download the artifact (payroll CSV
     containing PAYROLL_SECRET=...). Search is parameterized — UNION does nothing.

Benign bots only hit public pages. Forecast should rise during (1)–(2), before (5).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from flask import Flask, abort, g, redirect, render_template_string, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
APP_ROOT = BASE_DIR.parent
DATA_DIR = APP_ROOT / "data"
DB_PATH = DATA_DIR / "lab.db"

app = Flask(__name__)
app.secret_key = "harborline-lab-session"

LAYOUT = """<!doctype html>
<html><head><meta charset="utf-8"/><title>{{ title }} — Harborline</title>
<style>
  :root { --ink:#102a43; --muted:#627d98; --line:#d9e2ec; --acc:#0b6e4f; }
  body { margin:0; font-family: Segoe UI, sans-serif; background:#f7f9fb; color:var(--ink); }
  header { background:#102a43; color:#fff; padding:14px 28px; display:flex; justify-content:space-between; }
  header a { color:#b6eadd; margin-left:14px; text-decoration:none; font-size:14px; }
  main { max-width:860px; margin:28px auto; background:#fff; border:1px solid var(--line);
         border-radius:10px; padding:28px 32px; }
  h1 { margin-top:0; }
  input, button { font-size:15px; padding:8px 10px; }
  .muted { color:var(--muted); }
  .card { border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:10px 0; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  td, th { border-bottom:1px solid var(--line); padding:8px; text-align:left; }
</style></head>
<body>
<header>
  <strong>Harborline Internal</strong>
  <nav>
    <a href="/">Home</a>
    <a href="/careers">Careers</a>
    <a href="/status">Status</a>
    <a href="/search">News</a>
    <a href="/tickets">Tickets</a>
    <a href="/files/jobs">Files</a>
    <a href="/login">Login</a>
    <a href="/register">Register</a>
  </nav>
</header>
<main>
  <h1>{{ title }}</h1>
  {{ body|safe }}
</main>
</body></html>"""


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
    return get_db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()


def page(title: str, body: str) -> str:
    return render_template_string(LAYOUT, title=title, body=body)


@app.route("/")
def index():
    return page(
        "Employee portal",
        "<p>Welcome to Harborline. Public news and careers are below. "
        "Payroll, runbooks, and file jobs are staff-only.</p>"
        "<p class='muted'>Need help? Open a ticket after you sign in.</p>",
    )


@app.route("/careers")
def careers():
    return page(
        "Careers",
        "<p>We are hiring netops and helpdesk. Apply through HR — this page "
        "does not list service accounts.</p>",
    )


@app.route("/status")
def status():
    return page(
        "Public status",
        "<p>All customer systems operational.</p>"
        "<p class='muted'>Internal SLO runbooks are not on this page.</p>",
    )


@app.route("/robots.txt")
def robots():
    return "User-agent: *\nDisallow: /internal/\nDisallow: /files/\n", 200, {
        "Content-Type": "text/plain"
    }


@app.route("/api/health")
def health():
    return {"status": "ok", "service": "harborline-web"}


@app.route("/search")
def search():
    q = request.args.get("q", "")
    rows = []
    if q:
        cur = get_db().execute(
            "SELECT title, body FROM posts WHERE title LIKE ? OR body LIKE ?",
            (f"%{q}%", f"%{q}%"),
        )
        rows = cur.fetchall()
    items = "".join(f"<div class='card'><b>{r['title']}</b><p>{r['body']}</p></div>" for r in rows)
    return page(
        "News search",
        f"<form method='get'><input name='q' value='{q}' placeholder='search news'>"
        f"<button type='submit'>Search</button></form>{items or '<p class=muted>No posts.</p>'}",
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user = (request.form.get("username") or "").strip()
        pw = request.form.get("password") or ""
        if not user or not pw:
            return page("Register", "<p>username and password required</p>")
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'guest')",
                (user, generate_password_hash(pw)),
            )
            db.commit()
        except sqlite3.IntegrityError:
            return page("Register", "<p>username taken</p>")
        return redirect(url_for("login"))
    return page(
        "Register",
        "<form method='post'><p><input name='username' placeholder='username'></p>"
        "<p><input type='password' name='password' placeholder='password'></p>"
        "<button>Create guest account</button></form>",
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    err = ""
    if request.method == "POST":
        user = (request.form.get("username") or "").strip()
        pw = request.form.get("password") or ""
        row = get_db().execute("SELECT * FROM users WHERE username = ?", (user,)).fetchone()
        if row and check_password_hash(row["password_hash"], pw):
            session["user_id"] = row["id"]
            return redirect(url_for("index"))
        err = "<p>invalid credentials</p>"
    return page(
        "Login",
        err
        + "<form method='post'><p><input name='username' placeholder='username'></p>"
        "<p><input type='password' name='password' placeholder='password'></p>"
        "<button>Sign in</button></form>",
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/internal/runbook")
def runbook():
    if current_user() is None:
        return page("Runbook", "<p>401 — staff login required.</p>"), 401
    return page(
        "Internal runbook",
        "<p>File-job runner is documented in the helpdesk ticket stream. "
        "Look up the quarterly rotation ticket (ID is four digits, filed by it-sec).</p>"
        "<p class='muted'>This page does not print passwords.</p>",
    )


@app.route("/tickets")
def tickets_index():
    if current_user() is None:
        return redirect(url_for("login"))
    return page(
        "My tickets",
        "<p>Open a ticket by id in the URL bar if you were cc'd: "
        "<code>/tickets/&lt;id&gt;</code></p>"
        "<p class='muted'>The list view only shows tickets you authored. "
        "Direct links are not re-checked.</p>",
    )


@app.route("/tickets/<int:tid>")
def ticket_detail(tid: int):
    # CWE-639: any authenticated user can read any ticket by id.
    if current_user() is None:
        return redirect(url_for("login"))
    row = get_db().execute("SELECT * FROM tickets WHERE id = ?", (tid,)).fetchone()
    if row is None:
        abort(404)
    return page(
        f"Ticket {row['id']}",
        f"<div class='card'><b>{row['title']}</b><p>Author: {row['author']}</p>"
        f"<p>{row['body']}</p></div>",
    )


@app.route("/files/jobs")
def file_jobs():
    user = current_user()
    if user is None:
        return redirect(url_for("login"))
    if user["role"] not in ("ops", "admin"):
        return page("Files", "<p>403 — ops role required for file jobs.</p>"), 403
    rows = get_db().execute("SELECT id, label FROM file_jobs").fetchall()
    items = "".join(
        f"<tr><td>{r['label']}</td><td><a href='/files/jobs/{r['id']}/artifact'>{r['id']}</a></td></tr>"
        for r in rows
    )
    return page("File jobs", f"<table><tr><th>job</th><th>artifact</th></tr>{items}</table>")


@app.route("/files/jobs/<job_id>/artifact")
def file_artifact(job_id: str):
    user = current_user()
    if user is None:
        return redirect(url_for("login"))
    if user["role"] not in ("ops", "admin"):
        abort(403)
    row = get_db().execute("SELECT path, label FROM file_jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        abort(404)
    path = Path(row["path"])
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"{row['label']}.csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
