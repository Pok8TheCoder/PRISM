"""WebSocket ↔ PTY bridge for Lab Console shell tab."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

ROOT = Path(__file__).resolve().parents[4]


def _default_shell() -> str:
    if sys.platform == "win32":
        return os.environ.get("COMSPEC", "cmd.exe")
    return os.environ.get("SHELL", "/bin/bash")


def _parse_resize(text: str) -> tuple[int, int] | None:
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return None
    if msg.get("type") != "resize":
        return None
    cols = int(msg.get("cols", 80))
    rows = int(msg.get("rows", 24))
    return max(rows, 2), max(cols, 20)


async def relay_terminal(ws: WebSocket) -> None:
    await ws.accept()
    cwd = str(ROOT)
    shell = _default_shell()

    try:
        if sys.platform == "win32":
            await _relay_winpty(ws, shell, cwd)
        else:
            await _relay_subprocess(ws, shell, cwd)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_text(f"\r\n\x1b[31m[terminal] {exc}\x1b[0m\r\n")
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


async def _relay_winpty(ws: WebSocket, shell: str, cwd: str) -> None:
    from winpty import PtyProcess

    pty = PtyProcess.spawn(shell, cwd=cwd, dimensions=(24, 120))
    closed = asyncio.Event()

    async def read_pty() -> None:
        while not closed.is_set() and pty.isalive():
            try:
                chunk = await asyncio.to_thread(pty.read, 4096)
            except (EOFError, OSError):
                break
            if not chunk:
                await asyncio.sleep(0.02)
                continue
            data = chunk.encode("utf-8", errors="replace") if isinstance(chunk, str) else chunk
            await ws.send_bytes(data)

    async def write_ws() -> None:
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    text = msg["bytes"].decode("utf-8", errors="replace")
                    if pty.isalive():
                        pty.write(text)
                elif msg.get("text") is not None:
                    text = msg["text"]
                    size = _parse_resize(text)
                    if size and pty.isalive():
                        pty.setwinsize(size[0], size[1])
                        continue
                    if pty.isalive():
                        pty.write(text)
        finally:
            closed.set()
            if pty.isalive():
                pty.terminate(force=True)

    await asyncio.gather(read_pty(), write_ws())


async def _relay_subprocess(ws: WebSocket, shell: str, cwd: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        shell,
        cwd=cwd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "TERM": "xterm-256color"},
    )
    assert proc.stdin and proc.stdout

    async def read_stdout() -> None:
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            await ws.send_bytes(chunk)

    async def write_ws() -> None:
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    proc.stdin.write(msg["bytes"])
                    await proc.stdin.drain()
                elif msg.get("text") is not None:
                    size = _parse_resize(msg["text"])
                    if size:
                        continue
                    proc.stdin.write(msg["text"].encode("utf-8"))
                    await proc.stdin.drain()
        finally:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    proc.kill()

    await asyncio.gather(read_stdout(), write_ws())
