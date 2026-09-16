"""Best-effort guard against two machines training the same run at once.

Several Colab notebooks share one Drive folder. Two processes appending to the same run's
files would silently corrupt it, and on a network filesystem that corruption shows up later
as missing or interleaved epochs. A running run keeps a lock file fresh by rewriting it every
epoch; a second process that finds a fresh lock refuses to start.

This is deliberately heartbeat-based rather than a strict mutex. A preempted VM never gets to
delete its lock, so a lock that has not been refreshed within STALE_SECONDS is treated as
abandoned. The guard catches the realistic failure -- a notebook reopened later and pointed
at a run another notebook is still training -- but not two processes started within the few
seconds Drive takes to propagate a write.
"""

import json
import os
import socket
import time

LOCK_NAME = "running.lock"
STALE_SECONDS = 300  # longest epoch is ~70 s, so a live run refreshes this several times over


class RunLockedError(RuntimeError):
    """Raised when a run appears to be training in another process right now."""


def _path(run_dir):
    return run_dir / LOCK_NAME


def acquire(run_dir, label, stale_seconds=STALE_SECONDS, now=None):
    """Claim the run, or raise RunLockedError if another process holds a fresh lock."""
    now = time.time() if now is None else now
    path = _path(run_dir)
    if path.exists():
        age = now - path.stat().st_mtime
        if age < stale_seconds:
            try:
                holder = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                holder = {}
            raise RunLockedError(
                f"{label} looks like it is already training elsewhere: its lock was refreshed "
                f"{age:.0f}s ago by host {holder.get('host', '?')}. Refusing to start a second "
                f"writer, which would corrupt the run. If you are CERTAIN nothing else is "
                f"training it (for example you just reconnected after a disconnect), wait "
                f"{max(0, stale_seconds - age):.0f}s and retry, or delete {path}."
            )
    heartbeat(run_dir, label)


def heartbeat(run_dir, label):
    """Refresh the lock. Called once per epoch by the training loop."""
    _path(run_dir).write_text(
        json.dumps({
            "label": label,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "time": time.time(),
        }),
        encoding="utf-8",
    )


def release(run_dir):
    """Drop the lock. Safe to call when it is already gone."""
    try:
        _path(run_dir).unlink()
    except FileNotFoundError:
        pass
