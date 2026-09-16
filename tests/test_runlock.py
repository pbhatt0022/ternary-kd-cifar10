"""The run lock. Torch-free, so it runs anywhere."""

import os
import time

import pytest

from src import runlock


def test_acquire_creates_lock(tmp_path):
    runlock.acquire(tmp_path, "D_T2_0")
    assert (tmp_path / runlock.LOCK_NAME).exists()


def test_fresh_lock_blocks_a_second_writer(tmp_path):
    runlock.acquire(tmp_path, "D_T2_0")
    with pytest.raises(runlock.RunLockedError):
        runlock.acquire(tmp_path, "D_T2_0")


def test_stale_lock_is_treated_as_abandoned(tmp_path):
    """A preempted VM never deletes its lock; once it stops refreshing, the run is free."""
    runlock.acquire(tmp_path, "D_T2_0")
    old = time.time() - runlock.STALE_SECONDS - 60
    os.utime(tmp_path / runlock.LOCK_NAME, (old, old))
    runlock.acquire(tmp_path, "D_T2_0")  # must not raise


def test_release_frees_the_run(tmp_path):
    runlock.acquire(tmp_path, "D_T2_0")
    runlock.release(tmp_path)
    assert not (tmp_path / runlock.LOCK_NAME).exists()
    runlock.acquire(tmp_path, "D_T2_0")


def test_release_is_idempotent(tmp_path):
    runlock.release(tmp_path)
    runlock.release(tmp_path)


def test_heartbeat_keeps_a_live_run_locked(tmp_path):
    runlock.acquire(tmp_path, "D_T2_0")
    old = time.time() - runlock.STALE_SECONDS - 60
    os.utime(tmp_path / runlock.LOCK_NAME, (old, old))
    runlock.heartbeat(tmp_path, "D_T2_0")  # the owning process refreshes it
    with pytest.raises(runlock.RunLockedError):
        runlock.acquire(tmp_path, "D_T2_0")


def test_error_message_names_the_holder(tmp_path):
    runlock.acquire(tmp_path, "D_T2_0")
    with pytest.raises(runlock.RunLockedError, match="already training elsewhere"):
        runlock.acquire(tmp_path, "D_T2_0")
