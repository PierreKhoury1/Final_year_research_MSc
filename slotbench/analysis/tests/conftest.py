"""Shared fixtures: one synthetic run tree per test session (copied before any test mutates it)."""
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis import synth  # noqa: E402

N_SLOTS = 6000


@pytest.fixture(scope="session")
def synth_tree_ro(tmp_path_factory):
    """Read-only synthetic tree: <tmp>/runs/synth/... (3 mechanisms x 2 workloads x d0/50/100 + SOLO + M3 idle)."""
    base = tmp_path_factory.mktemp("synth") / "runs"
    synth.make_tree(str(base), n_slots=N_SLOTS, seed=3)
    return base


@pytest.fixture
def synth_tree(synth_tree_ro, tmp_path):
    """Private writable copy of the synthetic tree."""
    dst = tmp_path / "runs"
    shutil.copytree(synth_tree_ro, dst)
    return dst
