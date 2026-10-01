"""Shared loader for the source/transform coverage tests (synthetic, local mechanics only)."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("stl_ledger", Path(__file__).resolve().parents[1] / "tools/source_transform_ledger.py")
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)


def make_tree(root: Path, files):
    for rel, payload in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
    return root
