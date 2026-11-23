"""Wrapper around TabPFN (Hollmann et al., Nature 2025).

OWNER: B (Thu). Stub with the agreed interface only (see src/interfaces.py).
Must raise a clear ValueError when the data exceeds 10,000 samples,
500 features or 10 classes.
"""
from __future__ import annotations

from src.models.nn import _NNStub


class TabPFNModel(_NNStub):
    pass
