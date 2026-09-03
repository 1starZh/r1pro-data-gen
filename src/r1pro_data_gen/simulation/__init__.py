"""Simulation backend interfaces with simulator imports kept in submodules."""

from .evidence_recorder import EvidenceAdapter, EvidenceRecorder

__all__ = ["EvidenceAdapter", "EvidenceRecorder"]
