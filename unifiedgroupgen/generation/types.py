"""Shared sampling results, independent of the selected network architecture."""
from dataclasses import dataclass

@dataclass
class JointSample:
    state: object
    time: float
    events: list
    reason: str
    integration_steps: int
    epsilon: float
