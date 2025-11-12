"""
Agent Version Registry - stub implementation for development startup
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentVersionInfo:
    version: str = "0.0.0"
    notes: Optional[str] = None

class AgentVersionRegistry:
    def __init__(self) -> None:
        self._current = AgentVersionInfo()

    def current(self) -> AgentVersionInfo:
        return self._current

    def update(self, info: AgentVersionInfo) -> None:
        self._current = info

