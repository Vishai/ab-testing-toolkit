"""abkit - a small, trustworthy A/B test analysis toolkit."""

from .pipeline import Settings, analyze, analyze_all
from .scenarios import SCENARIOS

__all__ = ["Settings", "analyze", "analyze_all", "SCENARIOS"]
__version__ = "0.1.0"
