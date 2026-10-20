"""abkit - Bayesian experimentation and uplift modelling, built on a trustworthy A/B toolkit."""

from .pipeline import Settings, analyze, analyze_all
from .scenarios import SCENARIOS

__all__ = ["Settings", "analyze", "analyze_all", "SCENARIOS"]
__version__ = "0.2.0"
