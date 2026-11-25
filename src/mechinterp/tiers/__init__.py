"""
Three tiers of mechanistic interpretability analysis.

Tier 1: Feature Directions - Does the model encode the concept?
Tier 2: Linear Probing - Where does information emerge?
Tier 3: Activation Patching - What causes the behavior?
"""

from mechinterp.tiers.tier1 import Tier1Analyzer
from mechinterp.tiers.tier2 import Tier2Analyzer
from mechinterp.tiers.tier3 import Tier3Analyzer

__all__ = [
    "Tier1Analyzer",
    "Tier2Analyzer",
    "Tier3Analyzer",
]
