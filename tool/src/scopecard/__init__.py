"""ScopeCard audit: compare a benchmark release with a declared target universe.

The release and the universe are enough: the construction rule does not have to be known.
The package reports what the release can never sample (W), how that relates to depth, which
simple rule (if any) reproduces the release, the derived scope d_min, and the break-even c*
for a stated comparison. It does not repair data and it does not compute c, which needs
labels on the excluded region.
"""
from .core import Universe, Release
from .mapping import map_units, map_triples, MappingReport
from .scope import unit_exclusion, pair_exclusion, depth_profile, derived_scope
from .rules import rule_units, search_rules
from .compare import c_star
from .hygiene import reverse_leakage, unseen_test_units
from .card import build_card, to_json, to_markdown

__version__ = '0.0.2'
__all__ = ['Universe', 'Release', 'map_units', 'map_triples', 'MappingReport', 'unit_exclusion', 'pair_exclusion',
           'depth_profile', 'derived_scope', 'rule_units', 'search_rules', 'c_star', 'reverse_leakage',
           'unseen_test_units', 'build_card', 'to_json', 'to_markdown']
