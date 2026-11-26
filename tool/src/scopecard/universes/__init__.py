"""Built-in universes. Each loader reads a public source file the user downloads (or that the
package ships where the licence allows) and returns a Universe."""
from .wordnet import (load_wordnet30, load_wordnet30_source, load_wordnet30_table, export_wordnet30_table,
                      WN18_RELATIONS, WN18RR_RELATIONS)
from .interactions import from_pair_stream, from_pairs, load_ratings_csv, load_snap_checkins
from .triples import from_triples, load_triples_tsv, load_yago3_facts
from .pairs import from_unit_pairs

__all__ = ['load_wordnet30', 'load_wordnet30_source', 'load_wordnet30_table', 'export_wordnet30_table', 'WN18_RELATIONS',
           'WN18RR_RELATIONS', 'from_pairs', 'from_pair_stream', 'load_ratings_csv', 'load_snap_checkins', 'from_triples',
           'load_triples_tsv', 'load_yago3_facts', 'from_unit_pairs']
