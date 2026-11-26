"""Loaders for common benchmark release formats.

Release files have a fixed format, and a line that does not fit stops the audit with an InputError naming the
file and the line; so does an empty release file or unit list, named by file. Raw universe files (check-ins, ratings, qrels, triple dumps) are read leniently: a line that
does not fit is skipped, but counted in an InputLog, which the card reports, so nothing is dropped silently.
"""
import gzip
import hashlib
import os
import pickle
import sys
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np


class InputError(ValueError):
    """A file that does not have the format the audit needs. The message names the file and, where it can,
    the line, so it can be shown to a user as is."""


def basename(path: str) -> str:
    return path.replace('\\', '/').rsplit('/', 1)[-1]


def progress(stage: str) -> None:
    """One line per audit stage on stderr ('[stage] ...'); the browser page shows it while an audit runs."""
    print(f'[stage] {stage}', file=sys.stderr, flush=True)


class InputLog:
    """Lines read and skipped per input file, for the card's `inputs` field and its warnings."""

    def __init__(self):
        self.files: Dict[str, Dict] = {}

    def record(self, path: str, lines: int, skipped: int, expected: str, first: Optional[Tuple[int, str]] = None):
        name = basename(path)
        if lines == 0:
            raise InputError(f'{name} has no data lines.')
        if skipped and skipped == lines:
            where = f' Line {first[0]}: {first[1][:80]!r}.' if first else ''
            raise InputError(f'{name}: none of its {lines:,} lines has the expected format ({expected}).{where}')
        entry = {'lines': lines, 'skipped': skipped, 'expected': expected}
        if skipped and first:
            entry['first_skipped'] = {'line': first[0], 'text': first[1][:120]}
        self.files[name] = entry

    def warnings(self) -> List[str]:
        return [f"{e['skipped']:,} of the {e['lines']:,} lines of {name} did not have the expected format "
                f"({e['expected']}) and {'was' if e['skipped'] == 1 else 'were'} skipped "
                f"(first at line {e['first_skipped']['line']})."
                for name, e in sorted(self.files.items()) if e['skipped']]

    def to_dict(self) -> Dict:
        return dict(sorted(self.files.items()))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_triples(path: str) -> List[Tuple[str, str, str]]:
    """Tab-separated (head, relation, tail), one per line (ConvE / LibKGE raw / PyG raw format)."""
    out = []
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            tok = line.rstrip('\r\n').split('\t')
            if len(tok) != 3 or not tok[0] or not tok[2]:
                raise InputError(f'{basename(path)}, line {n}: expected 3 tab-separated fields (head, relation, tail), '
                                 f'found {len(tok)}: {line.strip()[:80]!r}')
            out.append(tuple(tok))
    if not out:                                         # an empty split would be audited as a smaller release
        raise InputError(f'{basename(path)} has no data lines.')
    return out


def read_kg_release(directory: str, splits=('train', 'valid', 'test')) -> Tuple[Dict[str, list], Dict[str, str]]:
    trips, hashes = {}, {}
    for s in splits:
        p = os.path.join(directory, f'{s}.txt')
        trips[s] = read_triples(p)
        hashes[f'{s}.txt'] = sha256(p)
    return trips, hashes


def read_lightgcn_release(directory: str) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, str]]:
    """LightGCN / NGCF release: train.txt and test.txt hold 'user item item ...' with remapped ids;
    user_list.txt and item_list.txt map them back to original ids. Returns original-id pairs per split."""
    def mapping(fn):
        m = {}
        with open(os.path.join(directory, fn), encoding='utf-8') as f:
            next(f, None)                                   # header: 'org_id remap_id'
            for n, line in enumerate(f, 2):
                tok = line.split()
                if not tok:
                    continue
                if len(tok) != 2 or not tok[1].isdigit():
                    raise InputError(f'{fn}, line {n}: expected 2 fields (original id, remapped integer id), '
                                     f'found {line.strip()[:80]!r}')
                m[int(tok[1])] = tok[0]
        if not m:
            raise InputError(f'{fn} has no data lines.')
        return m
    um, im = mapping('user_list.txt'), mapping('item_list.txt')
    out, hashes = {}, {}
    for s in ('train', 'test'):
        p = os.path.join(directory, f'{s}.txt')
        hashes[f'{s}.txt'] = sha256(p)
        rows, lines = [], 0
        with open(p, encoding='utf-8') as f:
            for n, line in enumerate(f, 1):
                tok = line.split()
                lines += bool(tok)
                if not all(t.isdigit() for t in tok):
                    raise InputError(f'{s}.txt, line {n}: expected space-separated integer ids (user, then items), '
                                     f'found {line.strip()[:80]!r}')
                if len(tok) > 1:                            # a user with no items in this split has one field
                    if int(tok[0]) not in um:
                        raise InputError(f'{s}.txt, line {n}: user id {tok[0]} is not in user_list.txt')
                    bad = next((t for t in tok[1:] if int(t) not in im), None)
                    if bad is not None:
                        raise InputError(f'{s}.txt, line {n}: item id {bad} is not in item_list.txt')
                    u = um[int(tok[0])]
                    rows += [(u, im[int(i)]) for i in tok[1:]]
        if not rows:                                    # an empty split would be audited as a smaller release
            raise InputError(f'{s}.txt has no data lines.' if not lines else
                             f'{s}.txt lists no user-item pairs: each of its {lines:,} lines has a user id only.')
        out[s] = rows
    for fn in ('user_list.txt', 'item_list.txt'):
        hashes[fn] = sha256(os.path.join(directory, fn))
    return out, hashes


def open_maybe_gz(path: str):
    return gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else open(path, encoding='utf-8')


def read_trec_qrels(path: str, log: Optional[InputLog] = None) -> Dict[str, Dict[str, int]]:
    """TREC qrels: 'topic iteration doc grade' per line -> topic -> {doc: grade}. Lines with another shape or a
    non-integer grade are skipped and counted in `log`."""
    out: Dict[str, Dict[str, int]] = {}
    lines = skipped = 0
    first = None
    with open_maybe_gz(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            lines += 1
            tok = line.split()
            if len(tok) == 4 and tok[3].lstrip('-').isdigit():
                out.setdefault(tok[0], {})[tok[2]] = int(tok[3])
            else:
                skipped += 1
                first = first or (n, line.strip())
    (log or InputLog()).record(path, lines, skipped, '4 whitespace-separated fields: topic, iteration, document, '
                                                     'integer grade', first)
    return out


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickles numpy arrays and nothing else: any other global is refused, so no code from a
    downloaded file can run (the allow-list of torch.load(weights_only=True) plus numpy's)."""
    ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
               ('numpy', 'ndarray'), ('numpy', 'dtype'), ('_codecs', 'encode'), ('collections', 'OrderedDict')}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED or (module == 'numpy.dtypes' and name.endswith('DType')):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'refusing to load {module}.{name}')

    def persistent_load(self, pid):
        raise pickle.UnpicklingError('the file holds torch tensors, not numpy arrays; convert it with torch first')


def read_torch_numpy_dict(path: str) -> Dict[str, np.ndarray]:
    """A torch.save'd dict of numpy arrays (OGB split files, e.g. split/time/valid.pt) without torch.
    A .npz with the same keys is read as well."""
    if path.endswith('.npz'):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith('/data.pkl') or n == 'data.pkl')
            with z.open(name) as f:
                obj = _NumpyOnlyUnpickler(f).load()
    else:
        # legacy torch format: magic number, protocol version, system info, the object, then storage keys
        with open(path, 'rb') as f:
            if _NumpyOnlyUnpickler(f).load() != 0x1950a86a20f9469cfc6c:
                raise ValueError(f'{path}: not a torch file')
            _NumpyOnlyUnpickler(f).load()
            _NumpyOnlyUnpickler(f).load()
            obj = _NumpyOnlyUnpickler(f).load()
            if _NumpyOnlyUnpickler(f).load():
                raise pickle.UnpicklingError(f'{path}: holds torch storages; convert it with torch first')
    return {k: np.asarray(v) for k, v in obj.items()}


def read_int_column(path: str) -> np.ndarray:
    """One integer per line (.gz allowed), e.g. OGB raw/node_year.csv.gz."""
    with open_maybe_gz(path) as f:
        tok = f.read().split()
    try:
        return np.array(tok, dtype=np.int64)
    except ValueError:
        bad = next(t for t in tok if not t.lstrip('-').isdigit())
        raise InputError(f'{basename(path)}: expected one integer per line, found {bad[:40]!r}') from None


def read_pair_rows(path: str, a_col, b_col, label_col=None, header: bool = True, sep: str = None,
                   log: Optional[InputLog] = None, columns: Optional[list] = None) -> List[Tuple]:
    """Rows of a pair table (CSV/TSV, .gz allowed) as (a, b, label or None). Columns are names (with a header)
    or 0-based indexes. The separator defaults to ',' for .csv(.gz) files and a tab otherwise. Rows too short
    for the named columns are skipped and counted in `log`; `columns` receives the header, if any."""
    import csv
    if sep is None:
        sep = ',' if path.lower().endswith(('.csv', '.csv.gz')) else '\t'
    name = basename(path)
    with open_maybe_gz(path) as f:
        rows = csv.reader(f, delimiter=sep)
        head = next(rows, None) if header else None
        if columns is not None and head:
            columns.extend(head)

        def index(c):
            if isinstance(c, int) or (isinstance(c, str) and c.isdigit()):
                return int(c)
            if head is None:
                raise InputError(f'{name}: column {c!r} is given by name, but the file is read without a header')
            if c not in head:
                raise InputError(f'{name}: no column {c!r} (columns: {", ".join(head)}; separator {sep!r})')
            return head.index(c)
        ia, ib = index(a_col), index(b_col)
        il = index(label_col) if label_col is not None else None
        need = max(i for i in (ia, ib, il) if i is not None)
        out, lines, skipped, first = [], 0, 0, None
        for n, r in enumerate(rows, 2 if header else 1):
            if not r or not any(x.strip() for x in r):
                continue
            lines += 1
            if len(r) > need:
                out.append((r[ia].strip(), r[ib].strip(), r[il].strip() if il is not None else None))
            else:
                skipped += 1
                first = first or (n, sep.join(r))
    (log or InputLog()).record(path, lines, skipped, f'at least {need + 1} fields separated by {sep!r}', first)
    return out


def read_unit_list(path: str, column=None, header: bool = False, sep: str = None) -> List[str]:
    """Universe units: one per line, or one column of a table (name or 0-based index)."""
    if column is None:
        with open_maybe_gz(path) as f:
            lines = [x.strip() for x in f if x.strip()]
        lines = lines[1:] if header else lines
        if not lines:
            raise InputError(f'{basename(path)} has no data lines.')
        return lines
    return [a for a, _, _ in read_pair_rows(path, column, column, None, header, sep)]


def read_topic_ids(path: str) -> List[str]:
    """Topic list, one topic per line with the id in the first tab-separated field (MS MARCO queries .tsv(.gz))."""
    with open_maybe_gz(path) as f:
        topics = [line.split('\t')[0].strip() for line in f if line.strip()]
    if not topics:
        raise InputError(f'{basename(path)} has no data lines.')
    return topics
