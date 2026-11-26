import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Local copies of the public source files used by the golden tests (experiments/pilot_data);
# override with SCOPECARD_PILOT_DATA. Golden tests are skipped when the files are absent.
PILOT = os.environ.get('SCOPECARD_PILOT_DATA',
                       os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'pilot_data'))
