# V7: peak memory of the two largest audits (2026-11-17)

Copied from a run of `reproduce.py --tier memory` from the reproduction package v0.1.0, in a new virtual
environment (Python 3.14.3 with numpy 2.5.5, on a Windows 11 laptop with 22 logical CPUs and 15 GB of memory). The step runs the YAGO3-10 and Amazon-Book audits one after the other, each alone in a
child process. When the audit ends, that process reads its own peak memory: on Windows its peak working set, which
is the peak resident memory, and its peak private memory (GetProcessMemoryInfo); on Linux and macOS its maximum
resident set size (getrusage). The cards the two audits wrote equal the stored ones. `memory.tsv` is the file the
step wrote. Other jobs were running on the machine during the run (about 2.9 GB of memory was available when it started), so the times are upper bounds; in version 0.0.1 these two audits took 182 and 313 s (`../v5_clean_env/`).

| Card | Peak working set | Peak private memory | Time |
|---|---|---|---|
| YAGO3-10 | 2.86 GB (2,858,549,248 bytes) | 3.56 GB | 226 s |
| Amazon-Book | 3.28 GB (3,278,667,776 bytes) | 4.03 GB | 764 s |

The peak working set is the figure the paper gives. The private memory counts all memory the process committed,
used or not; unlike the working set, it is not lowered when Windows trims working sets for other processes.

A first run of the step on the same day read the peak of the process that `reproduce.py` starts. In a virtual
environment on Windows that process is a launcher, which starts the interpreter as a further process, so the
figure it gave (0.01 GB) was the launcher's. `reproduce.py` was corrected so that the process running the audit
reads its own peak; the run above used the corrected step.

Before this record, the paper's figure for both audits, about 2.5 GB, came from the tool's design notes, where it
was measured when each audit was added; those notes are not part of the package, and the clean-environment run of
2026-10-28 (`../v5_clean_env/`) did not log memory.
