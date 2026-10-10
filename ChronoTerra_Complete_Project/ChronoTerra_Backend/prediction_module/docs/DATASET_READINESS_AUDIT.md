# Dataset readiness audit — this working session

## Verified in the available workspace

- PyTorch 2.10.0 CPU build was importable in Python 3.13.5; CUDA was not available.
- The OSCD training package and source code are present.
- The separate smoke-training directory contains a tiny generated fixture with scenes named `scene_a`, `scene_b`, and `scene_c`; the files are only around 1.1 MB in total and are not the real OSCD archive.
- Real OSCD / Sen1Floods11 / xBD datasets were not found in the workspace.
- An external `curl` to the official OSCD website failed in this runtime because DNS could not resolve the host. No real dataset download was completed.
- New software-path tests pass on the tiny synthetic fixtures. They do not establish real satellite-model performance.

## Prior synthetic smoke result: do not cite as model accuracy

A prior one-epoch run against the synthetic `scene_a` fixture showed about 0.048 F1 and 0.0248 IoU at its chosen threshold, with a false-positive rate of 0.94375. A JSON output incorrectly labelled that generated fixture as the “OSCD official test split”; the source files demonstrate it was not genuine OSCD data. That output is excluded from this package and must not be reported in a presentation or benchmark.

## Actual trained status

No scientifically valid real-dataset checkpoint or accuracy result is available yet. This suite contains training code and dataset analysis tools; actual training requires the real labelled data from the maintainers and a dataset-specific manifest.
