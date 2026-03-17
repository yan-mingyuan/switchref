# SwitchRef

This repository contains the code, data, and experiment artifacts for the manuscript **"Switching-Reference Voltage Control for Distribution Systems with AI-Training Data Centers"**.

The codebase is prepared for public release alongside the paper. The current preprint is available on arXiv: [arXiv:2603.15588](https://arxiv.org/abs/2603.15588).

## Overview

Large-scale AI training workloads can induce rapid, periodic power fluctuations in modern data centers. When such data centers are connected to a distribution feeder, these fluctuations can create substantial voltage deviations. Conventional voltage regulation methods, including standard droop-based schemes, are primarily designed for more slowly varying demand and may therefore be inefficient or overly aggressive in this setting.

This repository studies that problem and implements a **switching-reference voltage control** framework that exploits the structured behavior of AI-training loads. The main goal is to reduce:

- voltage deviations along the feeder
- reactive power control effort required for voltage support

The repository includes both the main switching-reference control experiments and supporting data-center-aware feeder studies used during manuscript development.

## Main Components

The repository is organized around two complementary experiment tracks.

### 1. Switching-reference control experiments

The main manuscript experiments are implemented in:

```text
decentralized_switching/main.ipynb
```

This notebook contains the decentralized switching-reference control workflow. It loads the linearized feeder model, applies the local voltage-control logic, and generates the main simulation results.

### 2. Data-center-aware feeder experiments

Supporting experiments involving AI-training traces, feeder voltage response, and internal data-center control structure are implemented in:

```text
datacenter_optimization/historic-power-droop.ipynb
```

This notebook works with real or processed AI-training traces together with the IEEE 33-bus feeder model and the data-center abstractions defined in `datacenter_network_class.py`.

## Repository Structure

```text
switchref/
|-- data/
|   |-- traces_raw/                  # Raw AI-training power trace CSV files
|   `-- traces_regulated.pkl         # Processed trace artifact
|-- datacenter_optimization/
|   |-- datacenter_network_class.py  # Distribution network, storage, and data-center models
|   |-- generate_rack_load.py        # Trace loading and synthetic rack/PDU load generation
|   |-- Bus_Data_33bus.csv           # IEEE 33-bus bus data
|   |-- Branch_Data_33bus.csv        # IEEE 33-bus branch data
|   `-- historic-power-droop.ipynb   # Data-center-aware voltage and load-shaping experiments
|-- decentralized_switching/
|   |-- main.ipynb                   # Main switching-reference control experiments
|   |-- TestCase33.mat               # Linearized IEEE 33-bus feeder model
|   `-- linear_gain.pckl             # Precomputed controller gain
`-- figures/                         # Exported manuscript figures
```

## Environment

The repository is notebook-driven and was developed for Python 3.10.

Recommended packages:

- `numpy`
- `pandas`
- `matplotlib`
- `cvxpy`
- `networkx`
- `mat4py`
- `gymnasium`
- `jupyter`

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib cvxpy networkx mat4py gymnasium jupyter
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib cvxpy networkx mat4py gymnasium jupyter
```

## Data and Included Models

This repository includes:

- raw AI-training power traces in `data/traces_raw/`
- a processed trace artifact in `data/traces_regulated.pkl`
- IEEE 33-bus feeder data in CSV and MAT formats
- a precomputed linear gain used by the switching experiments

The feeder studies rely on the included model files:

- `datacenter_optimization/Bus_Data_33bus.csv`
- `datacenter_optimization/Branch_Data_33bus.csv`
- `decentralized_switching/TestCase33.mat`

## Usage

A typical workflow for this repository is:

1. Create a Python environment and install the required packages.
2. Open the notebooks from the repository root or from their respective subdirectories.
3. Update any local file paths if your environment differs from the original development setup.
4. Run the notebooks top to bottom to reproduce the experiments and figures.

For most users, the recommended entry point is:

```text
decentralized_switching/main.ipynb
```

This notebook reproduces the main switching-reference voltage control experiments reported in the manuscript.

If you are interested in the data-center-side modeling and feeder response to AI-training traces, use:

```text
datacenter_optimization/historic-power-droop.ipynb
```

## Running the Code

### A. Reproduce the main switching-reference experiments

Open and run:

```text
decentralized_switching/main.ipynb
```

This notebook:

- loads the linearized feeder model
- initializes the voltage dynamics
- loads the precomputed controller gain
- simulates decentralized switching-reference voltage control
- generates the main plots for unregulated, weak-grid, and regulated cases

### B. Run the data-center-aware feeder experiments

Open and run:

```text
datacenter_optimization/historic-power-droop.ipynb
```

This notebook:

- loads feeder data from the included IEEE 33-bus CSV files
- loads AI-training traces from `data/traces_raw/`
- constructs data-center and storage objects
- evaluates voltage and power behavior under data-center-aware settings

Before running, update the notebook's trace path if needed. The current notebook contains a development-machine path, so it should be replaced with a repository-local path such as:

```python
data_path = "../data/traces_raw"
```

## Figures

The `figures/` directory contains exported figures generated during manuscript preparation. These files are useful as reference outputs when checking whether a reproduced run is qualitatively consistent with the paper.

Small differences in formatting or plotting appearance may occur across environments, package versions, or notebook execution order.

## Reproducibility Notes

- The two notebooks above are the main experiment entry points.
- Some notebook cells still reflect the original research environment and may require small local path edits.
- Random seeds are fixed in parts of the code, but exact numerical traces or figure formatting may still vary slightly across systems.
- The repository is organized as a research artifact rather than as a packaged software library.

## Citation

If you use this repository, please cite the arXiv preprint:

```bibtex
@misc{yan2026switchingreferencevoltagecontroldistribution,
  title         = {Switching-Reference Voltage Control for Distribution Systems with AI-Training Data Centers},
  author        = {Mingyuan Yan and Trager Joswig-Jones and Baosen Zhang and Yize Chen and Wenqi Cui},
  year          = {2026},
  eprint        = {2603.15588},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SY},
  url           = {https://arxiv.org/abs/2603.15588}
}
```

Preprint:

- [arXiv:2603.15588](https://arxiv.org/abs/2603.15588)

## License

This project is released under the MIT License. See `LICENSE` for details.

## Contact

For questions about the code, experiments, or reproduction details, please contact the paper authors.

