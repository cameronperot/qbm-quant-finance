# Quantum Boltzmann Machines: Applications in Quantitative Finance
The QBM implemented here is based on Quantum Boltzman Machine by Amin et al. [[1]](#1).
The `qbm` Python package is designed to train models using both a simulation and a D-Wave quantum annealer.

The latest version of the thesis can be found [here](https://jugit.fz-juelich.de/qip/quantum-boltzmann-machines/-/jobs/artifacts/main/raw/latex/report/main.pdf?job=report).

The `qbm` Python package has been broken out into its own repository [here](https://jugit.fz-juelich.de/qip/qbm).

## Table of Contents
* [Installation](#installation)
    * [Conda Environment](#conda-environment)
* [Usage](#usage)
    * [Basic Configuration](#basic-configuration)
    * [BQRBM Model](#bqrbm-model)
        * [Instantiation](#instantiation)
        * [Training](#training)
        * [Sampling](#sampling)
        * [Saving and Loading](#saving-and-loading)
    * [Example](#example)
* [References](#references)

## Installation
The `qbm` package itself can be installed with
```
git clone git@jugit.fz-juelich.de:c.perot/quantum-boltzmann-machines.git
cd quantum-boltzmann-machines
pip install .
```

### Conda Environment
A predefined conda environment is already configured and ready for installation.
This can be installed by running
```
conda env create -f environment.yml
```
or alternatively, running the `conda-create-env.sh` script (make sure to properly set the env vars in the script).

Extra dev dependencies can be installed with
```
conda env update --file environment-dev.yml
```

## Usage
The Python package development has been moved [here](https://jugit.fz-juelich.de/qip/qbm).

# References
<a name="1">[1]</a> Mohammad H. Amin et al. “Quantum Boltzmann Machine”. In: Phys. Rev. X 8 (2 May 2018), p. 021050. doi: 10.1103/PhysRevX.8.021050. url: [https://link.aps.org/doi/10.1103/PhysRevX.8.021050](https://link.aps.org/doi/10.1103/PhysRevX.8.021050).
