# Quantum Boltzmann Machines: Applications in Quantitative Finance
The latest version of the thesis can be found [here](https://jugit.fz-juelich.de/qip/quantum-boltzmann-machines/-/jobs/artifacts/main/raw/latex/report/main.pdf?job=report).

The `qbm` Python package designed for training and analyzing QBMs has been moved to its own repository [here](https://jugit.fz-juelich.de/qip/qbm).

## Abstract
In this thesis we explore using the D-Wave Advantage 4.1 quantum annealer to sample from quantum Boltzmann distributions and train quantum Boltzmann machines (QBMs).
We focus on the real-world problem of using QBMs as generative models to produce synthetic foreign exchange market data and analyze how the results stack up against classical models based on restricted Boltzmann machines.
Additionally, we study a small 12-qubit problem which we use to compare samples obtained from the annealer to theory, and in the process gain vital insights into how well the Advantage 4.1 can sample quantum Boltzmann random variables and be used to train QBMs.
Through this we are able to show that the D-Wave Advantage 4.1 can sample classical Boltzmann random variables to some extent, but is limited in its ability to sample from quantum Boltzmann distributions.
Our findings indicate that models trained using the annealer are much noisier than simulations and struggle to perform at the same level as classical models.

## Table of Contents
* [Installation](#installation)

## Installation
This code in this thesis is best used with the predefined conda environment, which can be installed by running
```
conda env create -f environment.yml
```
or alternatively, running the `conda-create-env.sh` script (make sure to properly set the env vars in the script).

Extra dev dependencies can be installed with
```
conda env update --file environment-dev.yml
```

Additionally, a fork of scikit-learn is required, which can be installed by running
```
pip install --no-build-isolation git+https://github.com/cameronperot/scikit-learn.git@1.0.1-rbm#egg=scikit-learn
```
The thesis package can be installed by running
```
git clone git@jugit.fz-juelich.de:c.perot/quantum-boltzmann-machines.git
cd quantum-boltzmann-machines
pip install .
```
