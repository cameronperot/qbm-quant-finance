# Quantum Boltzmann Machines: Applications in Quantitative Finance
The latest version of the thesis can be found [here on arXiv](https://arxiv.org/abs/2301.13295).

The `qbm` Python package designed for training and analyzing QBMs has been moved to its own repository [here](https://github.com/cameronperot/qbm).

## Abstract
In this thesis we explore using the D-Wave Advantage 4.1 quantum annealer to sample from quantum Boltzmann distributions and train quantum Boltzmann machines (QBMs).
We focus on the real-world problem of using QBMs as generative models to produce synthetic foreign exchange market data and analyze how the results stack up against classical models based on restricted Boltzmann machines (RBMs).
Additionally, we study a small 12-qubit problem which we use to compare samples obtained from the Advantage 4.1 with theory, and in the process gain vital insights into how well the Advantage 4.1 can sample quantum Boltzmann random variables and be used to train QBMs.
Through this, we are able to show that the Advantage 4.1 can sample classical Boltzmann random variables to some extent, but is limited in its ability to sample from quantum Boltzmann distributions.
Our findings indicate that QBMs trained using the Advantage 4.1 are much noisier than those trained using simulations and struggle to perform at the same level as classical RBMs.
However, there is the potential for QBMs to outperform classical RBMs if future generation annealers can generate samples closer to the desired theoretical distributions.

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

Additionally, a fork of scikit-learn is required, and can be installed by running
```
pip install --no-build-isolation git+https://github.com/cameronperot/scikit-learn.git@1.0.1-rbm#egg=scikit-learn
```
The thesis package can be installed by running
```
git clone git@github.com:cameronperot/qbm-quant-finance.git
cd qbm-quant-finance
pip install .
```
