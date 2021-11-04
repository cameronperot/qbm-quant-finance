# Quantum Boltzmann Machines

## Installation
This is best used with the conda environment defined in `environment.yml`.
To install the conda enviroment run
```
conda env create -f environment.yml
```
or alternatively, run the `conda-create-env.sh` script (make sure to properly set the env vars in the script).

Additionally, a [fork of scikit-learn](https://github.com/cameronperot/scikit-learn) is required.
To install it run
```
pip install --no-build-isolation git+https://github.com/cameronperot/scikit-learn.git@1.0.1-rbm#egg=scikit-learn
```

Finally, to install the qbm package run
```
git clone git@jugit.fz-juelich.de:c.perot/quantum-boltzmann-machines.git
cd quantum-boltzmann-machines
pip install .
```

To install extra dev dependencies run
```
conda env update --file environment-dev.yml
```
