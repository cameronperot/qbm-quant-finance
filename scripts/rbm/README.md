# RBM Scripts
This directory contains scripts for training, sampling, and analyzing a scikit-learn BernoulliRBM.

## Config
The scripts revolve around the `config.json` file which contains configuration parameters for training, sampling, and analyzing.
Always set the desired parameters in `config.json` before running any of the scripts in this directory.
Below is a commented example.
```
{
    # this is the name of the model that will be loaded for generating the ensembles
    # this also controls the artifacts directory in which the samples and analysis are stored
    "load_model_name": "BernoulliRBM_20211102_105601",
    # the parameters for training the model
    "model_params": {
        # the seed value for the rng
        "seed": 42,
        # the number of bits to use when discretizing the data
        "n_bits": 16,
        # the size of the hidden layer
        "n_components": 30,
        # the learning rate
        "learning_rate": 1e-3,
        # the number of contrastive divergence iterations to train for
        "n_iter": 10000,
        # the epoch at which to begin decaying the learning rate
        "lr_decay_epoch": 5000,
        # the period over which to decay the learning rate by half
        "lr_decay_period": 1000,
        # a negative log-likelihood value to stop the training at (if it reaches)
        "early_stopping_criteria": -35,
        # the training batch size
        "batch_size": 10,
        # the points at which the features are split
        "split_points": [16, 32, 48]
    },
    # the parameters for generating a statistical ensemble
    "ensemble": {
        # the size of the ensemble
        "size": 100,
        # the number of steps between samples in the ensemble
        "n_steps": 1000,
        # the number of jobs to run in parallel
        "n_jobs": 6,
        # the seed value to generate the initial visible layer values
        "seed": 42
    },
    # these parameters are for generating a long Markov chain for autocorrelation analysis
    "autocorrelation": {
        # the number of samples per sub-chain
        "n_samples_per_df": 10000000,
        # the number of sub-chains
        "n_sample_dfs": 10,
        # the number of lags to use in computing/plotting the autocorrelation function
        "n_lags": 2000
    }
}
```

## Model Training
To train a model run the `train_model.py` script.

## Sample Generation
To generate a sample ensemble run the `generate_ensemble.py` script.

To generate a long Markov chain for analyzing autocorrelations run the `generate_autocorrelation_samples.py` script.

## Analysis
To analyze a generated ensemble run the `analyze_ensemble.py` script.

To analyze the autocorrelations run the `analyze_autocorrelation.py` script.
