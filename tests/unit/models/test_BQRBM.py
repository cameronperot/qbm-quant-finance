import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from qbm.models import BQRBM
from qbm.utils import binarize_df, get_rng, get_binarization_params, prepare_training_data

n_visible = 8
n_hidden = 4
n_samples = 100
n_qubits = n_visible + n_hidden
learning_rate = 2e-3


def mock_initialize_sampler(model):
    setattr(model, "qpu", None)
    setattr(model, "h_range", np.array([-4, 4]))
    setattr(model, "J_range", np.array([-1, 1]))


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr("qbm.models.BQRBM._initialize_sampler", mock_initialize_sampler)

    anneal_schedule_data = pd.DataFrame.from_dict(
        {0.5: {"A(s) (GHz)": 0.1, "B(s) (GHz)": 1.1}}, orient="index"
    )

    rng = get_rng(0)
    df = pd.DataFrame.from_dict({"x": rng.normal(0, 1, 1000)})
    binarization_params = get_binarization_params(df, n_bits=n_visible)
    df_binarized = binarize_df(df, binarization_params)
    X_train = prepare_training_data(df_binarized)["X_train"]

    model = BQRBM(
        X=X_train,
        n_hidden=n_hidden,
        embedding={},
        anneal_params={"s": 0.5},
        anneal_schedule_data=anneal_schedule_data,
        β_initial=1.5,
        qpu_params={"region": "eu-central-1", "solver": "Advantage_system5.1"},
        seed=0,
        exact=True,
    )

    return model


def test_init(monkeypatch):
    monkeypatch.setattr("qbm.models.BQRBM._initialize_sampler", mock_initialize_sampler)

    anneal_schedule_data = pd.DataFrame.from_dict(
        {0.5: {"A(s) (GHz)": 0.1, "B(s) (GHz)": 1.1}}, orient="index"
    )

    rng = get_rng(0)
    df = pd.DataFrame.from_dict({"x": rng.normal(0, 1, 1000)})
    binarization_params = get_binarization_params(df, n_bits=n_visible)
    df_binarized = binarize_df(df, binarization_params)
    X_train = prepare_training_data(df_binarized)["X_train"]

    embedding = {}
    anneal_params = {"s": 0.5}
    β_initial = 1.5
    β_range = (0.1, 10)
    qpu_params = {"region": "eu-central-1", "solver": "Advantage_system5.1"}
    seed = 0
    exact = True

    model = BQRBM(
        X=X_train,
        n_hidden=n_hidden,
        embedding=embedding,
        anneal_params=anneal_params,
        anneal_schedule_data=anneal_schedule_data,
        β_initial=β_initial,
        β_range=β_range,
        qpu_params=qpu_params,
        seed=seed,
        exact=exact,
    )

    assert (model.X == 1 - 2 * X_train).all()
    assert (model.X_binary == X_train).all()
    assert model.n_hidden == n_hidden
    assert model.n_visible == n_visible
    assert model.n_qubits == n_qubits
    assert model.embedding == embedding
    assert model.anneal_params == anneal_params
    assert model.qpu_params == qpu_params
    assert model.seed == seed
    assert model.exact == exact
    assert model.A == anneal_schedule_data.loc[anneal_params["s"], "A(s) (GHz)"]
    assert model.B == anneal_schedule_data.loc[anneal_params["s"], "B(s) (GHz)"]
    assert model.β == β_initial
    assert model.βs == [β_initial]
    assert model.β_range == β_range


def test__compute_mean_energy(model):
    rng = get_rng(0)
    V = rng.rand(n_samples, n_visible)
    H = rng.rand(n_samples, n_hidden)
    W = rng.rand(n_visible, n_hidden)

    E = 0
    for k in range(n_samples):
        E += -V[k] @ model.a - H[k] @ model.b - V[k] @ W @ H[k]
    E /= n_samples

    E_model = model._compute_mean_energy(V, H, V @ W)

    assert np.isclose(E, E_model)


def test__compute_positive_grads(model):
    rng = get_rng(0)
    V_pos = rng.rand(n_samples, n_visible)
    Γ = model.β * model.A
    b_eff = model.b + V_pos @ model.W
    D = np.sqrt(Γ ** 2 + b_eff ** 2)
    H_pos = (b_eff / D) * np.tanh(D)

    grads = {
        "a_pos": np.zeros(model.a.shape),
        "b_pos": np.zeros(model.b.shape),
        "W_pos": np.zeros(model.W.shape),
    }
    for k in range(n_samples):
        grads["a_pos"] += V_pos[k] / n_samples
        grads["b_pos"] += H_pos[k] / n_samples
        grads["W_pos"] += np.outer(V_pos[k], H_pos[k]) / n_samples

    model._compute_positive_grads(V_pos)

    assert b_eff.shape == (n_samples, n_hidden)
    assert D.shape == (n_samples, n_hidden)
    assert H_pos.shape == (n_samples, n_hidden)
    for grad_name, grad in grads.items():
        assert np.isclose(grad, model.grads[grad_name]).all()


def test__compute_negative_grads(monkeypatch, model):
    rng = get_rng(0)
    state_vectors = rng.rand(n_samples, n_qubits)
    V_neg = state_vectors[:, :n_visible]
    H_neg = state_vectors[:, n_visible:]
    monkeypatch.setattr(
        "qbm.models.BQRBM.sample", lambda self, n_samples: {"state_vectors": state_vectors}
    )

    grads = {
        "a_neg": np.zeros(model.a.shape),
        "b_neg": np.zeros(model.b.shape),
        "W_neg": np.zeros(model.W.shape),
    }
    for k in range(n_samples):
        grads["a_neg"] += V_neg[k] / n_samples
        grads["b_neg"] += H_neg[k] / n_samples
        grads["W_neg"] += np.outer(V_neg[k], H_neg[k]) / n_samples

    model._compute_negative_grads(n_samples)

    for grad_name, grad in grads.items():
        assert np.isclose(grad, model.grads[grad_name]).all()


def test__update_β(monkeypatch, model):
    rng = get_rng(0)
    state_vectors = rng.rand(n_samples, n_qubits)
    monkeypatch.setattr("qbm.models.BQRBM.sample", lambda self, n_samples: state_vectors)

    β = model.β
    setattr(model, "learning_rate", learning_rate)

    V_train = model.X
    VW_train = V_train @ model.W
    b_eff = model.b + VW_train
    D = np.sqrt(model.Γ ** 2 + b_eff ** 2)
    H_train = (b_eff / D) * np.tanh(D)
    E_train = model._compute_mean_energy(V_train, H_train, VW_train)

    V_model = state_vectors[:, :n_visible]
    H_model = state_vectors[:, n_visible:]
    E_model = model._compute_mean_energy(V_model, H_model, V_model @ model.W)

    Δβ = learning_rate * (E_train - E_model)

    model._update_β({"state_vectors": state_vectors}, learning_rate)

    assert model.β == np.clip(β + Δβ, *model.β_range)


def test__clip_weights_and_biases(monkeypatch, model):
    rng = get_rng(0)
    μ = 0
    scaling_factor = model.β * model.B
    setattr(model, "a", rng.normal(μ, model.h_range.max() * scaling_factor, n_visible))
    setattr(model, "b", rng.normal(μ, model.h_range.max() * scaling_factor, n_hidden))
    setattr(
        model,
        "W",
        rng.normal(μ, model.h_range.max() * scaling_factor, (n_visible, n_hidden)),
    )

    model._clip_weights_and_biases()
    h = np.concatenate((model.a, model.b)) / scaling_factor
    J = np.zeros((n_qubits, n_qubits))
    J[:n_visible, n_visible:] = model.W / scaling_factor

    assert (np.abs(np.array([h.min(), h.max()])) < np.abs(model.h_range)).all()
    assert (np.abs(np.array([J.min(), J.max()])) < np.abs(model.J_range)).all()
    assert np.isclose(np.array([h.min(), h.max()]), model.h_range).all()
    assert np.isclose(np.array([h.min(), h.max()]), model.h_range).all()
