import json
import numpy as np
import pandas as pd
import pickle
import pytest

from pathlib import Path
from unittest.mock import patch, mock_open

from qbm.utils import (
    compute_df_stats,
    compute_stats_over_dfs,
    get_rng,
    get_project_dir,
    load_artifact,
    lr_exp_decay,
    save_artifact,
)


def test_compute_df_stats(monkeypatch):
    monkeypatch.setattr("qbm.utils.misc.pd.DataFrame.min", lambda self: 1)
    monkeypatch.setattr("qbm.utils.misc.pd.DataFrame.max", lambda self: 2)
    monkeypatch.setattr("qbm.utils.misc.pd.DataFrame.mean", lambda self: 3)
    monkeypatch.setattr("qbm.utils.misc.pd.DataFrame.median", lambda self: 4)
    monkeypatch.setattr("qbm.utils.misc.pd.DataFrame.std", lambda self: 5)

    df = pd.DataFrame(np.ones((2, 3)))
    expected_df = pd.DataFrame.from_dict(
        {"min": 1, "max": 2, "mean": 3, "median": 4, "std": 5}, orient="index"
    )

    df_stats = compute_df_stats(df)

    assert df_stats.equals(expected_df)


def test_compute_stats_over_dfs(monkeypatch):
    df = pd.DataFrame.from_dict(np.arange(6).reshape((3, 2))).astype(np.float64)

    stats_over_dfs = compute_stats_over_dfs([df, df])

    assert stats_over_dfs["means"].equals(df)
    assert stats_over_dfs["medians"].equals(df)
    assert stats_over_dfs["stds"].equals(pd.DataFrame(np.zeros((3, 2))))


def test_get_project_dir_env_not_set(monkeypatch):
    monkeypatch.setattr("qbm.utils.misc.os.getenv", lambda x: None)

    with pytest.raises(Exception):
        get_project_dir()


def test_get_project_dir_path_does_not_exist(monkeypatch):
    monkeypatch.setattr("qbm.utils.misc.os.getenv", lambda x: "/test/path")
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: False)

    with pytest.raises(Exception):
        get_project_dir()


def test_get_project_dir_success(monkeypatch):
    monkeypatch.setattr("qbm.utils.misc.os.getenv", lambda x: "/test/path")
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    project_dir = get_project_dir()
    assert project_dir == Path("/test/path")


@patch("qbm.utils.misc.SeedSequence")
@patch("qbm.utils.misc.MT19937")
@patch("qbm.utils.misc.RandomState")
def test_get_rng(mock_RandomState, mock_MT19937, mock_SeedSequence):
    mock_MT19937.return_value = "test_MT19937"
    mock_SeedSequence.return_value = "test_SeedSequence"
    mock_RandomState.return_value = "test_RandomState"

    seed = 42
    rng = get_rng(seed)

    mock_SeedSequence.assert_called_with(seed)
    mock_MT19937.assert_called_with("test_SeedSequence")
    mock_RandomState.assert_called_with("test_MT19937")
    assert rng == "test_RandomState"


def test_load_artifact_invalid_file_path(monkeypatch):
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: False)

    file_path = Path("/test/path/file")

    with pytest.raises(Exception):
        load_artifact(file_path)


def test_load_artifact_invalid_file_extension(monkeypatch):
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file")

    with pytest.raises(Exception):
        load_artifact(file_path)


def test_load_artifact_json_success(monkeypatch):
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file.json")
    artifact = {"a": 1, "b": 2}
    json_ = json.dumps(artifact)

    with patch("builtins.open", mock_open(read_data=json_)) as mock_file:
        loaded_artifact = load_artifact(file_path)

        assert loaded_artifact == artifact
        mock_file.assert_called_with(file_path, "r")


def test_load_artifact_pickle_success(monkeypatch):
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file.pkl")
    artifact = {"a": 1, "b": 2}
    pickle_ = pickle.dumps(artifact)

    with patch("builtins.open", mock_open(read_data=pickle_)) as mock_file:
        loaded_artifact = load_artifact(file_path)

        assert loaded_artifact == artifact
        mock_file.assert_called_with(file_path, "rb")


@pytest.mark.parametrize("epoch, decay_epoch, period", [(0, 5, 10), (5, 5, 10), (6, 5, 10)])
def test_lr_exp_decay(epoch, decay_epoch, period):
    lr_factor = lr_exp_decay(epoch, decay_epoch, period)

    if epoch <= decay_epoch:
        assert lr_factor == 1
    else:
        assert lr_factor == 2 ** ((decay_epoch - epoch) / period)


@patch("qbm.utils.misc.Path.mkdir")
def test_save_artifact_parent_does_not_exist(mock_mkdir, monkeypatch):
    mock_mkdir.return_value = None
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: False)

    file_path = Path("/test/path/file.json")
    artifact = {"a": 1, "b": 2}
    json_ = json.dumps(artifact)

    with patch("builtins.open", mock_open(read_data=json_)):
        save_artifact(artifact, file_path)
        mock_mkdir.assert_called_with(parents=True)


def test_save_artifact_invalid_suffix(monkeypatch):
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file.invalid")
    artifact = {"a": 1, "b": 2}

    with pytest.raises(Exception):
        save_artifact(artifact, file_path)


@patch("qbm.utils.misc.json.dump")
def test_save_artifact_json_success(mock_dump, monkeypatch):
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file.json")
    artifact = {"a": 1, "b": 2}

    with patch("builtins.open") as mock_file:
        save_artifact(artifact, file_path)

        mock_file.assert_called_with(file_path, "w")
        mock_dump.assert_called()


@patch("qbm.utils.misc.pickle.dump")
def test_save_artifact_pickle_success(mock_dump, monkeypatch):
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file.pkl")
    artifact = {"a": 1, "b": 2}

    with patch("builtins.open") as mock_file:
        save_artifact(artifact, file_path)

        mock_file.assert_called_with(file_path, "wb")
        mock_dump.assert_called()
