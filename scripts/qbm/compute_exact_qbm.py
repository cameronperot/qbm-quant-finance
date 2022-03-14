import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.constants import h as h_P, k as k_B


from qbm.utils import get_project_dir, load_artifact, save_artifact
from qbm.utils.exact_qbm import get_pauli_kron, compute_H, compute_rho


if __name__ == "__main__":
    # compute exact data for all specified configs
    config_ids = (5,)
    project_dir = get_project_dir()

    # set s and T values
    T_values = np.concatenate(([1e-6], np.round(np.arange(2e-3, 202e-3, 2e-3), 3)))  # [K]
    s_values = np.round(np.arange(0, 1.01, 0.01), 2)

    # configure tqdm bars
    config_bar = tqdm(range(len(config_ids)), desc="configs")
    s_bar = tqdm(range(len(s_values)), desc="s values")
    T_bar = tqdm(range(len(T_values)), desc="T values")

    # Boltzmann's constant in GHz/K
    k_B_GHz_K = k_B / (h_P * 1e9)

    for config_id in config_ids:
        # load the config
        config_dir = project_dir / f"artifacts/exact_analysis/{config_id:02}"
        config = load_artifact(config_dir / "config.json")
        n_qubits = config["n_qubits"]

        # load the anneal schedule data
        if config["qpu_params"]["solver"] == "Advantage_system5.1":
            anneal_schedule_data = pd.read_csv(
                project_dir
                / "data/anneal_schedules/csv/09-1265A-A_Advantage_system5_1_annealing_schedule.csv",
                index_col="s",
            )
        elif config["qpu_params"]["solver"] == "Advantage_system4.1":
            anneal_schedule_data = pd.read_csv(
                project_dir
                / "data/anneal_schedules/csv/09-1263A-A_Advantage_system4_1_annealing_schedule.csv",
                index_col="s",
            )
        # for some reason 0.5 is missing for Advantage_system5.1 so we need to interpolate
        if 0.5 not in anneal_schedule_data.index:
            anneal_schedule_data.loc[0.5] = (
                anneal_schedule_data.loc[0.499] + anneal_schedule_data.loc[0.501]
            ) / 2

        # create Kronecker σ matrices
        pauli_kron = get_pauli_kron(config["n_visible"], config["n_hidden"])

        # load h's and J's
        h = load_artifact(config_dir / "h.pkl")
        J = load_artifact(config_dir / "J.pkl")

        # compute the exact E and p for all s and T values
        data = {}
        errors = {}
        for s in s_values:
            A = anneal_schedule_data.loc[s, "A(s) (GHz)"]
            B = anneal_schedule_data.loc[s, "B(s) (GHz)"]
            for T in T_values:
                beta = 1 / (k_B_GHz_K * T)
                H = compute_H(h, J, A, B, n_qubits, pauli_kron)
                rho = compute_rho(H, beta)
                data[s, T] = {"E": np.diag(H).copy(), "p": np.diag(rho).copy()}

                T_bar.update(1)

            T_bar.reset()
            s_bar.update(1)

        # save the exact data and errors (if any)
        save_artifact(data, config_dir / "exact_data.pkl")
        if errors:
            save_artifact(errors, config_dir / "errors.pkl")

        s_bar.reset()
        config_bar.update(1)
