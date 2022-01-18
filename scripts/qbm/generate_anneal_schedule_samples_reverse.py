from decimal import Decimal

import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from minorminer import find_embedding

from qbm.utils import get_project_dir, get_rng, load_artifact, save_artifact


def main(h, J, config, anneal_params_dict, qpu, save_dir, embedding_id, gauge_id, gauge):
    h = h * gauge
    J = J * np.outer(gauge, gauge)
    save_artifact(
        gauge,
        save_dir / f"samples/embedding_{embedding_id:02}/gauge_{gauge_id:02}/gauge.pkl",
    )

    embedding_path = save_dir / f"samples/embedding_{embedding_id:02}/embedding.json"
    if embedding_path.exists():
        # load the saved embedding
        print(f"Loading embedding at {embedding_path}")
        embedding = load_artifact(embedding_path)

        # convert the keys to integers (issue with json loading/saving)
        for k, v in list(embedding.items()):
            embedding[int(k)] = v
            del embedding[k]
    else:
        # generate the underlying graphical structure to use for determining the embedding
        source_edgelist = []
        for i in range(config["n_visible"]):
            for j in range(config["n_visible"], config["n_qubits"]):
                source_edgelist.append((i, j))
        _, target_edgelist, target_adjacency = qpu.structure

        # generate direct embedding
        direct = False
        while not direct:
            embedding = find_embedding(source_edgelist, target_edgelist)
            for qubits in embedding.values():
                if len(qubits) > 1:
                    break
            else:
                direct = True

        save_artifact(embedding, embedding_path)

    # initialize the sampler
    sampler = FixedEmbeddingComposite(qpu, embedding)

    # generate samples for each annealing schedule
    for i, (name, anneal_params) in enumerate(anneal_params_dict.items()):
        print(
            f"{i + 1} of {len(anneal_params_dict)} {name}, num_reads = {anneal_params['num_reads']}"
        )
        samples_path = (
            save_dir / f"samples/embedding_{embedding_id:02}/gauge_{gauge_id:02}/{name}.pkl"
        )

        # ensure that samples do not get accidentally overwritten
        if samples_path.exists():
            continue
            answer = input(
                f"Directory {save_dir}/samples/{name} already exists, do you wish to overwrite the data in it? [y/N] "
            )
            if answer.lower() not in ("yes", "y"):
                continue

        # generate samples using the annealer
        samples = sampler.sample_ising(h, J, auto_scale=False, label=name, **anneal_params)

        # undo the gauge transformation
        samples.record.sample *= gauge

        # save the samples
        save_artifact(samples, samples_path)


if __name__ == "__main__":
    config_id = 2
    embedding_id = 1

    project_dir = get_project_dir()
    config_dir = project_dir / f"artifacts/exact_analysis/{config_id:02}"

    config = load_artifact(config_dir / "config.json")
    n_visible = config["n_visible"]
    n_hidden = config["n_hidden"]
    n_qubits = config["n_qubits"]

    qpu = DWaveSampler(**config["qpu_params"])

    # configure h's and J's
    if (config_dir / "h.pkl").exists() and (config_dir / "J.pkl").exists():
        print(f"Loading h's and J's at {config_dir}")
        h = load_artifact(config_dir / "h.pkl")
        J = load_artifact(config_dir / "J.pkl")
    else:
        rng = get_rng(config["seed"])
        if config["distribution"]["type"] == "normal":
            μ = config["distribution"]["mu"]
            σ = config["distribution"]["sigma"]
            a = rng.normal(μ, σ, n_visible)
            b = rng.normal(μ, σ, n_hidden)
            W = rng.normal(μ, σ, (n_visible, n_hidden))
        elif config["distribution"]["type"] == "uniform":
            low = config["distribution"]["low"]
            high = config["distribution"]["high"]
            a = rng.uniform(low, high, n_visible)
            b = rng.uniform(low, high, n_hidden)
            W = rng.uniform(low, high, (n_visible, n_hidden))

        h = np.concatenate((a, b))
        J = np.zeros((n_qubits, n_qubits))
        J[:n_visible, n_visible:] = W

        h = np.clip(h, *qpu.properties["h_range"])
        J = np.clip(J, *qpu.properties["j_range"])

        save_artifact(h, config_dir / "h.pkl")
        save_artifact(J, config_dir / "J.pkl")

    # set anneal schedules and max allowed number of reads
    anneal_durations = [Decimal(x) for x in (20, 100)]
    s_pauses = [Decimal(str(round(x, 2))) for x in np.arange(0.4, 0.55, 0.05)]
    pause_durations = [Decimal(x) for x in (0, 10, 100, 1_000)]
    quench_slope = Decimal(1 / min(qpu.properties["annealing_time_range"]))
    max_problem_duration = 1_000_000 - 1_000  # subtract 1_000 for buffer
    anneal_params_dict = {}
    for anneal_duration in anneal_durations:
        for s_pause in s_pauses:
            for pause_duration in pause_durations:
                t_pause = anneal_duration * (1 - s_pause)
                quench_duration = (1 - s_pause) / quench_slope
                if pause_duration > 0:
                    anneal_schedule = [
                        (0, 1),
                        (t_pause, s_pause),
                        (t_pause + pause_duration, s_pause),
                        (t_pause + pause_duration + quench_duration, 1),
                    ]
                else:
                    anneal_schedule = [
                        (0, 1),
                        (t_pause, s_pause),
                        (t_pause + quench_duration, 1),
                    ]
                anneal_schedule = [(float(t), float(s)) for (t, s) in anneal_schedule]
                print(anneal_schedule)

                num_reads = min(int(max_problem_duration / anneal_schedule[-1][0]), 10_000)

                name = f"t_pause={float(t_pause)},s_pause={float(s_pause)},pause_duration={float(pause_duration)},quench_slope={float(quench_slope)},reverse=True,reinit=True"
                anneal_params_dict[name] = {
                    "anneal_schedule": anneal_schedule,
                    "num_reads": num_reads,
                    "initial_state": np.ones(n_qubits),
                    "reinitialize_state": True,
                }

    # sample different gauges for each anneal schedule
    gauge_ids = range(1, 11)
    for gauge_id in gauge_ids:
        print(f"Gauge {gauge_id} / {len(gauge_ids)}")
        rng = get_rng(gauge_id)
        gauge = rng.choice([-1, 1], n_qubits)
        main(
            h=h.copy(),
            J=J.copy(),
            config=config,
            anneal_params_dict=anneal_params_dict,
            qpu=qpu,
            save_dir=config_dir,
            embedding_id=embedding_id,
            gauge_id=gauge_id,
            gauge=gauge,
        )
