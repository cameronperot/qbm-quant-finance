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
        for i in range(config["n_qubits"]):
            for j in range(i + 1, config["n_qubits"]):
                source_edgelist.append((i, j))
        _, target_edgelist, target_adjacency = qpu.structure

        # generate embedding
        embedding = find_embedding(source_edgelist, target_edgelist)

        print(embedding)

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
            answer = input(
                f"Directory {save_dir}/samples/{name} already exists, do you wish to overwrite the data in it? [y/N] "
            )
            if answer.lower() not in ("yes", "y"):
                continue

        # generate samples using the annealer
        samples = sampler.sample_ising(
            h, J, chain_strength=1, auto_scale=False, label=name, **anneal_params,
        )

        # undo the gauge transformation
        samples.record.sample *= gauge

        # save the samples
        save_artifact(samples, samples_path)


if __name__ == "__main__":
    config_id = 4
    embedding_id = 1
    gauge_id = 1

    project_dir = get_project_dir()
    config_dir = project_dir / f"artifacts/exact_analysis/{config_id:02}"

    config = load_artifact(config_dir / "config.json")
    n_visible = config["n_visible"]
    n_hidden = config["n_hidden"]
    n_qubits = config["n_qubits"]

    qpu = DWaveSampler(**config["qpu_params"])

    # configure h's and J's
    print(f"Loading h's and J's at {config_dir}")
    h = load_artifact(config_dir / "h.pkl")
    J = load_artifact(config_dir / "J.pkl")

    # set anneal schedules and max allowed number of reads
    anneal_params_dict = {}
    for max_slope in (1, 2):
        if max_slope == 2:
            anneal_durations = [Decimal(x) for x in (1, 20)]
        elif max_slope == 1:
            anneal_durations = [Decimal(x) for x in (1,)]

        s_pauses = [Decimal(str(round(x, 3))) for x in np.arange(5.5, 6.75, 0.25) / 10]
        pause_durations = [Decimal(x) for x in (10, 100, 1_000)]
        max_problem_duration = 1_000_000 - 1_000  # subtract 1_000 for small buffer
        for anneal_duration in anneal_durations:
            for s_pause in s_pauses:
                for pause_duration in pause_durations:
                    t_pause = anneal_duration * s_pause
                    quench_duration = (1 - s_pause) / max_slope
                    anneal_schedule = [
                        (0, 0),
                        (t_pause, s_pause),
                        (t_pause + pause_duration, s_pause),
                        (t_pause + pause_duration + quench_duration, 1),
                    ]
                    anneal_schedule = [(float(t), float(s)) for (t, s) in anneal_schedule]

                    num_reads = min(
                        int(
                            max_problem_duration
                            / (t_pause + pause_duration + quench_duration)
                        ),
                        10_000,
                    )

                    name = f"anneal_duration={anneal_duration},s_pause={s_pause:.3f},pause_duration={pause_duration},max_slope={max_slope}"
                    anneal_params_dict[name] = {
                        "anneal_schedule": anneal_schedule,
                        "num_reads": num_reads,
                    }

    # sample different gauges for each anneal schedule
    gauge_ids = range(1, 11)
    for gauge_id in gauge_ids:
        print(f"Gauge {gauge_id} / {len(gauge_ids)}")
        rng = get_rng(gauge_id)
        gauge = rng.choice([-1, 1], n_qubits)
        main(
            h=h,
            J=J,
            config=config,
            anneal_params_dict=anneal_params_dict,
            qpu=qpu,
            save_dir=config_dir,
            embedding_id=embedding_id,
            gauge_id=gauge_id,
            gauge=gauge,
        )
