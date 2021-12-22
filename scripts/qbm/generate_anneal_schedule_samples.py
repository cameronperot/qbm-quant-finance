import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from minorminer import find_embedding

from qbm.utils import get_project_dir, get_rng, load_artifact, save_artifact


def main(h, J, config, anneal_schedules, qpu, save_dir, embedding_id, batch_id, gauge=None):
    if gauge is not None:
        h = h * gauge
        J = J * np.outer(gauge, gauge)
        save_artifact(
            gauge,
            save_dir / f"samples/embedding_{embedding_id:02}/batch_{batch_id:02}/gauge.pkl",
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
    for name, anneal_schedule in anneal_schedules.items():
        samples_path = (
            save_dir / f"samples/embedding_{embedding_id:02}/batch_{batch_id:02}/{name}.pkl"
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
            h,
            J,
            anneal_schedule=anneal_schedule,
            num_reads=config["sampling_params"]["num_reads"],
            auto_scale=False,
            label=name,
        )

        # save the samples
        save_artifact(samples, samples_path)


if __name__ == "__main__":
    config_id = 1
    embedding_id = 1
    batch_id = 1

    project_dir = get_project_dir()
    config_dir = project_dir / f"artifacts/exact_analysis/{config_id:02}"

    config = load_artifact(config_dir / "config.json")
    n_visible = config["n_visible"]
    n_hidden = config["n_hidden"]
    n_qubits = config["n_qubits"]

    qpu = DWaveSampler(**config["qpu_params"])

    if (config_dir / "h.pkl").exists() and (config_dir / "J.pkl").exists():
        print(f"Loading h's and J's at {config_dir}")
        h = load_artifact(config_dir / "h.pkl")
        J = load_artifact(config_dir / "J.pkl")
    else:
        μ = config["mu"]
        σ = config["sigma"]
        rng = get_rng(config["seed"])
        a = rng.normal(μ, σ, n_visible)
        b = rng.normal(μ, σ, n_hidden)
        W = rng.normal(μ, σ, (n_visible, n_hidden))

        h = np.concatenate((a, b))
        J = np.zeros((n_qubits, n_qubits))
        J[:n_visible, n_visible:] = W

        h = np.clip(h, *qpu.properties["h_range"])
        J = np.clip(J, *qpu.properties["j_range"])

        save_artifact(h, config_dir / "h.pkl")
        save_artifact(J, config_dir / "J.pkl")

    quench_start_times = range(4, 13)
    anneal_schedules = {
        "anneal_schedule=0,0_20,1": [(0, 0), (20, 1)],
    }
    max_slope = 2
    for t in quench_start_times:
        quench_start = (t, t / 20)
        quench_duration = (1 - quench_start[1]) / max_slope
        quench_stop = (t + quench_duration, 1)
        name = f"anneal_schedule=0,0_{quench_start[0]},{quench_start[1]}_{quench_stop[0]},{quench_stop[1]}"
        anneal_schedules[name] = [(0, 0), quench_start, quench_stop]

    for batch_id in range(1, 11):
        rng = get_rng(batch_id)
        gauge = rng.choice([-1, 1], n_qubits)
        main(
            h=h,
            J=J,
            config=config,
            anneal_schedules=anneal_schedules,
            qpu=qpu,
            save_dir=config_dir,
            embedding_id=embedding_id,
            batch_id=batch_id,
            gauge=gauge,
        )
