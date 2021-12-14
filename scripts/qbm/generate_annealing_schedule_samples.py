import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from minorminer import find_embedding

from qbm.utils import get_project_dir, get_rng, load_artifact, save_artifact


project_dir = get_project_dir()


def generate_annealing_schedule_samples(
    Q,
    n_visible,
    n_hidden,
    n_embeddings,
    n_samples,
    annealing_schedules,
    relative_chain_strengths,
):
    # set the maximum absolute value of Q for determining the chain strength
    Q_max_abs = np.abs(Q).max()

    # initialize the QPU sampler
    sampler_qpu = DWaveSampler(region="eu-central-1", solver="Advantage_system5.1")

    save_dir = (
        project_dir / f"artifacts/embeddings/{n_visible}x{n_hidden}/annealing_schedule"
    )
    if (save_dir / "embedding.json").exists():
        # load the saved embedding
        embedding = load_artifact(save_dir / "embedding.json")
    else:
        # generate the underlying graphical structure to use for determining the embedding
        source_edgelist = []
        for i in range(n_visible):
            for j in range(n_visible, n_visible + n_hidden):
                source_edgelist.append((i, j))
        _, target_edgelist, target_adjacency = sampler_qpu.structure

        # generate embedding
        embedding = find_embedding(source_edgelist, target_edgelist)
        sampler = FixedEmbeddingComposite(sampler_qpu, embedding)
        save_artifact(embedding, save_dir / "embedding.json")

    # generate multiple embeddings
    for name, annealing_schedule in annealing_schedules.items():
        # ensure that samples do not get accidentally overwritten
        if (save_dir / f"samples/{name}").exists():
            answer = input(
                f"Directory {save_dir} already exists, do you wish to overwrite it? [y/N] "
            )
            if answer.lower() not in ("yes", "y"):
                continue

        # generate and save samples for each of the relative chain strengths
        for relative_chain_strength in relative_chain_strengths:
            chain_strength = relative_chain_strength * Q_max_abs

            # generate samples using the annealer
            samples = None
            # samples = sampler.sample_qubo(
            # Q,
            # chain_strength=chain_strength,
            # num_reads=n_samples,
            # label=f"embedding_{i}-rcs={chain_strength}",
            # )

            # save the samples
            save_artifact(
                samples,
                save_dir
                / f"samples/{name}/relative_chain_strength={relative_chain_strength}-n_samples={n_samples}.pkl",
            )


if __name__ == "__main__":
    # size of the RBM
    n_visible = 64
    n_hidden = 30

    # generate a random Q matrix to use for sampling
    rng = get_rng(42)
    a = rng.normal(0, 0.01, n_visible)
    b = rng.normal(0, 0.01, n_hidden)
    W = rng.normal(0, 0.01, (n_visible, n_hidden))
    Q = np.diag(np.concatenate((a, b)))
    Q[:n_visible, n_visible:] = W

    # set annealing schedules
    annealing_schedules = {
        "default": [(0, 0), (20, 1)],
    }
    for i in range(4, 13):
        quench_start = (i, i / 20)
        quench_stop = (i + 1, 1)
        name = (
            f"quench_{quench_start[0]},{quench_start[1]}_{quench_stop[0]},{quench_stop[1]}"
        )
        annealing_schedules[name] = [(0, 0), quench_start, quench_stop]

    # set the relative chain strengths
    relative_chain_strengths = np.arange(1, 11) / 10

    # sample and save the embeddings
    generate_annealing_schedule_samples(
        Q=Q,
        n_visible=n_visible,
        n_hidden=n_hidden,
        n_embeddings=10,
        n_samples=1000,
        annealing_schedules=annealing_schedules,
        relative_chain_strengths=relative_chain_strengths,
    )
