import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from minorminer import find_embedding

from qbm.utils import get_project_dir, get_rng, load_artifact, save_artifact


project_dir = get_project_dir()


def generate_anneal_schedule_samples(
    Q, n_visible, n_hidden, n_samples, anneal_schedules, save_dir
):
    # set the maximum absolute value of Q for determining the chain strength
    Q_max_abs = np.abs(Q).max()

    # initialize the QPU sampler
    sampler_qpu = DWaveSampler(region="eu-central-1", solver="Advantage_system5.1")

    if (save_dir / "embedding.json").exists():
        # load the saved embedding
        print(f"Loading embedding at {save_dir / 'embedding.json'}")
        embedding = load_artifact(save_dir / "embedding.json")

        # convert the keys to integers (issue with json loading/saving)
        for k, v in list(embedding.items()):
            embedding[int(k)] = v
            del embedding[k]
    else:
        # generate the underlying graphical structure to use for determining the embedding
        source_edgelist = []
        for i in range(n_visible):
            for j in range(n_visible, n_visible + n_hidden):
                source_edgelist.append((i, j))
        _, target_edgelist, target_adjacency = sampler_qpu.structure

        # generate embedding
        embedding = find_embedding(source_edgelist, target_edgelist)
        save_artifact(embedding, save_dir / "embedding.json")

    # initialize the sampler
    sampler = FixedEmbeddingComposite(sampler_qpu, embedding)

    # generate samples for each annealing schedule
    for name, anneal_schedule in anneal_schedules.items():
        # ensure that samples do not get accidentally overwritten
        if (save_dir / f"samples/{name}.pkl").exists():
            answer = input(
                f"Directory {save_dir}/samples/{name} already exists, do you wish to overwrite the data in it? [y/N] "
            )
            if answer.lower() not in ("yes", "y"):
                continue

        # generate samples using the annealer
        samples = sampler.sample_qubo(
            Q, anneal_schedule=anneal_schedule, num_reads=n_samples, label=name,
        )

        # save the samples
        save_artifact(
            samples, save_dir / f"samples/{name}.pkl",
        )


if __name__ == "__main__":
    # size of the RBM
    n_visible = 6
    n_hidden = 6

    # generate a random Q matrix to use for sampling
    rng = get_rng(42)
    a = rng.normal(0, 0.01, n_visible)
    b = rng.normal(0, 0.01, n_hidden)
    W = rng.normal(0, 0.01, (n_visible, n_hidden))
    Q = np.diag(np.concatenate((a, b)))
    Q[:n_visible, n_visible:] = W

    # set anneal schedules
    anneal_schedules = {
        "anneal_schedule=0,0_20,1": [(0, 0), (20, 1)],
    }
    for i in range(4, 13):
        quench_start = (i, i / 20)
        quench_duration = (1 - quench_start[1]) / 2
        quench_stop = (i + quench_duration, 1)
        name = f"anneal_schedule=0,0_{quench_start[0]},{quench_start[1]}_{quench_stop[0]},{quench_stop[1]}"
        anneal_schedules[name] = [(0, 0), quench_start, quench_stop]

    save_dir = project_dir / f"artifacts/anneal_schedule_samples/{n_visible}x{n_hidden}_01/"

    # sample and save the embeddings
    generate_anneal_schedule_samples(
        Q=Q,
        n_visible=n_visible,
        n_hidden=n_hidden,
        n_samples=10 ** 4,
        anneal_schedules=anneal_schedules,
        save_dir=save_dir,
    )
