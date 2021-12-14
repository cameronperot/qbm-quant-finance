import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from minorminer import find_embedding

from qbm.utils import get_project_dir, get_rng, save_artifact


project_dir = get_project_dir()


def sample_embeddings(
    Q, n_visible, n_hidden, n_embeddings, n_samples, relative_chain_strengths
):
    # set the maximum absolute value of Q for determining the chain strength
    Q_max_abs = np.abs(Q).max()

    # initialize the QPU sampler
    sampler_qpu = DWaveSampler(region="eu-central-1", solver="Advantage_system5.1")

    # generate the underlying graphical structure to use for determining the embedding
    source_edgelist = []
    for i in range(n_visible):
        for j in range(n_visible, n_visible + n_hidden):
            source_edgelist.append((i, j))
    _, target_edgelist, target_adjacency = sampler_qpu.structure

    # generate multiple embeddings
    for i in range(1, n_embeddings + 1):
        save_dir = project_dir / f"artifacts/embeddings/{n_visible}x{n_hidden}/{i:02}"

        # ensure that embedding/samples do not get accidentally overwritten
        if (save_dir / "embedding.json").exists() or (save_dir / "samples").exists():
            answer = input(
                f"Directory {save_dir} already exists, do you wish to overwrite it? [y/N] "
            )
            if answer.lower() not in ("yes", "y"):
                continue

        embedding = find_embedding(source_edgelist, target_edgelist)
        sampler = FixedEmbeddingComposite(sampler_qpu, embedding)
        save_artifact(embedding, save_dir / "embedding.json")

        # for each embedding sample using each relative chain strength
        for relative_chain_strength in relative_chain_strengths:
            chain_strength = relative_chain_strength * Q_max_abs
            samples = sampler.sample_qubo(
                Q,
                chain_strength=chain_strength,
                num_reads=n_samples,
                label=f"embedding_{i}-rcs={chain_strength}",
            )

            # save the samples
            save_artifact(
                samples,
                save_dir
                / f"samples/relative_chain_strength={relative_chain_strength}-n_samples={n_samples}.pkl",
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

    # sample and save the embeddings
    sample_embeddings(
        Q=Q,
        n_visible=n_visible,
        n_hidden=n_hidden,
        n_embeddings=10,
        n_samples=1000,
        relative_chain_strengths=np.arange(1, 11) / 10,
    )
