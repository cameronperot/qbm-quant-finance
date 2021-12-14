import numpy as np

from qbm.utils import get_project_dir, load_artifact


project_dir = get_project_dir()


def load_samples(load_dir):
    embedding_samples = {}
    embedding_dirs = sorted([x for x in load_dir.iterdir()])
    for embedding_dir in embedding_dirs:
        embedding_id = int(embedding_dir.name)
        embedding_samples[embedding_id] = {}
        sample_names = sorted([x.stem for x in (embedding_dir / "samples").iterdir()])
        for sample_name in sample_names:
            relative_chain_strength = [
                x for x in sample_name.split("-") if "relative_chain_strength" in x
            ][0].split("=")[1]
            embedding_samples[embedding_id][relative_chain_strength] = load_artifact(
                embedding_dir / f"samples/{sample_name}.pkl"
            )

    return embedding_samples


def process_samples(embedding_samples):
    success_probabilities = {}
    mean_energies = {}
    for embedding_id, samples in embedding_samples.items():
        relative_chain_strengths = sorted(samples.keys())
        success_probabilities[embedding_id] = {
            "relative_chain_strength": relative_chain_strengths
        }
        mean_energies[embedding_id] = {"relative_chain_strength": relative_chain_strengths}
        for relative_chain_strength in relative_chain_strengths:
            energy = samples[relative_chain_strength].record.energy
            success_probabilities[embedding_id]["success_probability"] = np.mean(
                energy == energy.min()
            )
            mean_energies[embedding_id]["mean_energy"] = energy.mean()

    return success_probabilities, mean_energies


if __name__ == "__main__":
    n_visible = 64
    n_hidden = 30

    load_dir = project_dir / f"artifacts/embeddings/{n_visible}x{n_hidden}"

    embedding_samples = load_samples(load_dir)
    success_probabilities, mean_energies = process_samples(embedding_samples)
