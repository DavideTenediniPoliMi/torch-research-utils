import contextlib
import random

import numpy as np
import torch


def set_seeds(seed: int) -> np.random.Generator:
    """Seed all primary randomness sources for reproducibility:
    - python `random` module
    - `torch`
    - `torch.cuda`
    - `numpy`
    Ensure deterministic behavior in torch operations.
    Returns a NumPy random generator (on top of seeding the global NumPy RNG).

    Args:
        seed (int): the seed to set

    Returns:
        np.random.Generator: a NumPy random generator instance seeded with the
        given seed.
    """

    # Python random module
    random.seed(seed)

    # Torch & Cuda reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)

        with contextlib.suppress(AttributeError):
            # This attribute is only in newer PyTorch versions
            torch.set_float32_matmul_precision("highest")

    # NumPy generator
    np.random.seed(seed)  # noqa: NPY002
    return np.random.default_rng(seed)


def get_seed_sequence(num_seeds: int, rng: np.random.Generator | None) -> list[int]:
    """Generate a sequence of `num_seeds` random seeds. If passed, use `rng`
    for the generation, otherwise use python `random`. The generated seeds are
    in the range [0, 2**32 - 1].

    Args:
        num_seeds (int): number of seeds to generate
        rng (np.random.Generator | None): random number generator to use

    Returns:
        list[int]: list of generated random seeds
    """
    if rng is None:
        return [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]
    return rng.integers(low=0, high=2**32 - 1, size=num_seeds, dtype=np.uint32).tolist()
