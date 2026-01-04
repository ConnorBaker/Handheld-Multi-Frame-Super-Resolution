import torch
import helion
from helion.runtime.kernel import BoundKernel
from helion.autotuner.differential_evolution import DifferentialEvolutionSearch
from helion.autotuner.base_search import BaseAutotuner
from helion.autotuner.local_cache import StrictLocalAutotuneCache
import helion.language as hl
from collections.abc import Sequence


def my_autotuner_fn[_R](
    bound_kernel: BoundKernel[_R], args: Sequence[object], **kwargs: object
) -> BaseAutotuner:
    return StrictLocalAutotuneCache(
        DifferentialEvolutionSearch(
            bound_kernel,
            args,
            # population_size=40,
            # max_generations=40,
            population_size=10,
            max_generations=5,
            crossover_rate=0.8,
            immediate_update=True,
            **kwargs,
        )
    )


@helion.kernel
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out


if __name__ == "__main__":
    example_input = (
        torch.randn([2048, 2048], device="cuda"),
        torch.randn([2048, 2048], device="cuda"),
    )
    bound = matmul.bind(example_input)
    tuner = DifferentialEvolutionSearch(
        bound,
        example_input,
        # population_size=40,
        # max_generations=40,
        population_size=100,
        max_generations=10,
        crossover_rate=0.8,
        immediate_update=False,
    )
    best_config = tuner.autotune()
    best_config.save("configs/my_kernel.json")
    print("best_perf_so_far:", tuner.best_perf_so_far)
    print("statistics:", tuner.statistics())

