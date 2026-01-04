from functools import partial
import optuna
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torch import Tensor
from jaxtyping import Float32, BFloat16
from torchvision import io
from torchvision.transforms.v2 import functional as F
import time
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure,
)
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from torchmetrics.functional.image import lpips as LPIPS

from handheld_super_resolution.super_resolution_object import SRObject
from handheld_super_resolution.utils import getTime


# @torch.compile(mode="max-autotune", fullgraph=True)
# def compute_metrics(
#     sr: Float32[Tensor, "H W 3"], gt: BFloat16[Tensor, "3 H W"]
# ) -> tuple[Tensor, Tensor]:
#     sr = preprocess_sr(sr, gt.shape)
#     # NOTE: PSNR only works when using torch.compile -- otherwise it returns Nan!
#     # NOTE: IT was because we were using float16 instead of float32 and we need the full precision.
#     psnr = peak_signal_noise_ratio(sr, gt, data_range=1.0)

#     ssim = structural_similarity_index_measure(
#         sr[None, ...],
#         gt[None, ...],
#         data_range=1.0,
#     )

#     return psnr, ssim

@torch.compile(mode="max-autotune")
def compute_metrics(
    sr: Float32[Tensor, "H W 3"], gt: BFloat16[Tensor, "3 H W"], lpips_model,
) -> Tensor:
    sr = preprocess_sr(sr, gt.shape)
    lpips = lpips_model(sr[None, ...], gt[None,...], normalize=True).squeeze().mean()

    return lpips


def preprocess_sr(
    sr: Float32[Tensor, "H W 3"], gt_shape: torch.Size
) -> BFloat16[Tensor, "3 H W"]:
    sr = F.center_crop(
        sr.permute([
            2,
            0,
            1,
        ])
        .nan_to_num()
        .clip(0.0, 1.0),
        list(gt_shape[1:]),
    )
    sr /= sr.max()
    # We need the dynamic range of float32 but can't fit it all in memory, so we use bfloat16
    # NOTE: float16 doesn't have the dynamic range we need, so the PSNR computation results in NaN.
    sr = sr.to(dtype=torch.bfloat16)
    return sr


# def objective(
#     gt: BFloat16[Tensor, "3 H W"], sr_object: SRObject, trial: optuna.Trial
# ) -> tuple[float, float]:
#     # All parameters are replaced because we sample all of them; no parameters
#     # depend on the values of these parameters, so we can re-use sr_object without
#     # worrying about forgetting to reset some value or needing to do a deep copy of
#     # the config.
#     sr_object.config.robustness.tuning.update({
#         "t": trial.suggest_float("t", low=0.04, high=0.5),
#         "s1": trial.suggest_float("s1", low=0.2, high=5),
#         "s2": trial.suggest_float("s2", low=5, high=15),
#         "Mt": trial.suggest_float("Mt", low=0.2, high=2),
#     })

#     with torch.no_grad():
#         psnr, ssim = compute_metrics(sr_object.run()[0], gt)

#         return psnr.item(), ssim.item()

def objective(
    gt: BFloat16[Tensor, "3 H W"], sr_object: SRObject, lpips_model, trial: optuna.Trial,
) -> float:
    # All parameters are replaced because we sample all of them; no parameters
    # depend on the values of these parameters, so we can re-use sr_object without
    # worrying about forgetting to reset some value or needing to do a deep copy of
    # the config.
    sr_object.config.robustness.tuning.update({
        "t": trial.suggest_float("t", low=0.04, high=0.5),
        "s1": trial.suggest_float("s1", low=0.2, high=5),
        "s2": trial.suggest_float("s2", low=5, high=15),
        "Mt": trial.suggest_float("Mt", low=0.2, high=2),
    })

    with torch.no_grad():
        lpips = compute_metrics(sr_object.run()[0], gt, lpips_model)

        return lpips.item()


# def batched_objective(
#     gt: BFloat16[Tensor, "3 H W"], sr_object: SRObject, configs: list[dict[str, float]]
# ) -> list[tuple[float, float]]:
#     # Structure of arrays -> array of structures transformation for performance before returning?
#     psnrs: list[Tensor] = []
#     ssims: list[Tensor] = []

#     # TODO: Ideally we could run these with some sort of vmap instead of relying on mutated state like an object.
#     for config in configs:
#         # All parameters are replaced because we sample all of them; no parameters
#         # depend on the values of these parameters, so we can re-use sr_object without
#         # worrying about forgetting to reset some value or needing to do a deep copy of
#         # the config.
#         sr_object.config.robustness.tuning.update(config)
#         psnr, ssim = compute_metrics(sr_object.run()[0], gt)
#         # Clone them since they will be overwritten by the cuda graph
#         psnrs.append(psnr.clone())
#         ssims.append(ssim.clone())

#     objectives: list[tuple[float, float]] = [
#         (psnr.item(), ssim.item()) for (psnr, ssim) in zip(psnrs, ssims)
#     ]
#     return objectives


def batched_objective(
    gt: BFloat16[Tensor, "3 H W"], sr_object: SRObject, configs: list[dict[str, float]], lpips_model,
) -> list[float]:
    # Structure of arrays -> array of structures transformation for performance before returning?
    lpipss: list[Tensor] = []

    # TODO: Ideally we could run these with some sort of vmap instead of relying on mutated state like an object.
    for config in configs:
        # All parameters are replaced because we sample all of them; no parameters
        # depend on the values of these parameters, so we can re-use sr_object without
        # worrying about forgetting to reset some value or needing to do a deep copy of
        # the config.
        sr_object.config.robustness.tuning.update(config)
        lpips = compute_metrics(sr_object.run()[0], gt, lpips_model)
        # Clone them since they will be overwritten by the cuda graph
        lpipss.append(lpips.clone())

    objectives: list[float] = [lpips.item() for lpips in lpipss]
    return objectives


def batched_optimization(
    study: optuna.study.Study,
    batch_size: int,
    gt: BFloat16[Tensor, "3 H W"],
    sr_object: SRObject,
    lpips_model,
) -> None:
    # create batch
    trial_numbers: list[int] = []
    configs: list[dict[str, float]] = []
    for _ in range(batch_size):
        trial = study.ask()
        trial_numbers.append(trial.number)
        configs.append({
            # "t": trial.suggest_float("t", low=0.04, high=0.5),
            # "s1": trial.suggest_float("s1", low=0.2, high=5),
            # "s2": trial.suggest_float("s2", low=5, high=15),
            # "Mt": trial.suggest_float("Mt", low=0.2, high=2),
            "t": trial.suggest_float("t", low=0.01, high=10),
            "s1": trial.suggest_float("s1", low=0.01, high=10),
            "s2": trial.suggest_float("s2", low=1, high=20),
            "Mt": trial.suggest_float("Mt", low=0.1, high=10),
        })

    # evaluate batched objective
    with torch.no_grad():
        objectives = batched_objective(gt, sr_object, configs, lpips_model)

    # finish all trials in the batch
    for trial_number, objective in zip(trial_numbers, objectives):
        study.tell(trial_number, objective)


def main():
    # torch.backends.fp32_precision = "tf32"
    # torch.backends.cuda.matmul.fp32_precision = "tf32"
    # torch.backends.cudnn.fp32_precision = "tf32"
    # torch.backends.cudnn.conv.fp32_precision = "tf32"
    # torch.backends.cudnn.rnn.fp32_precision = "tf32"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.capture_scalar_outputs = True
    # torch._dynamo.config.capture_dynamic_output_shape_ops = True
    # torch._dynamo.config.nested_graph_breaks = True
    torch._dynamo.config.use_graph_deduplication = True
    torch._inductor.config.memory_planning = True
    torch._inductor.config.permute_fusion = True
    # torch._inductor.config.combo_kernels = True
    # torch._inductor.config.benchmark_combo_kernel = True
    # torch._inductor.config.combo_kernels_autotune = 2
    # torch._inductor.config.combo_kernel_allow_mixed_sizes = 2

    # current_time = time.perf_counter()
    # print("loading dng burst")
    # arw_burst = Burst.load(str(Path.home() / "Pictures" / "ARW" / "2025-09-06" / "011"), "*.ARW")
    # current_time = getTime(current_time, "loaded dng burst")

    gp_sampler = optuna.samplers.GPSampler(
        seed=42, n_startup_trials=4196, deterministic_objective=True
    )
    study = optuna.create_study(
        storage="sqlite:///corrected-color.db",
        sampler=gp_sampler,
        study_name="corrected-color",
        # directions=["maximize", "maximize"],
        directions=["minimize"],
        load_if_exists=True,
    )

    with torch.no_grad():
        gt = F.to_dtype(
            io.decode_image(str(Path.home() / "Pictures" / "DSC05288_DxO-all-x2.png")),
            torch.float32,
            scale=True,
        ).cuda()
        gt /= gt.max()
        gt = gt.to(dtype=torch.bfloat16)

        # torchvision.utils.save_image(list(F.five_crop(gt, [256, 256])), "sample.jpg")

        lpips_model = LPIPS._NoTrainLpips(net="alex").to(device=gt.device, dtype=gt.dtype)

        sr_object = SRObject(
            str(Path.home() / "Pictures" / "ARW" / "2025-09-06" / "011"),
            OmegaConf.load("configs/ica.yaml"),
        )


    # # NOTE: Optuna requires the ability to compute gradients when using the GP sampler, so we must disable gradients
    # # inside the objective function.
    # # current_time = time.perf_counter()
    print("Beginning unbatched optimization")
    study.optimize(
        partial(objective, gt, sr_object, lpips_model), n_trials=20, gc_after_trial=False
    )
    # # current_time = getTime(current_time, "Finished unbatched optimization")

    # current_time = time.perf_counter()
    # print("Running single optimization trial to warm JIT cache")
    # batched_optimization(study, 1, gt, sr_object)
    # current_time = getTime(current_time, "Finished warmup iteration")
    # print("Beginning batched optimization")
    # for i in range(100):
    #     batched_optimization(study, 50, gt, sr_object)
    #     current_time = getTime(current_time, f"Finished iteration {i}")


if __name__ == "__main__":
    main()
