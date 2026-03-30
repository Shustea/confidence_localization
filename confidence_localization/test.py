import os
import tempfile

from runtime_setup import configure_runtime

_REPO_ROOT = configure_runtime(__file__)

import hydra
import torch
from tqdm import tqdm

import confidence_localization_dataloader as cld
from eval_utils import save_prediction_plots, summarize_prediction
from model import kappa_to_circ_std
from train import DOAMAMBA


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, default)
        return default if value is None else value
    return getattr(node, key, default)


def _eval_cfg(cfg):
    return _cfg_get(cfg, "eval", {})


def _checkpoint_path(cfg):
    eval_cfg = _eval_cfg(cfg)
    return _cfg_get(eval_cfg, "checkpoint_path", _cfg_get(cfg, "resume_from_checkpoint"))


def _plot_dir(cfg):
    eval_cfg = _eval_cfg(cfg)
    return _cfg_get(eval_cfg, "plot_dir", _cfg_get(cfg, "sample_dir", str(_REPO_ROOT / "samples")))


def _tmp_dir(cfg):
    eval_cfg = _eval_cfg(cfg)
    return _cfg_get(eval_cfg, "tmp_dir", str(_REPO_ROOT / "tmp"))


def _prepare_tmp_dir(cfg):
    tmp_dir = os.path.abspath(_tmp_dir(cfg))
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    tempfile.tempdir = tmp_dir
    return tmp_dir


def _resolve_device(cfg):
    eval_cfg = _eval_cfg(cfg)
    device_mode = str(_cfg_get(eval_cfg, "device", "auto")).lower()
    device_index = int(_cfg_get(eval_cfg, "device_index", _cfg_get(cfg, "default_gpu", 0)))

    if device_mode == "cpu":
        return torch.device("cpu")

    if device_mode in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise ValueError("eval.device is set to CUDA, but CUDA is not available on this machine.")
        return torch.device(f"cuda:{device_index}")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def _progress_total(eval_loader, max_batches):
    total_batches = len(eval_loader)
    if max_batches is None:
        return total_batches
    return min(total_batches, max(int(max_batches), 0))


def _short_title(title, limit=36):
    title = str(title)
    if len(title) <= limit:
        return title
    return title[: limit - 3] + "..."


def _write_sample_metrics(batch_idx, sample_idx, title, metrics):
    tqdm.write(f"[{batch_idx}:{sample_idx}] {title}")
    tqdm.write(f"  valid frames: {metrics['num_valid']}")
    tqdm.write(f"  acc@10 deg: {metrics['acc10']:.4f}")
    tqdm.write(f"  acc@15 deg: {metrics['acc15']:.4f}")
    tqdm.write(f"  coverage@pred-bound: {metrics['coverage']:.4f}")
    tqdm.write(f"  mean error [deg]: {metrics['mean_error_deg']:.2f}")
    tqdm.write(
        "  q20/q50/q70/q90/q95 [deg]: "
        f"{metrics['q20_deg']:.2f} / {metrics['q50_deg']:.2f} / {metrics['q70_deg']:.2f} / "
        f"{metrics['q90_deg']:.2f} / {metrics['q95_deg']:.2f}"
    )


@hydra.main(config_path="..", config_name="config", version_base="1.1")
def main(cfg):
    """Run evaluation for the configured data source and print aggregate metrics."""
    eval_cfg = _eval_cfg(cfg)
    max_plots = int(_cfg_get(eval_cfg, "max_plots", 3))
    max_batches = _cfg_get(eval_cfg, "max_batches", None)

    checkpoint_path = _checkpoint_path(cfg)
    if not checkpoint_path:
        raise ValueError("No checkpoint path was provided. Set eval.checkpoint_path or resume_from_checkpoint in config.yaml.")

    print("Stage 1/4: preparing evaluation data")
    eval_loader = cld.get_dataloader(cfg, stage="eval")
    total_batches = _progress_total(eval_loader, max_batches)
    device = _resolve_device(cfg)
    tmp_dir = _prepare_tmp_dir(cfg)
    print(f"Evaluating on device: {device}")
    print(f"Using temp directory: {tmp_dir}")
    print(f"Evaluation batches: {total_batches}")

    print("Stage 2/4: loading checkpoint")
    model = DOAMAMBA(cfg)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    if device.type != "cuda":
        print("Using CPU reference Mamba path (slower than CUDA).")

    saved_plots = 0
    all_errors = []
    total_valid = 0
    coverage_sum = 0.0
    acc10_sum = 0.0
    acc15_sum = 0.0

    print("Stage 3/4: running evaluation")
    with tqdm(total=total_batches, desc="eval", dynamic_ncols=True) as progress:
        for batch_idx, (spectrum, labels, titles) in enumerate(eval_loader):
            if batch_idx >= total_batches:
                break

            spectrum = spectrum.to(device)
            labels = labels.to(device)
            if isinstance(titles, str):
                titles = [titles]

            with torch.no_grad():
                doa_unit, kappa = model(spectrum)

            bound = kappa_to_circ_std(model.alpha.exp() * kappa)
            doa = torch.atan2(doa_unit[..., 1], doa_unit[..., 0])

            last_title = None
            for sample_idx in range(doa.shape[0]):
                title = titles[sample_idx] if sample_idx < len(titles) else f"sample_{batch_idx}_{sample_idx}"
                metrics = summarize_prediction(model, doa[sample_idx], labels[sample_idx], bound[sample_idx])
                _write_sample_metrics(batch_idx, sample_idx, title, metrics)
                last_title = title

                if metrics["num_valid"] > 0:
                    all_errors.append(metrics["errors_deg"])
                    total_valid += metrics["num_valid"]
                    coverage_sum += metrics["coverage"] * metrics["num_valid"]
                    acc10_sum += metrics["acc10"] * metrics["num_valid"]
                    acc15_sum += metrics["acc15"] * metrics["num_valid"]

                    if saved_plots < max_plots:
                        save_prediction_plots(
                            doa[sample_idx].detach().cpu(),
                            labels[sample_idx].detach().cpu(),
                            bound[sample_idx].detach().cpu(),
                            str(title),
                            f"eval_{batch_idx:03d}_{sample_idx:02d}_{title}",
                            _plot_dir(cfg),
                        )
                        saved_plots += 1

            if last_title is not None:
                progress.set_postfix_str(f"{_short_title(last_title)} plots={saved_plots}/{max_plots}")
            progress.update(1)

    print("Stage 4/4: aggregating metrics")
    if not all_errors:
        print("No valid evaluation frames were found for the configured dataset.")
        return

    errors = torch.cat(all_errors)
    print("=== aggregate ===")
    print(f"valid frames: {errors.numel()}")
    print(f"acc@10 deg: {acc10_sum / total_valid:.4f}")
    print(f"acc@15 deg: {acc15_sum / total_valid:.4f}")
    print(f"coverage@pred-bound: {coverage_sum / total_valid:.4f}")
    print(f"mean error [deg]: {errors.mean().item():.2f}")
    print(
        "q20/q50/q70/q90/q95 [deg]: "
        f"{errors.quantile(0.20).item():.2f} / {errors.quantile(0.50).item():.2f} / "
        f"{errors.quantile(0.70).item():.2f} / {errors.quantile(0.90).item():.2f} / {errors.quantile(0.95).item():.2f}"
    )


if __name__ == "__main__":
    main()
