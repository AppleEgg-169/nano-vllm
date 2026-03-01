import importlib.util
from pathlib import Path

import torch
import triton


# ===== fixed test config (edit these 5 lines if needed) =====
BSZ = 32
SEQLEN = 512
HIDDEN = 4096
DTYPE = torch.float16  # torch.float16 / torch.bfloat16
EPS = 1e-6


def _load_rmsnorm():
    root = Path(__file__).resolve().parent
    path = root / "nanovllm" / "ops" / "rmsnorm.py"
    spec = importlib.util.spec_from_file_location("nano_rmsnorm", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def torch_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    y = x_fp32 * torch.rsqrt(var + eps)
    return y.to(x.dtype) * weight


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    mod = _load_rmsnorm()
    rms_norm_forward = mod.rms_norm_forward

    x = torch.randn((BSZ, SEQLEN, HIDDEN), device="cuda", dtype=DTYPE)
    weight = torch.randn((HIDDEN,), device="cuda", dtype=DTYPE)

    # correctness
    out_tri = rms_norm_forward(x, weight, EPS)
    out_ref = torch_ref(x, weight, EPS)
    abs_err = (out_tri.float() - out_ref.float()).abs().max().item()
    rel_err = (
        (
            (out_tri.float() - out_ref.float()).abs()
            / out_ref.float().abs().clamp_min(1e-12)
        )
        .max()
        .item()
    )
    print(f"max_abs_err={abs_err:.3e}, max_rel_err={rel_err:.3e}")

    # performance (triton.testing)
    triton_ms, _, _ = triton.testing.do_bench(
        lambda: rms_norm_forward(x, weight, EPS),
        quantiles=[0.5, 0.2, 0.8],
        rep=50,
        warmup=25,
    )
    torch_ms, _, _ = triton.testing.do_bench(
        lambda: torch_ref(x, weight, EPS), quantiles=[0.5, 0.2, 0.8], rep=50, warmup=25
    )
    print(
        f"shape=({BSZ},{SEQLEN},{HIDDEN}), dtype={DTYPE}, "
        f"triton={triton_ms:.3f}ms, torch={torch_ms:.3f}ms, speedup={torch_ms / triton_ms:.2f}x"
    )


if __name__ == "__main__":
    main()
