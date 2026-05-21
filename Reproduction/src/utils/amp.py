from __future__ import annotations

from contextlib import nullcontext


try:
    from torch.amp import GradScaler as TorchAmpGradScaler
    from torch.amp import autocast as torch_amp_autocast

    def build_grad_scaler(enabled: bool):
        return TorchAmpGradScaler("cuda", enabled=enabled)

    def autocast_context(enabled: bool):
        return torch_amp_autocast("cuda", enabled=enabled)

except ImportError:
    from torch.cuda.amp import GradScaler as CudaGradScaler
    from torch.cuda.amp import autocast as cuda_autocast

    def build_grad_scaler(enabled: bool):
        return CudaGradScaler(enabled=enabled)

    def autocast_context(enabled: bool):
        return cuda_autocast(enabled=enabled) if enabled else nullcontext()
