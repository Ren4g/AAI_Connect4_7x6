"""Small helpers for recording peak device memory consistently across resumed runs."""
from __future__ import annotations


def peak_vram_bytes(torch_module, device, previous_peak: int = 0) -> int:
    """Return the largest CUDA allocated-memory peak across this and earlier segments."""
    current = (int(torch_module.cuda.max_memory_allocated(device))
               if getattr(device, "type", str(device)) == "cuda" else 0)
    return max(int(previous_peak), current)


def reset_peak_vram_stats(torch_module, device) -> None:
    """Start a fresh per-segment CUDA peak after model and checkpoint setup."""
    if getattr(device, "type", str(device)) == "cuda":
        torch_module.cuda.reset_peak_memory_stats(device)
