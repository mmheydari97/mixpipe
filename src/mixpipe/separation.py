"""Source separation, for when the input is an already-mixed recording.

Wraps Demucs (`pip install demucs`, see requirements-chords.txt). Demucs is
trained on drums/bass/vocals/other, so a guitar+harmonica duo mostly lands in
"other" - useful for removing a backing track or isolating a vocal, less useful
for splitting two similar acoustic instruments.

Runs on Apple Silicon (MPS) or CUDA when available, otherwise CPU.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .io_utils import Stem, to_sample_rate

DEMUCS_SOURCES = ("drums", "bass", "other", "vocals")

_separators: dict[tuple[str, str], object] = {}


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _separator(model: str):
    try:
        import demucs.api
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "source separation needs Demucs: pip install -r requirements-chords.txt"
        ) from exc
    device = _device()
    key = (model, device)
    if key not in _separators:
        _separators[key] = demucs.api.Separator(model=model, device=device)
    return _separators[key]


def separate(
    audio: np.ndarray,
    sr: int,
    model: str = "htdemucs",
    progress: Callable[[float], None] | None = None,
) -> list[Stem]:
    """Split a mixed recording into Demucs stems.

    Returns one Stem per source (drums, bass, other, vocals), at the input
    sample rate and with the input's channel layout (mono in, mono out).
    """
    import torch

    sep = _separator(model)
    if progress is not None:
        def _cb(info: dict) -> None:
            total = max(1, info.get("audio_length", 1))
            progress(min(1.0, (info.get("segment_offset", 0) + 1) / total))

        sep.update_parameter(callback=_cb)

    was_mono = audio.ndim == 1
    stereo = np.stack([audio, audio], axis=1) if was_mono else audio[:, :2]
    model_sr = sep.samplerate
    stereo = to_sample_rate(stereo, sr, model_sr)

    wav = torch.from_numpy(np.ascontiguousarray(stereo.T, dtype=np.float32))
    _, sources = sep.separate_tensor(wav, model_sr)

    stems: list[Stem] = []
    for name, tensor in sources.items():
        out = tensor.detach().cpu().numpy().T  # (n, 2)
        out = to_sample_rate(out, model_sr, sr)
        if was_mono:
            out = out.mean(axis=1)
        role = "vocal" if name == "vocals" else name
        stems.append(Stem(name=name, audio=out.astype(np.float32), sr=sr, role=role,
                          meta={"separated_by": model}))
    return stems
