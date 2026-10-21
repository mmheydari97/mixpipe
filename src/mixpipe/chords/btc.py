"""Neural chord recognition: the BTC model, as an optional second opinion.

BTC is the Bi-directional Transformer for Chord recognition (Park et al., ISMIR
2019; https://github.com/jayg996/BTC-ISMIR19, MIT licence). This module is a
re-implementation of its published architecture, written so the authors' trained
weights load into it, plus the same feature extraction, so that a trained model can
sit behind `recognize()` as the TODO always said it should.

  features   CQT, 144 bins at 24 per octave, hop 2048 at 22.05 kHz, log magnitude,
             computed in 10 s pieces (the 108 frames the model is trained on),
             then standardised with the mean and std stored in the checkpoint
  model      8 layers of forward and backward masked self-attention (4 heads,
             128 wide) with convolutional feed-forwards, 170 outputs: 12 roots x 14
             qualities, X (unknown) and N (no chord)
  output     `neural_probs()` gives the per-frame probabilities; `recognize()` turns
             them into scores for its own labels and blends them with its template
             scores (see `recognize.neural_scores`)

The weights (12 MB) are not part of this repository. `ensure_model()` downloads them
once into ~/.cache/mixpipe/models/btc (or $MIXPIPE_MODEL_DIR) from a pinned commit of
the authors' repository and checks their SHA-256 before use. The checkpoint is
read with `weights_only=True` (tensors and numpy scalars only), so nothing in it
can run code. Needs torch (installed with Demucs).
"""

from __future__ import annotations

import hashlib
import math
import os
import urllib.request
from pathlib import Path

import numpy as np

COMMIT = "2682317be668032e6e4b269ded36adaa2ad57df0"
MODEL_FILE = "btc_model_large_voca.pt"
MODEL_URL = f"https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/{COMMIT}/test/{MODEL_FILE}"
MODEL_SHA256 = "1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51"

SR = 22_050
HOP = 2048
N_BINS = 144
BINS_PER_OCTAVE = 24
CHUNK_S = 10.0
TIMESTEP = 108                      # frames per model window (= 10 s)
FRAME_S = HOP / SR

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = ["min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7",
             "dim7", "hdim7", "sus2", "sus4"]
N_CLASSES = 170                     # 12 * 14, X = 168, N = 169
X_INDEX, N_INDEX = 168, 169
# BTC quality -> the closest quality the template recognizer knows ("" = major)
TO_OURS = {"min": "m", "maj": "", "dim": "dim", "aug": "aug", "min6": "m", "maj6": "",
           "min7": "m7", "minmaj7": "m", "maj7": "maj7", "7": "7", "dim7": "dim",
           "hdim7": "dim", "sus2": "sus2", "sus4": "sus4"}


def label_of(index: int) -> str:
    """The BTC name of a class, e.g. 'C#:min7', 'N'."""
    if index == N_INDEX:
        return "N"
    if index == X_INDEX:
        return "X"
    return f"{ROOTS[index // 14]}:{QUALITIES[index % 14]}"


class ModelUnavailable(RuntimeError):
    """The model can't be used (no torch, no network and no cached copy, bad checksum)."""


# --------------------------------------------------------------------------- #
# getting the weights
# --------------------------------------------------------------------------- #


def model_dir() -> Path:
    base = os.environ.get("MIXPIPE_MODEL_DIR")
    return Path(base) / "btc" if base else Path.home() / ".cache" / "mixpipe" / "models" / "btc"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_model(progress=None, url: str = MODEL_URL, sha256: str = MODEL_SHA256) -> Path:
    """The path of the weights, downloading them first if needed. Raises
    ModelUnavailable if the download fails or the file is not the one pinned."""
    path = model_dir() / MODEL_FILE
    if path.exists():
        if _sha256(path) == sha256:
            return path
        path.unlink()                                 # damaged or replaced: fetch it again
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        if progress:
            progress("downloading the BTC chord model (12 MB, once)")
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            while True:
                block = resp.read(1 << 18)
                if not block:
                    break
                out.write(block)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailable(f"couldn't download the BTC model from {url}: {exc}") from exc
    if _sha256(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailable("the downloaded BTC model doesn't match its pinned checksum; "
                               "refusing to use it")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- #
# the network (parameter names match the authors' checkpoint)
# --------------------------------------------------------------------------- #

_torch = None


def _imports():
    global _torch
    if _torch is None:
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailable("the neural recognizer needs torch (installed with Demucs: "
                                   "pip install -r requirements-chords.txt)") from exc
        _torch = torch
    return _torch


def build_model(hidden: int = 128, layers: int = 8, heads: int = 4, timestep: int = TIMESTEP,
                feature_size: int = N_BINS, classes: int = N_CLASSES):
    """The BTC network as a torch Module (eval mode, CPU)."""
    torch = _imports()
    nn = torch.nn

    def bias_mask(length: int, backward: bool):
        mask = torch.triu(torch.full((length, length), float("-inf")), 1)
        return (mask.T if backward else mask)[None, None]          # [1, 1, L, L]

    def timing_signal(length: int, channels: int):
        position = np.arange(length)
        n = channels // 2
        step = math.log(1.0e4) / (n - 1)
        inv = np.exp(np.arange(n, dtype=np.float64) * -step)
        scaled = position[:, None] * inv[None, :]
        sig = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
        return torch.from_numpy(sig[None]).float()

    class LayerNorm(nn.Module):
        def __init__(self, features: int, eps: float = 1e-6):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)              # unbiased, with eps added to the std
            return self.gamma * (x - mean) / (std + self.eps) + self.beta

    class MultiHeadAttention(nn.Module):
        def __init__(self, backward: bool):
            super().__init__()
            self.num_heads = heads
            self.query_scale = (hidden // heads) ** -0.5
            self.query_linear = nn.Linear(hidden, hidden, bias=False)
            self.key_linear = nn.Linear(hidden, hidden, bias=False)
            self.value_linear = nn.Linear(hidden, hidden, bias=False)
            self.output_linear = nn.Linear(hidden, hidden, bias=False)
            self.bias_mask = bias_mask(timestep, backward)

        def split(self, x):
            b, t, d = x.shape
            return x.view(b, t, self.num_heads, d // self.num_heads).permute(0, 2, 1, 3)

        def forward(self, x):
            q = self.split(self.query_linear(x)) * self.query_scale
            k, v = self.split(self.key_linear(x)), self.split(self.value_linear(x))
            logits = q @ k.permute(0, 1, 3, 2)
            logits = logits + self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits)
            ctx = torch.softmax(logits, dim=-1) @ v
            b, h, t, d = ctx.shape
            return self.output_linear(ctx.permute(0, 2, 1, 3).contiguous().view(b, t, h * d))

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad1d((1, 1), 0)                     # kernel 3, padded both sides
            self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=0)

        def forward(self, x):
            return self.conv(self.pad(x.permute(0, 2, 1))).permute(0, 2, 1)

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Conv(), Conv()])

        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))                  # ReLU after every layer, the last too
            return x

    class AttentionBlock(nn.Module):
        def __init__(self, backward: bool):
            super().__init__()
            self.multi_head_attention = MultiHeadAttention(backward)
            self.positionwise_convolution = FeedForward()
            self.layer_norm_mha = LayerNorm(hidden)
            self.layer_norm_ffn = LayerNorm(hidden)

        def forward(self, x):
            x = x + self.multi_head_attention(self.layer_norm_mha(x))
            return x + self.positionwise_convolution(self.layer_norm_ffn(x))

    class BiDirectionalLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_block = AttentionBlock(backward=False)
            self.backward_attn_block = AttentionBlock(backward=True)
            self.linear = nn.Linear(hidden * 2, hidden)

        def forward(self, x):
            return self.linear(torch.cat([self.attn_block(x), self.backward_attn_block(x)], dim=2))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.timing_signal = timing_signal(timestep, hidden)
            self.embedding_proj = nn.Linear(feature_size, hidden, bias=False)
            self.self_attn_layers = nn.Sequential(*[BiDirectionalLayer() for _ in range(layers)])
            self.layer_norm = LayerNorm(hidden)

        def forward(self, x):
            x = self.embedding_proj(x) + self.timing_signal[:, :x.shape[1], :].type_as(x)
            return self.layer_norm(self.self_attn_layers(x))

    class Output(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_projection = nn.Linear(hidden, classes)
            # present in the checkpoint, unused at inference (it belongs to the CRF variant)
            self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden // 2, batch_first=True,
                                bidirectional=True)

    class BTC(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn_layers = Encoder()
            self.output_layer = Output()

        def forward(self, x):                              # [B, T, 144] -> logits [B, T, 170]
            return self.output_layer.output_projection(self.self_attn_layers(x))

    return BTC().eval()


_MODEL_CACHE: dict = {}


def load_model(path: Path | None = None, progress=None):
    """(model, mean, std), read from the downloaded weights (cached per process)."""
    torch = _imports()
    path = Path(path) if path else ensure_model(progress)
    key = str(path)
    if key not in _MODEL_CACHE:
        try:
            # the checkpoint stores mean/std as numpy scalars, pickled under numpy's old module
            # name; allow exactly that scalar constructor and the float64 dtype classes
            # it needs (inert numpy types) and nothing else
            scalar = getattr(getattr(np, "_core", None) or np.core, "multiarray").scalar
            with torch.serialization.safe_globals([(scalar, "numpy.core.multiarray.scalar"),
                                                   np.dtype, type(np.dtype("float64"))]):
                ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
            model = build_model()
            model.load_state_dict(ckpt["model"])
        except Exception as exc:  # noqa: BLE001 - any failure means "can't use this file"
            raise ModelUnavailable(f"couldn't load the BTC weights: {type(exc).__name__}: {exc}") from exc
        _MODEL_CACHE[key] = (model, float(ckpt["mean"]), float(ckpt["std"]))
    return _MODEL_CACHE[key]


# --------------------------------------------------------------------------- #
# audio -> probabilities
# --------------------------------------------------------------------------- #


def features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Log-CQT frames, shape (n_frames, 144), computed in 10 s pieces like the
    training data (each piece gives exactly the model's 108 frames)."""
    import librosa

    from ..io_utils import to_sample_rate

    wav = y if y.ndim == 1 else y.mean(axis=1)
    wav = to_sample_rate(wav.astype(np.float32), sr, SR)
    piece = int(SR * CHUNK_S)
    parts, cur = [], 0
    while len(wav) > cur + piece:
        parts.append(librosa.cqt(wav[cur:cur + piece], sr=SR, n_bins=N_BINS,
                                 bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP))
        cur += piece
    tail = wav[cur:]
    if len(tail) >= HOP:                                         # librosa needs a little audio
        parts.append(librosa.cqt(tail, sr=SR, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
                                 hop_length=HOP))
    if not parts:
        return np.zeros((0, N_BINS), dtype=np.float32)
    return np.log(np.abs(np.concatenate(parts, axis=1)) + 1e-6).T.astype(np.float32)


def neural_probs(y: np.ndarray, sr: int, progress=None, model_path: Path | None = None) -> dict:
    """Per-frame chord probabilities: {"probs": (n_frames, 170), "frame_s": float}.

    Frame f covers [f * frame_s, (f + 1) * frame_s). Raises ModelUnavailable when
    torch or the weights can't be had."""
    torch = _imports()
    model, mean, std = load_model(model_path, progress)
    feat = features(y, sr)
    n = len(feat)
    if n == 0:
        return {"probs": np.zeros((0, N_CLASSES), dtype=np.float32), "frame_s": FRAME_S}
    feat = (feat - mean) / std
    pad = (-n) % TIMESTEP
    feat = np.pad(feat, ((0, pad), (0, 0)), mode="constant", constant_values=0)
    windows = torch.from_numpy(feat.reshape(-1, TIMESTEP, N_BINS)).float()
    with torch.no_grad():
        probs = torch.softmax(model(windows), dim=-1).reshape(-1, N_CLASSES).numpy()
    return {"probs": probs[:n].astype(np.float32), "frame_s": FRAME_S}


def decode(probs: np.ndarray, frame_s: float = FRAME_S) -> list[tuple[float, float, str]]:
    """The most likely chord run by run, as [(start_s, end_s, 'C:maj7')]; handy for
    looking at what the model hears."""
    if len(probs) == 0:
        return []
    best = probs.argmax(axis=1)
    out, start = [], 0
    for i in range(1, len(best) + 1):
        if i == len(best) or best[i] != best[start]:
            out.append((start * frame_s, i * frame_s, label_of(int(best[start]))))
            start = i
    return out
