"""Tests for the optional neural chord recognizer (BTC) and how it blends into recognize().

No network and no real weights: checkpoints are random, the "download" is a file:// URL.
Needs torch (installed with Demucs).
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("librosa")
torch = pytest.importorskip("torch")

from mixpipe.chords import btc  # noqa: E402
from mixpipe.chords import recognize as rec  # noqa: E402
from mixpipe.chords.recognize import recognize  # noqa: E402

SR = 44_100
BEAT = 0.5


# --------------------------------------------------------------------------- #
# vocabulary and architecture
# --------------------------------------------------------------------------- #

def test_label_vocabulary():
    assert btc.label_of(0) == "C:min" and btc.label_of(1) == "C:maj"
    assert btc.label_of(14 * 9 + 9) == "A:7" and btc.label_of(14 * 11 + 8) == "B:maj7"
    assert btc.label_of(btc.X_INDEX) == "X" and btc.label_of(btc.N_INDEX) == "N"
    assert len({btc.label_of(i) for i in range(btc.N_CLASSES)}) == btc.N_CLASSES
    assert set(btc.TO_OURS) == set(btc.QUALITIES)
    assert set(btc.TO_OURS.values()) <= set(rec.QUALITIES)         # everything folds onto known qualities


def test_architecture_matches_the_published_model():
    model = btc.build_model()
    assert sum(p.numel() for p in model.parameters()) == 3_036_842   # the authors' checkpoint size
    names = set(model.state_dict())
    assert len(names) == 221
    for expected in (
        "self_attn_layers.embedding_proj.weight",
        "self_attn_layers.self_attn_layers.0.attn_block.multi_head_attention.query_linear.weight",
        "self_attn_layers.self_attn_layers.7.backward_attn_block.positionwise_convolution.layers.1.conv.bias",
        "self_attn_layers.self_attn_layers.3.linear.weight",
        "self_attn_layers.layer_norm.gamma",
        "output_layer.output_projection.bias",
        "output_layer.lstm.weight_ih_l0_reverse",
    ):
        assert expected in names, expected
    out = model(torch.randn(2, btc.TIMESTEP, btc.N_BINS))
    assert out.shape == (2, btc.TIMESTEP, btc.N_CLASSES) and torch.isfinite(out).all()


def test_attention_blocks_look_in_opposite_directions():
    model = btc.build_model()
    layer = model.self_attn_layers.self_attn_layers[0]
    fwd = layer.attn_block.multi_head_attention.bias_mask[0, 0]
    bwd = layer.backward_attn_block.multi_head_attention.bias_mask[0, 0]
    assert torch.isinf(fwd[0, 1]) and fwd[1, 0] == 0 and fwd[3, 3] == 0        # forward block: past only
    assert torch.isinf(bwd[1, 0]) and bwd[0, 1] == 0 and bwd[3, 3] == 0        # backward block: future only
    # and a change in the future cannot reach the forward block's earlier frames
    block = layer.attn_block.eval()
    x = torch.randn(1, btc.TIMESTEP, 128)
    y1 = block(x)
    x2 = x.clone()
    x2[0, 60:] += 5.0
    y2 = block(x2)
    assert torch.allclose(y1[0, :60], y2[0, :60], atol=1e-5) and not torch.allclose(y1[0, 60:], y2[0, 60:])


def test_features_shape_and_chunking():
    t = np.arange(btc.SR * 25) / btc.SR
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    f = btc.features(y, btc.SR)
    assert f.shape[1] == btc.N_BINS and np.isfinite(f).all()
    assert f.shape[0] == 108 + 108 + 54                      # 10 s + 10 s + the 5 s tail
    assert btc.features(y[: btc.SR * 10 - 5], btc.SR).shape[0] == 108      # under 10 s: one piece
    assert btc.features(np.zeros(10, dtype=np.float32), btc.SR).shape == (0, btc.N_BINS)
    # resamples any input rate
    assert abs(btc.features(np.repeat(y, 2)[: 44_100 * 12], 44_100).shape[0] - 130) <= 2


def test_decode():
    probs = np.zeros((6, btc.N_CLASSES))
    probs[0:2, 1] = probs[2:5, 14 * 7 + 1] = probs[5, btc.N_INDEX] = 1.0
    runs = btc.decode(probs, 0.5)
    assert runs == [(0.0, 1.0, "C:maj"), (1.0, 2.5, "G:maj"), (2.5, 3.0, "N")]
    assert btc.decode(np.zeros((0, btc.N_CLASSES))) == []


# --------------------------------------------------------------------------- #
# the weights: download, checksum, safe loading
# --------------------------------------------------------------------------- #

def random_checkpoint(path: Path, mean=-2.2, std=1.7, extra=None) -> str:
    model = btc.build_model()
    torch.manual_seed(0)
    for p in model.parameters():
        torch.nn.init.normal_(p, 0.0, 0.05)
    blob = {"model": model.state_dict(), "mean": mean, "std": std}
    if extra is not None:
        blob["extra"] = extra
    torch.save(blob, path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXPIPE_MODEL_DIR", str(tmp_path / "models"))
    btc._MODEL_CACHE.clear()
    yield tmp_path / "models"
    btc._MODEL_CACHE.clear()


def test_ensure_model_downloads_verifies_and_reuses(tmp_path, model_dir):
    src = tmp_path / "remote.pt"
    sha = random_checkpoint(src)
    seen = []
    path = btc.ensure_model(progress=seen.append, url=src.as_uri(), sha256=sha)
    assert path == model_dir / "btc" / btc.MODEL_FILE and path.exists() and seen
    src.unlink()                                          # the source is gone: the cached copy is used
    assert btc.ensure_model(url=src.as_uri(), sha256=sha) == path
    # a damaged cached copy is replaced rather than trusted
    src2 = tmp_path / "remote2.pt"
    random_checkpoint(src2)
    path.write_bytes(b"corrupt")
    with pytest.raises(btc.ModelUnavailable, match="checksum"):
        btc.ensure_model(url=src2.as_uri(), sha256="0" * 64)      # a wrong download is refused...
    assert not path.exists() and not list(path.parent.glob("*.part"))   # ...and leaves nothing behind
    good = hashlib.sha256(src2.read_bytes()).hexdigest()
    assert btc.ensure_model(url=src2.as_uri(), sha256=good).exists()


def test_ensure_model_reports_a_failed_download(tmp_path, model_dir):
    with pytest.raises(btc.ModelUnavailable, match="download"):
        btc.ensure_model(url=(tmp_path / "missing.pt").as_uri(), sha256="0" * 64)


def test_pinned_values_are_sane():
    assert len(btc.MODEL_SHA256) == 64 and btc.COMMIT in btc.MODEL_URL
    assert btc.MODEL_URL.startswith("https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/")


def test_load_model_reads_only_tensors_and_numbers(tmp_path):
    good = tmp_path / "good.pt"
    random_checkpoint(good)
    model, mean, std = btc.load_model(good)
    assert (mean, std) == (-2.2, 1.7) and not model.training

    class Payload:                                         # anything that is not a tensor or a number
        def __reduce__(self):
            return (print, ("this must never run",))

    bad = tmp_path / "bad.pt"
    random_checkpoint(bad, extra=Payload())
    with pytest.raises(btc.ModelUnavailable, match="couldn't load"):
        btc.load_model(bad)
    wrong = tmp_path / "wrong.pt"
    torch.save({"model": {"nonsense": torch.zeros(1)}, "mean": 0.0, "std": 1.0}, wrong)
    with pytest.raises(btc.ModelUnavailable):
        btc.load_model(wrong)


def test_neural_probs_are_a_distribution_per_frame(tmp_path):
    ckpt = tmp_path / "w.pt"
    random_checkpoint(ckpt)
    t = np.arange(btc.SR * 14) / btc.SR
    y = (0.3 * np.sin(2 * np.pi * 262 * t)).astype(np.float32)
    out = btc.neural_probs(y, btc.SR, model_path=ckpt)
    assert out["probs"].shape == (len(btc.features(y, btc.SR)), btc.N_CLASSES)
    assert out["probs"].shape[0] == 108 + 1 + (btc.SR * 4) // btc.HOP        # a full piece, then the 4 s tail
    assert np.allclose(out["probs"].sum(axis=1), 1.0, atol=1e-4)
    assert out["frame_s"] == pytest.approx(2048 / 22050)
    empty = btc.neural_probs(np.zeros(5, dtype=np.float32), btc.SR, model_path=ckpt)
    assert empty["probs"].shape == (0, btc.N_CLASSES)


# --------------------------------------------------------------------------- #
# blending into recognize()
# --------------------------------------------------------------------------- #

def tone(midi, n):
    f = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(n) / SR
    return sum((0.6 ** (h - 1)) * np.sin(2 * np.pi * f * h * t) for h in range(1, 6))


def progression():
    chords = [([60, 64, 67], 48), ([59, 62, 67], 43), ([60, 64, 69], 45), ([60, 65, 69], 41)]
    n = int(BEAT * SR)
    env = np.exp(-np.arange(n) / (0.35 * SR))
    parts = []
    for notes, bass in chords * 2:
        hit = (sum(tone(m, n) for m in notes) * 0.25 + tone(bass, n) * 0.4) * env
        parts.append(np.tile(hit, 4))
    y = np.concatenate(parts)
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def fake_neural(class_at, total_s, frame_s=btc.FRAME_S):
    """Probabilities that say class_at(t) with certainty."""
    n = int(total_s / frame_s) + 1
    probs = np.zeros((n, btc.N_CLASSES))
    for f in range(n):
        probs[f, class_at(f * frame_s)] = 1.0
    return {"probs": probs, "frame_s": frame_s}


def test_neural_scores_fold_qualities_and_average_over_intervals():
    labels = rec._labels()
    probs = np.zeros((20, btc.N_CLASSES))
    probs[:10, 14 * 2 + 6] = 1.0                       # D:min7 for the first 10 frames
    probs[10:, 14 * 7 + 1] = 0.5                       # G:maj at half weight...
    probs[10:, btc.N_INDEX] = 0.5                      # ...and no-chord
    times = np.array([0.0, 10 * btc.FRAME_S, 20 * btc.FRAME_S])
    s = rec.neural_scores({"probs": probs, "frame_s": btc.FRAME_S}, labels, times)
    assert s.shape == (len(labels) + 1, 2)
    assert s[labels.index((2, "m7")), 0] == pytest.approx(1.0) and s[labels.index((2, "m7")), 1] == 0
    assert s[labels.index((7, "")), 1] == pytest.approx(0.5) and s[-1, 1] == pytest.approx(0.5)
    assert np.allclose(s.sum(axis=0), 1.0)
    # classes that fold together add up: hdim7 and dim both land on "dim"
    p2 = np.zeros((4, btc.N_CLASSES))
    p2[:, 14 * 11 + 10] = 0.25                         # B:dim7
    p2[:, 14 * 11 + 11] = 0.5                          # B:hdim7
    s2 = rec.neural_scores({"probs": p2, "frame_s": 1.0}, labels, np.array([0.0, 4.0]))
    assert s2[labels.index((11, "dim")), 0] == pytest.approx(0.75)
    # an interval wholly beyond the frames, or of zero length, doesn't blow up
    s3 = rec.neural_scores({"probs": p2, "frame_s": 1.0}, labels, np.array([0.0, 4.0, 4.0, 9.0]))
    assert np.isfinite(s3).all()


def test_a_confident_model_outvotes_the_templates():
    y = progression()
    base = recognize(y, SR)
    assert [s["candidates"][0]["label"] for s in base["segments"]][:4] == ["C", "G", "Am", "F"]
    bar = 4 * BEAT
    dm = 14 * 2 + 0                                    # D:min
    neural = fake_neural(lambda t: dm if t < bar else 14 * 7 + 1, 8 * bar)
    mixed = recognize(y, SR, neural=neural)
    first = mixed["segments"][0]["candidates"][0]["label"]
    assert first == "Dm"                               # the model's strong opinion wins the first bar
    assert mixed["segments"][1]["candidates"][0]["label"] in ("G", "G7")
    assert recognize(y, SR, neural=neural, neural_weight=0.0)["segments"][0]["candidates"][0]["label"] == "C"
    # candidates still come from the blended scores: the templates' pick stays on offer
    assert "C" in [c["label"] for c in mixed["segments"][0]["candidates"]]


def test_an_agreeing_model_changes_nothing_important():
    y = progression()
    names = {"C": 1, "G": 14 * 7 + 1, "Am": 14 * 9 + 0, "F": 14 * 5 + 1}
    seq = ["C", "G", "Am", "F"] * 2
    neural = fake_neural(lambda t: names[seq[min(7, int(t // (4 * BEAT)))]], 8 * 4 * BEAT)
    mixed = recognize(y, SR, neural=neural)
    assert [s["candidates"][0]["label"] for s in mixed["segments"]] == seq
    assert mixed["key"] == "C:maj"
    assert all(abs(s["start"] - k * 4 * BEAT) < 0.2 for k, s in enumerate(mixed["segments"]))


def test_a_no_chord_opinion_from_the_model():
    y = progression()
    neural = fake_neural(lambda t: btc.N_INDEX, 8 * 4 * BEAT)
    # a model that hears no chord anywhere, trusted at half and at nine tenths: the whole piece is N.C.
    for weight in (0.5, 0.9):
        out = recognize(y, SR, neural=neural, neural_weight=weight)
        assert [s["candidates"][0]["label"] for s in out["segments"]] == ["N.C."]
    # a model that is only unsure for one bar can't erase the rest
    one_bar = fake_neural(lambda t: btc.N_INDEX if t < 4 * BEAT else 14 * 7 + 1, 8 * 4 * BEAT)
    labels = [s["candidates"][0]["label"] for s in recognize(y, SR, neural=one_bar)["segments"]]
    assert labels[0] == "N.C." and len(labels) > 1


# --------------------------------------------------------------------------- #
# pipeline and options
# --------------------------------------------------------------------------- #

def test_pipeline_uses_the_model_or_falls_back(tmp_path, progression_audio_neural):
    from fastapi.testclient import TestClient

    import soundfile as sf

    from test_chords import fake_separator, fake_transcriber
    from mixpipe.web.server import create_app

    buf = io.BytesIO()
    sf.write(buf, progression_audio_neural, SR, format="WAV")
    calls = []

    def good(mix, sr, progress):
        calls.append(mix.shape)
        progress("working")
        return fake_neural(lambda t: 1, len(mix) / sr)

    def broken(mix, sr, progress):
        raise btc.ModelUnavailable("no network")

    def run(provider, root, **opts):
        client = TestClient(create_app(root, separator=fake_separator, transcriber=fake_transcriber,
                                       neural_provider=provider, run_inline=True))
        sid = client.post("/api/songs", files=[("files", ("a.wav", buf.getvalue(), "audio/wav"))],
                          data={"language": "en", **opts}).json()["id"]
        return client, sid

    client, sid = run(good, tmp_path / "a", recognizer="neural")
    assert client.get(f"/api/songs/{sid}").json()["recognizer"] == "btc" and calls
    stored = json.loads((tmp_path / "a" / sid / "chords.json").read_text())
    assert stored["recognizer"] == "btc" and stored["recognizer_requested"] == "neural"

    client, sid = run(broken, tmp_path / "b", recognizer="neural")
    assert client.get(f"/api/songs/{sid}").json()["recognizer"] == "template"       # fell back
    assert client.get(f"/api/songs/{sid}/status").json()["state"] == "done"

    n_before = len(calls)
    client, sid = run(good, tmp_path / "c")                                          # default: no model
    assert client.get(f"/api/songs/{sid}").json()["recognizer"] == "template" and len(calls) == n_before


@pytest.fixture(scope="module")
def progression_audio_neural():
    return progression()


def test_option_parsing_and_cli(monkeypatch, tmp_path, capsys):
    from mixpipe import cli
    from mixpipe.web.server import parse_options

    assert parse_options({})["recognizer"] == "template"
    assert parse_options({"recognizer": "neural"})["recognizer"] == "neural"
    assert parse_options({"recognizer": "anything else"})["recognizer"] == "template"

    shown = []
    monkeypatch.setattr(btc, "ensure_model", lambda progress=None: (progress or print)("fetching") or tmp_path / "m.pt")
    monkeypatch.setattr(btc, "load_model", lambda path=None, progress=None: shown.append(path))
    assert cli.main(["fetch-model", "btc"]) == 0
    assert "BTC chord model ready" in capsys.readouterr().out and shown == [tmp_path / "m.pt"]

    def fail(progress=None):
        raise btc.ModelUnavailable("offline")

    monkeypatch.setattr(btc, "ensure_model", fail)
    assert cli.main(["fetch-model", "btc"]) == 1
    assert "offline" in capsys.readouterr().err
