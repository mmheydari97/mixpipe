"""Chord finder: upload a song, get lyrics with chord candidates on the words.

Heavy dependencies (librosa, demucs, mlx-whisper, fastapi) live in
requirements-chords.txt and are imported lazily, so the mixing pipeline never
needs them. This package may only import io_utils, separation, pitch,
openrouter, fetch and songstore from the rest of mixpipe, which keeps it easy
to split out later.
"""
