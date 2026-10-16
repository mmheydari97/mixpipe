"""Stem splitter: split a song into vocals / guitar / drums / bass / keys /
other, then rebalance them in the browser (mute the vocals to sing along, drop
the guitar to play along, slow it down, loop a section).

The reverse of the mixing pipeline. Shares the song store, link downloads and
web plumbing with the chord finder (songstore, fetch, web/common) and only
imports io_utils and separation besides; heavy dependencies are the chord
finder's (requirements-chords.txt) and are imported lazily.
"""
