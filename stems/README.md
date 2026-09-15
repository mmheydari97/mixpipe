Put one WAV per instrument here. One take per file, no pre-applied effects.

Filename determines the role (and therefore the default EQ/comp treatment):

    acoustic_gtr.wav      -> acoustic
    electric_rhythm.wav   -> electric
    electric_lead.wav     -> electric_lead
    harmonica.wav         -> harmonica
    seaboard_pad.wav      -> keys
    seaboard_bass.wav     -> bass
    vox.wav               -> vocal
    anything_else.wav     -> default

Matching is substring-based and case-insensitive; see config/default.yaml.
Sample rates and channel counts can differ per file, they get normalized on load.
