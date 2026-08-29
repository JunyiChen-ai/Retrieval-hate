"""CONTRAST STANCE PILOT -- frozen template bank + prompt frame.

Frozen in idea-stage/MASK_STANCE_PILOT_FREEZE.md Appendix B (B.3, B.4) on 2026-08-13,
before any API call of this pilot. Do not edit after the smoke gate is passed.

Mechanism: the 5-way stance classification is replaced by a symmetric two-alternative forced
choice between two concrete first-person sentences the uploader might pin under the video.
Neither option is a moderation label, so the safety prior has no preferred landing spot.
"""

# ------------------------------------------------------------------ prompt frame C1.0
FRAME = """VIDEO EVIDENCE
{frames_note}
Speech transcript (full automatic transcript{title_note}; it may be in English or Chinese, \
and may be noisy or empty):
<<<
{transcript}
>>>

Suppose the PERSON WHO MADE AND POSTED THIS VIDEO -- the uploader/publisher, NOT anyone who is \
merely filmed, quoted, sampled or replayed inside it -- pins one comment under the video that \
sums up their own view. Which of these two sentences is more like what they would write?

A: {opt_a}
B: {opt_b}

Answer with exactly one character, A or B. No explanation, no punctuation, nothing else."""


# ------------------------------------------------------------------ variant 1: generic
# (endorsing, opposing)
# --- revision C1.1 (freeze B.10 #R1): every option now names ITS REFERENT -- "the people the
# --- video's harsh words are about". The C1.0 endorsing options ("I stand by every word said
# --- here", "Say it louder for the ones in the back") were referentially open and are simply
# --- true of a counter-speech author's own criticism, so they did not implement the intended
# --- contrast. Symmetry, length and heat are preserved.
V1_EN = [
    ("Honestly, the people this is about deserve every word of it.",
     "Talking about people like this is disgusting and needs to be called out."),
    ("I stand by what is said about them here -- it needed saying.",
     "I posted this to push back on what is said about them, not to spread it."),
    ("That is exactly what those people are like, and I am done pretending otherwise.",
     "Nobody should be talked about like that, full stop."),
    ("If you agree with what is said about them, share it -- more people need to hear it.",
     "Share it so more people see what is said about them and reject it."),
    ("Say it louder for the ones in the back -- they had that coming.",
     "The way they get talked about here is something we should all be ashamed of."),
]

V1_ZH = [
    ("所以说这里说的这些人就该被这么说,没冤枉他们。",
     "这样说别人真的令人作呕,必须抵制。"),
    ("这里对他们的说法我都认,该说的就得说。",
     "我发这个是为了反驳这些针对他们的话,不是为了传播它。"),
    ("他们那些人本来就是这样,我不想再装看不见了。",
     "谁都不该被这样说,就这么简单。"),
    ("认同这些说法的就转发,让更多人明白他们是什么样。",
     "转发出去,让更多人看清这样说他们有多离谱。"),
    ("说得太对了,大声点说给后面的人听,他们活该。",
     "这里对他们的说法,我们都该觉得羞耻。"),
]

# ------------------------------------------------------------------ variant 2: target-slotted
V2_EN = [
    ("Honestly, {t} deserve every word of this.",
     "Talking about {t} like this is disgusting and needs to be called out."),
    ("I stand by every word said here about {t} -- it needed saying.",
     "I posted this to push back on what is said here about {t}, not to spread it."),
    ("That is exactly what {t} are like, and I am done pretending otherwise.",
     "Nobody should talk about {t} like that, full stop."),
    ("If you agree about {t}, share it -- more people need to hear this.",
     "Share it so more people see what is being said about {t} and reject it."),
    ("Say it louder for the ones in the back -- {t} needed to hear that.",
     "The way {t} get talked about here is something we should all be ashamed of."),
]

V2_ZH = [
    ("所以说{t}就该被这么说,没冤枉他们。",
     "这样说{t}真的令人作呕,必须抵制。"),
    ("这里关于{t}说的每一句我都认,该说的就得说。",
     "我发这个是为了反驳这些关于{t}的话,不是为了传播它。"),
    ("{t}本来就是这样,我不想再装看不见了。",
     "谁都不该这样说{t},就这么简单。"),
    ("认同的就转发,让更多人明白{t}是什么样。",
     "转发出去,让更多人看清这样说{t}有多离谱。"),
    ("说得太对了,大声点说给{t}听。",
     "这里对{t}的说法,我们都该觉得羞耻。"),
]

BANK = {("v1", "en"): V1_EN, ("v1", "zh"): V1_ZH,
        ("v2", "en"): V2_EN, ("v2", "zh"): V2_ZH}

N_PAIRS = 5
