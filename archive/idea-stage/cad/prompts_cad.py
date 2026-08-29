"""CAD prompt bank. Wording may be iterated during the smoke phase only; the version
placed in idea-stage/CAD_FREEZE.md section 2 is the one used for the frozen run.

One-way by construction: the instruction only ever asks for the REMOVAL of identity
attacks from an existing transcript. No prompt in this bank asks for hateful content.
"""

SYSTEM = (
    "You are a careful text editor working on an authorised content-moderation research "
    "dataset. Your job is to produce a MINIMALLY edited version of a video transcript in "
    "which content that attacks people for their group identity has been removed or made "
    "neutral, while everything else in the transcript is preserved exactly. You never add "
    "any new insult, attack or slur. You always reply with a single JSON object and "
    "nothing else."
)

V1 = """Below is the verbatim automatic-speech-recognition transcript of a short video.

Rewrite it so that it no longer attacks, demeans or dehumanises any person or group on \
the basis of a protected characteristic (race, ethnicity, national origin, religion, \
gender, sexual orientation, disability, immigration status, caste, or similar).

Make the SMALLEST edit that achieves this:
- Delete or neutralise ONLY the spans that carry the attack: slurs, dehumanising \
comparisons, calls for violence or exclusion, group-level accusations and stereotypes, \
and celebrations of harm done to a group.
- KEEP everything else exactly as it is: the topic, the speaker's other opinions, the \
ordering of sentences, the sentence structure, the register, the filler words and the \
ASR noise ("uh", "you know", repeated words, mis-transcriptions, missing punctuation).
- Do NOT summarise. Do NOT clean up grammar or spelling. Do NOT translate. Do NOT add \
commentary, apology, disclaimer or counter-speech. Do NOT mention that anything was edited.
- Do NOT introduce any new attack on any person or group.
- Keep the length close to the original: aim for 70-100% of the original length.
- If the ONLY content of a sentence is the attack, delete that sentence rather than \
replacing it with a comment.
- If the transcript contains no such attack, return it unchanged with "n_edits": 0.

Return ONE JSON object with exactly these two keys and nothing else:
{{"rewritten": "<the full rewritten transcript, as a single string>", "n_edits": \
<integer: how many distinct spans you deleted or changed>}}

TRANSCRIPT:
<<<
{transcript}
>>>
"""

V2 = """Below is the verbatim automatic-speech-recognition (ASR) transcript of a short \
video. The ASR is noisy: words are often mis-spelled, run together or garbled, and slurs \
frequently appear in a corrupted spelling.

Rewrite the transcript so that it no longer attacks, demeans or dehumanises any person or \
group on the basis of a protected characteristic (race, ethnicity, national origin, \
religion, gender, sexual orientation, disability, immigration status, caste, or similar).

Remove or neutralise ALL of the following, including when the ASR has garbled the wording:
- slurs and epithets aimed at a group, in any spelling;
- dehumanising comparisons (calling a group animals, vermin, disease, subhuman);
- calls for violence, harm, deportation or exclusion, and demands that a group "go back" \
somewhere or leave;
- group-level accusations and stereotypes ("they are lazy / criminal / parasites", \
conspiracy claims that a group secretly controls or harms others);
- celebration or approval of harm done to a group;
- praise of movements or figures whose point is the supremacy of one group over another.

Make the SMALLEST edit that achieves this:
- Change ONLY the spans listed above. Replace a slur with a plain neutral referent \
("people", "them", "he") or delete the span.
- KEEP everything else exactly as it is: the topic, the speaker's other opinions, the \
ordering of sentences, the sentence structure, the register, the filler words, the emoji \
and the ASR noise (repeated words, missing spaces, mis-transcriptions, broken punctuation).
- Do NOT summarise. Do NOT clean up grammar or spelling elsewhere. Do NOT translate. \
Do NOT add commentary, apology, disclaimer or counter-speech. Do NOT mention that anything \
was edited.
- Do NOT introduce any new attack on any person or group.
- Keep the length close to the original: aim for 70-100% of the original length.
- If the ONLY content of a sentence is the attack, delete that sentence rather than \
replacing it with a comment.
- If the transcript is pure noise with no attack in it, return it unchanged with \
"n_edits": 0.

Return ONE JSON object with exactly these two keys and nothing else. Do NOT copy the \
<<< and >>> markers into your answer:
{{"rewritten": "<the full rewritten transcript, as a single string>", "n_edits": \
<integer: how many distinct spans you deleted or changed>}}

TRANSCRIPT:
<<<
{transcript}
>>>
"""

BANK = {"V1": V1, "V2": V2}
