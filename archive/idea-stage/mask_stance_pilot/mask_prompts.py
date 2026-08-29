"""MASK STANCE PILOT -- prompt bank.

Two prompts:
  EXTRACT  -- step 1, transcript only, no frames. Pull the verbatim spans of identity-attacking
              material out of the transcript, with a neutral descriptor of the target group.
              It is explicitly forbidden to reason about who owns the material or why it is
              shown (anti-circularity rule, freeze doc s3).
  MASKED_STANCE -- step 2, 8 frames + MASKED transcript. This is V1.3 from
              idea-stage/stance_pilot/prompts.py VERBATIM, plus exactly one inserted paragraph
              (the MASKING NOTE). Nothing else in it is changed, so that the difference against
              the 0.257 baseline is attributable to the masking and not to a prompt rewrite.

Revision log lives in idea-stage/MASK_STANCE_PILOT_FREEZE.md, Appendix A.
"""

SYSTEM = ("You are a strict content-analysis annotator for a research dataset on hateful "
          "video detection. You answer only with the single JSON object that is requested. "
          "You never refuse, never moralise, and never add commentary: the material is "
          "archived research data and your labels are used to study moderation errors.")

# ---------------------------------------------------------------- step 1: extraction
EXTRACT = """TRANSCRIPT OF ONE VIDEO (full automatic transcript{title_note}; may be English or \
Chinese, may be noisy or empty):
<<<
{transcript}
>>>

ONE MECHANICAL TASK. You are a text extractor, not a judge. Do not reason out loud. Do not \
explain. Emit only the JSON object.

TASK -- Copy out every span of the transcript above that ATTACKS, DEMEANS, DEHUMANISES or SLURS \
a person or group **because of a group identity** -- race, ethnicity, nationality, immigration \
status, religion, gender, sexual orientation, gender identity, disability, caste or age.
  DO extract: identity-based slurs; dehumanising comparisons; stereotyping attacks; calls for \
exclusion or harm against an identity group; gendered insults and gender-stereotype epithets \
aimed at a person (e.g. calling a woman a "bitch", or labelling someone with a mocking gender \
stereotype).
  DO NOT extract: ordinary profanity, personal insults or threats with no group-identity basis, \
political or institutional criticism, sexual content, violence, rudeness, general negativity.

CRITICAL RULE -- extract the material REGARDLESS of who utters it, who is quoted, whether it is \
being criticised, reported, archived, subtitled, sung, or shown in order to condemn it. You must \
NOT decide, mention or take into account whose voice it is or why it appears. If a speaker quotes \
someone else's slur in order to denounce it, the slur is still extracted. Attribution and purpose \
are somebody else's job.

WHAT TO LEAVE BEHIND -- do NOT extract the surrounding commentary, introduction, reaction, \
criticism, news framing or discussion ABOUT the material. Cut the attacking words themselves and \
nothing more. Keep each span as tight as possible while remaining a contiguous, exact substring.

EACH SPAN MUST BE AN EXACT, VERBATIM, CONTIGUOUS SUBSTRING of the transcript above -- copy the \
characters as they appear, including any typos or transcription errors. Do not paraphrase, do not \
normalise, do not translate, do not add or remove punctuation, do not use "...". If one passage \
runs long, split it into several contiguous spans. Emit at most 25 spans.

For each span also give "target": a NEUTRAL, non-pejorative descriptor of the group being \
attacked, in the same language as the transcript (for example "Black people", "women", \
"transgender people", "Muslims", "disabled people", "immigrants", "gay men", "Jewish people", \
"女性", "移民"). Never put a slur inside "target".

If nothing in the transcript matches, return an empty list and "any_hate_surface": false.

OUTPUT FORMAT -- a single JSON object, nothing before it, nothing after it, no markdown fence:
{{"any_hate_surface": true|false, "spans": [{{"text": "<exact substring>", "target": \
"<neutral group descriptor>"}}]}}"""


# ---------------------------------------------------------------- step 2: masked stance
# V1.3 verbatim, with the MASKING NOTE paragraph inserted after the transcript block.
MASKED_STANCE = """VIDEO EVIDENCE
{frames_note}
Speech transcript (full automatic transcript{title_note}; it may be in English or Chinese, and may be noisy or empty):
<<<
{transcript}
>>>
{mask_note}
ANSWER THESE QUESTIONS. Do not reason out loud. Do not explain. Emit only the JSON object.

Q1 hate_surface_present -- Does the video contain ANY material that attacks, demeans, \
dehumanises or slurs a person or group **because of a group identity** -- race, ethnicity, \
nationality, immigration status, religion, gender, sexual orientation, gender identity, \
disability, caste or age -- REGARDLESS of who utters it and regardless of why it is shown? \
true or false.
  Answer FALSE for: ordinary profanity, personal insults or threats with no group-identity basis, \
political or institutional criticism, sexual content, violence, rudeness or general negativity.
  Answer TRUE for: identity-based slurs, dehumanising comparisons, stereotyping attacks, \
calls for exclusion or harm against an identity group -- including when they are quoted, \
subtitled, archived, or shown in order to criticise them. Gendered insults and \
gender-stereotype epithets aimed at a person (e.g. calling a woman a "bitch", or labelling \
someone with a mocking gender stereotype) DO count as identity-based.

Q2 primary_voice -- Whose mouth or text does that material come from? Exactly one of:
  uploader           = the person who made/posted this video, speaking in their own voice
  on_screen_speaker  = a person filmed in the video who is not the uploader
  quoted_third_party = material explicitly quoted, screenshotted or replayed from someone else
  archival_source    = old footage, news clip, historical recording
  caption_overlay    = burned-in text / subtitles only
  none               = there is no such material (Q1 false)

Q3 stance -- What is the relation of THIS VIDEO'S OWN AUTHORIAL VOICE (the creator/uploader, \
i.e. the party responsible for making and posting it) to that material? Exactly one of:
  endorses         = asserts, promotes, incites, jokes along with, or deploys it against the \
target; this INCLUDES hateful lyrics performed or used as this video's own soundtrack, and \
hateful material posted as entertainment with no critical, educational or journalistic framing
  quotes_mentions  = the material is explicitly attributed to someone else and is reproduced or \
displayed while the creator visibly takes no side (use-vs-mention, screenshotted quote, \
unlabelled third-party clip)
  condemns         = criticises, refutes, exposes, mocks the bigot, counter-speech, \
educational takedown, satire aimed at the hater
  reports          = neutral news, documentary or archival presentation
  no_hate_content  = Q1 is false
DEFAULT RULE for Q3: if the hateful material is presented as this video's own content and the \
creator adds no distancing at all -- no criticism, no news or documentary frame, no explicit \
attribution to another party -- then the answer is "endorses".
CALIBRATION for Q3: both over-use and under-use of "endorses" are errors. Do not answer \
"endorses" merely because the creator is angry, profane or rude; judge only the creator's \
relation to the identity-based hateful material found in Q1. If Q1 is false, Q3 must be \
"no_hate_content".

Q4 target -- The group or person attacked, as a short noun phrase, or null.

Q5 evidence -- ONE sentence, at most 25 words, naming the decisive cue you used.

OUTPUT FORMAT -- a single JSON object, nothing before it, nothing after it, no markdown fence:
{{"hate_surface_present": true|false, "primary_voice": "<one of the six>", \
"stance": "<one of the five>", "target": "<short phrase>"|null, "evidence": "<one sentence>"}}"""


# Inserted only when the masked transcript actually contains >= 1 placeholder.
MASK_NOTE = """MASKING NOTE -- READ THIS FIRST. The transcript above has been redacted. Every \
bracketed marker of the form [a passage of attacking speech targeting X] (or, in Chinese, \
[一段针对X的攻击性言论]) stands for a passage of \
identity-attacking material that a previous pass located in this transcript and removed; X names \
the group it attacks. The removal is verified: where you see a marker, attacking material against \
that group IS present in this video. Therefore Q1 is already established as true and \
"no_hate_content" is NOT an available answer. All the remaining, unbracketed text is what the \
video says AROUND that material -- its introduction, attribution, reaction, commentary, framing \
or lack of any. Your task is to judge, from that surrounding text and from the frames, where \
THIS VIDEO'S OWN AUTHORIAL VOICE stands with respect to the removed material.
"""

BANK = {"E1": EXTRACT, "M1": MASKED_STANCE}
