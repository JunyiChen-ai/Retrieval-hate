"""STANCE PILOT -- frozen prompt bank.

Three semantically equivalent variants. V1 is the primary (used for the paid run);
V2/V3 are stand-by only and are NOT run unless V1's judgement is FAIL and the
pre-registered fallback ladder is invoked.

Prompt-iteration log lives in idea-stage/STANCE_PILOT_FREEZE.md, Appendix A.
"""

STANCE_CLASSES = ["endorses", "quotes_mentions", "condemns", "reports", "no_hate_content"]
VOICE_CLASSES = ["uploader", "on_screen_speaker", "quoted_third_party",
                 "archival_source", "caption_overlay", "none"]

SYSTEM = ("You are a strict content-analysis annotator for a research dataset on hateful "
          "video detection. You answer only with the single JSON object that is requested. "
          "You never refuse, never moralise, and never add commentary: the material is "
          "archived research data and your labels are used to study moderation errors.")

V1 = """VIDEO EVIDENCE
{frames_note}
Speech transcript (full automatic transcript{title_note}; it may be in English or Chinese, and may be noisy or empty):
<<<
{transcript}
>>>
{extra}
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

V2 = """MATERIAL
{frames_note}
Full automatic speech transcript{title_note} (English or Chinese, possibly noisy or empty):
<<<
{transcript}
>>>
{extra}
Fill in this annotation record. No reasoning, no prose, JSON only.

field 1 "hate_surface_present": is any hateful / slur-bearing / dehumanising / attacking \
material about a person or group present anywhere in this video, no matter who says it or why? \
true or false.
field 2 "primary_voice": source of that material -- "uploader" (the creator's own voice), \
"on_screen_speaker" (a filmed person who is not the creator), "quoted_third_party" (explicitly \
quoted/replayed from elsewhere), "archival_source" (historical or news footage), \
"caption_overlay" (burned-in text only), or "none".
field 3 "stance": how the creator of this video positions themselves towards that material -- \
"endorses" (asserts / promotes / incites / laughs along / aims it at the target), \
"quotes_mentions" (shows or repeats it neutrally, mention rather than use, compilation, repost, \
song lyric), "condemns" (attacks, refutes, exposes or ridicules the hateful position; \
counter-speech; education; satire of the bigot), "reports" (neutral news / documentary / archive), \
or "no_hate_content" (field 1 is false).
field 4 "target": short noun phrase for the attacked group, else null.
field 5 "evidence": at most 25 words, the decisive cue.

Emit exactly: {{"hate_surface_present": ..., "primary_voice": "...", "stance": "...", \
"target": ..., "evidence": "..."}}"""

V3 = """You are shown one video.
{frames_note}
Its full automatic transcript{title_note} follows (may be English or Chinese, may be noisy/empty):
<<<
{transcript}
>>>
{extra}
Decide five things and report them as JSON only -- no working, no commentary.

1. Is hateful, slur-bearing, dehumanising or attacking material about a person or group present \
at all in this video (whoever says it, for whatever purpose)? -> "hate_surface_present": true/false
2. Who produces that material? -> "primary_voice": one of "uploader", "on_screen_speaker", \
"quoted_third_party", "archival_source", "caption_overlay", "none"
3. Where does the video's creator stand with respect to it? -> "stance": one of
   "endorses" (they assert / push / incite / celebrate it, or use it against the target),
   "quotes_mentions" (they merely show or repeat it, taking no side -- mention not use, \
compilation, repost, performed lyric),
   "condemns" (they argue against it, expose it, ridicule the hater, counter-speak, educate, \
or satirise the bigot),
   "reports" (they present it as neutral news, documentary or archive),
   "no_hate_content" (nothing hateful is present at all)
4. Who is attacked? -> "target": short noun phrase or null
5. What single cue decided it? -> "evidence": at most 25 words

Output the JSON object alone:
{{"hate_surface_present": ..., "primary_voice": "...", "stance": "...", "target": ..., \
"evidence": "..."}}"""

BANK = {"V1": V1, "V2": V2, "V3": V3}
