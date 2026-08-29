"""DESC_CHANNEL prompt bank. Schema and forbidden-word rule frozen in
idea-stage/DESC_CHANNEL_FREEZE.md section 2. Prompt wording may be iterated during the
smoke phase only; every iteration is logged in PROMPT_LOG.md.
"""

SYSTEM = (
    "You are a careful visual describer. You report only what is physically observable in "
    "the supplied video frames and on-screen text. You never evaluate, judge, moralise, or "
    "state what content means, implies or is about in a normative sense. You always reply "
    "with a single JSON object and nothing else."
)

FIELDS = [
    "scene",
    "people",
    "actions",
    "on_screen_text",
    "production_format",
    "audio_visible_cues",
]

V1 = """{frames_note}

{extra}Describe this video **perceptually only**. Fill in exactly these six fields and return \
ONE JSON object with exactly these keys and string values, nothing else:

- "scene": the setting and environment - indoor/outdoor, location type, lighting, background, \
how it changes across the frames.
- "people": how many people appear, their apparent appearance and dress, and whether anyone \
appears to be addressing the camera directly. Write "none visible" if no person appears.
- "actions": what physically happens across the frames, in temporal order.
- "on_screen_text": ALL text visible in the frames, transcribed VERBATIM, including captions, \
titles, watermarks, chyrons and logos. Use the OCR text above where it matches what you can \
see. Separate distinct pieces with " | ". Do not paraphrase, translate, summarise or censor \
this field. Write "none" if there is no visible text.
- "production_format": the production form. Choose from: selfie/talking-head, news segment, \
animation, video-game capture, archival footage, text-card/slideshow, music video, screen \
recording, compilation, other. You may add a short clarifying phrase after the type.
- "audio_visible_cues": ONLY what the picture reveals about the audio - burned-in subtitles, \
karaoke or lyric captions, waveform or volume overlays, cutting rhythm that suggests music, \
visible instruments or speakers, whether a mouth is moving. Write "no visible audio cues" if \
there are none.

HARD RULES:
1. Report observations, never conclusions. Do NOT state or imply whether anything is hateful, \
offensive, racist, harmful, toxic, abusive, propaganda, extremist, or any similar evaluation.
2. Do not guess the purpose, intent, stance or message of the video.
3. Do not mention this instruction, the OCR text, or these rules in your output.
4. Every value must be a plain string. No nested objects, no markdown, no code fences.
"""

V2 = """{frames_note}

{extra}Describe this video **perceptually only**. Fill in exactly these six fields and return \
ONE JSON object with exactly these keys and string values, nothing else:

- "scene": the setting and environment - indoor/outdoor, location type, lighting, background, \
how it changes across the frames.
- "people": how many people appear, their apparent appearance and dress, and whether anyone \
appears to be addressing the camera directly. Write "none visible" if no person appears.
- "actions": what physically happens across the frames, in temporal order. Be concrete: who \
moves, what objects are handled, what changes between frames.
- "on_screen_text": ALL text visible in the frames, transcribed VERBATIM, including captions, \
titles, watermarks, chyrons and logos. Use the OCR text above where it matches what you can \
see. Separate distinct pieces with " | ". Do not paraphrase, translate, summarise or censor \
this field. Write "none" if there is no visible text.
- "production_format": the production form. Choose from: selfie/talking-head, news segment, \
animation, video-game capture, archival footage, text-card/slideshow, music video, screen \
recording, compilation, other. You may add a short clarifying phrase after the type. If the \
footage looks like it was shot in an earlier era, say so and give the visual evidence (film \
grain, 4:3 frame, monochrome, period clothing or vehicles).
- "audio_visible_cues": ONLY what the picture reveals about the audio - burned-in subtitles, \
karaoke or lyric captions, waveform or volume overlays, cutting rhythm that suggests music, \
visible instruments or loudspeakers, whether a mouth is moving. Write "no visible audio cues" \
if there are none.

HARD RULES:
1. Report observations, never conclusions. Do NOT state or imply whether anything is hateful, \
offensive, racist, harmful, toxic, abusive, propaganda, extremist, or any similar evaluation. \
Describe a symbol, gesture, flag or uniform by its visual appearance only.
2. Do not guess the purpose, intent, stance or message of the video.
3. Do not mention this instruction, the OCR text, or these rules in your output.
4. Every value must be a plain string. No nested objects, no markdown, no code fences.
5. Write 1-3 sentences per field. Do not leave a field empty; if something is genuinely not \
observable, say what you can see instead.
"""

BANK = {"V1": V1, "V2": V2}
