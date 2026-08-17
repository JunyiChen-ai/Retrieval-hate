"""
r7_ocrprov.rules -- FROZEN rule-based source/provenance indicators from HateMM on-screen OCR text.

Six fixed feature families (order is part of the freeze):
    stock_watermark, news_chyron, date_stamp, ui_text, handle_watermark, copyright

Design notes (all decisions were made LABEL-BLIND, from frequency evidence over the
851 train+val videos in data/OCR/HateMM/ocr_video.jsonl; see vocab_recon.json):

  * Normalisation: text is uppercased and all whitespace runs (incl. the newlines that
    separate OCR windows) are collapsed to a single space. Matching is therefore
    case-insensitive and newline-insensitive by construction.
  * Counting: `extract` returns, per family, the number of regex matches over the WHOLE
    normalised string. Because window texts repeat, a persistent watermark yields a large
    count while an incidental mention yields 1-2. Both signals are preserved; use
    `extract_binary` if only presence is wanted.
  * Alphanumeric terms are matched with (?<![A-Z0-9]) / (?![A-Z0-9]) guards so that e.g.
    "CNN" does not fire inside "CNNX". Phrase terms use the same guards at both ends.
  * Terms deliberately EXCLUDED after inspecting corpus contexts (documented false positives):
      - "MINDS" (platform) -- all 11 videos were the English word ("feeble minds").
      - bare "LIKE" -- ubiquitous English word; only phrase forms are kept.
      - bare "RT" -- OCR noise fragments; only "RT.COM" / "RT NEWS" are kept.
      - bare "ARCHIVE" -- almost always archive.org URLs, not AP Archive.
      - bare "FOOTAGE", "PREVIEW", "DEMO", "HOME", "SEARCH", "SUPPORT", "FOLLOW" -- generic.
      - "(C)" without a following year -- ambiguous OCR/list marker.
  * OCR-noise handling that IS kept: "©" immediately followed by letters is, in this
    corpus, a mis-read "@handle" -- it is routed to handle_watermark, NOT to copyright.
    Copyright only fires on "©" followed by a year/space, "COPYRIGHT", "(C) 20xx",
    "ALL RIGHTS RESERVED" / "RIGHTS RESERVED" (incl. the OCR-run-together variants).

No I/O, no randomness, deterministic.
"""

import re

FEATURE_NAMES = [
    "stock_watermark",
    "news_chyron",
    "date_stamp",
    "ui_text",
    "handle_watermark",
    "copyright",
]

_WS = re.compile(r"\s+")


def normalize(text):
    """Uppercase + collapse every whitespace run (incl. OCR window newlines) to one space."""
    if not text:
        return ""
    return _WS.sub(" ", str(text)).strip().upper()


def _terms_to_pattern(terms):
    """Alternation of literal terms with alphanumeric boundary guards, longest-first."""
    ordered = sorted(set(terms), key=lambda s: (-len(s), s))
    body = "|".join(re.escape(t) for t in ordered)
    return re.compile(r"(?<![A-Z0-9])(?:" + body + r")(?![A-Z0-9])")


# --------------------------------------------------------------------------------------
# F1  stock_watermark -- stock-footage / photo-agency watermarks and attribution lines.
# Corpus-confirmed hits: ALAMY, GETTY / GETTY IMAGES, DREAMSTIME, 123RF, REUTERS, AFP,
# ASSOCIATED PRESS, IMAGO, WENN, "STOCK PHOTO", "SUPPLIED BY", "COURTESY", "SOURCE(S):",
# "VISUALS:", "CREDIT". The remaining agency names are kept as frozen vocabulary
# (zero-hit in train+val, retained for coverage of the motivating GLOBALIMAGEWORKS case
# and for held-out generalisation).
# --------------------------------------------------------------------------------------
_STOCK_TERMS = [
    # agencies / marketplaces
    "GLOBALIMAGEWORKS", "GLOBAL IMAGE WORKS", "GIW", "GIWCUSTOM",
    "SHUTTERSTOCK", "GETTY", "GETTY IMAGES", "ISTOCK", "ISTOCKPHOTO",
    "ALAMY", "ALUMY", "POND5", "STORYBLOCKS", "VIDEOBLOCKS", "DREAMSTIME",
    "DEPOSITPHOTOS", "123RF", "BIGSTOCK", "ADOBE STOCK", "ENVATO", "VIDEOHIVE",
    "PEXELS", "PIXABAY", "UNSPLASH", "MOTIONELEMENTS", "FILMSUPPLY", "ARTGRID",
    "CRITICALPAST", "BRITISH PATHE", "IMAGO", "IMAGO IMAGES", "WENN",
    "SPLASH NEWS", "BACKGRID", "ZUMA PRESS", "NEWSFLARE", "STORYFUL",
    # newswire / archive suppliers
    "REUTERS", "AFP", "ASSOCIATED PRESS", "AP ARCHIVE", "AP PHOTO", "VIA AP",
    "EPA-EFE", "ANADOLU",
    # attribution / licensing wording
    "STOCK PHOTO", "STOCK PHOTOS", "STOCK VIDEO", "STOCK IMAGE", "STOCK FOOTAGE",
    "ARCHIVE FOOTAGE", "FILE PHOTO", "ROYALTY FREE", "ROYALTY-FREE",
    "SUPPLIED BY", "COURTESY", "COURTESY OF", "SOURCE:", "SOURCES:",
    "VISUALS:", "CREDIT:", "PHOTO CREDIT", "IMAGE CREDIT", "VIDEO CREDIT",
    "NON-EXCLUSIVE", "LICENSED BY", "TCR",
]
# "CREDIT" without colon is kept separately so "CREDIT CARD" does not fire.
_STOCK_RE = _terms_to_pattern(_STOCK_TERMS)

# --------------------------------------------------------------------------------------
# F2  news_chyron -- broadcast station identifiers + lower-third / ticker wording.
# Corpus df over 851: LIVE 70, NEWS 61, CNN 25, REPORT 18, FOX 15, TONIGHT 14,
# BREAKING 12, ABC 10, BREAKING NEWS 9, FOX NEWS 6, BBC 6, SPORTS 6, CBS 5,
# EXCLUSIVE 5, DEVELOPING 4, WEATHER 4, INFOWARS 4, NBC 4, CGTN 3, DW 3, ITV 3,
# CNBC 3, HEADLINES 3, MSNBC 3, SKY NEWS 2, PBS 2.
# Bare FOX/ABC/NBC/CBS are retained (the brief fixes them and most corpus contexts are
# genuine chyrons, e.g. "NBC ABC ...", "CBS THIS MORNING"); a minority are false
# positives ("STAR FOX 64", "ABC FORMAT OPTIONS"). Recorded, not silently dropped.
# --------------------------------------------------------------------------------------
_NEWS_TERMS = [
    # networks / channels
    "CNN", "CNN.COM", "HLN", "FOX", "FOX NEWS", "FOXNEWS", "FOX BUSINESS",
    "MSNBC", "NBC", "NBC NEWS", "CNBC", "ABC", "ABC NEWS", "ABCNEWS",
    "CBS", "CBS NEWS", "CBSN", "BBC", "BBC NEWS", "SKY NEWS", "SKYNEWS",
    "AL JAZEERA", "ALJAZEERA", "AL-JAZEERA", "RT NEWS", "RT.COM",
    "CGTN", "DW", "ITV", "CHANNEL 4", "PBS", "NPR", "FRANCE 24", "EURONEWS",
    "TRT WORLD", "PRESS TV", "NDTV", "INDIA TODAY", "TIMES NOW", "ZEE NEWS",
    "AAJ TAK", "WION", "NEWS18", "REPUBLIC TV", "GLOBAL NEWS", "VICE NEWS",
    "INFOWARS", "HUFFPOST", "HUFFINGTON POST", "NEWSMAX", "OANN", "C-SPAN",
    # lower-third / ticker wording
    "NEWS", "NEWS LIVE", "LIVE NEWS", "BREAKING", "BREAKING NEWS", "LIVE",
    "WATCH LIVE", "LIVE NOW", "NEWSROOM", "HEADLINES", "DEVELOPING",
    "DEVELOPING STORY", "EXCLUSIVE", "REPORT", "REPORTS", "REPORTING",
    "CORRESPONDENT", "ANCHOR", "TONIGHT", "THIS MORNING", "SPORTS", "WEATHER",
    "UPDATE", "SPECIAL REPORT", "TOP STORIES", "NOW", "AP",
]
# "NOW" and "AP" are too generic on their own -- removed below to keep precision.
_NEWS_TERMS = [t for t in _NEWS_TERMS if t not in ("NOW", "AP")]
# Station-with-channel-number chyrons (FOX10, FOX45, ABC7, NBC4, CBS2, WXYZ-style).
_NEWS_STATION_RE = re.compile(r"(?<![A-Z0-9])(?:FOX|ABC|NBC|CBS|CW|KTLA|WGN)\d{1,2}(?![A-Z0-9])")
_NEWS_RE = _terms_to_pattern(_NEWS_TERMS)

# --------------------------------------------------------------------------------------
# F3  date_stamp -- date / timestamp patterns.
# Corpus df over 851: clock 125 (.147), ampm 62 (.073), month-day-year 53 (.062),
# numeric d/m/y 25 (.029).
# --------------------------------------------------------------------------------------
_DATE_PATTERNS = [
    re.compile(r"(?<!\d)\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}(?!\d)"),          # 7/26/2020
    re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,2})?(?![\d:])"),  # 12:07 / 01:04:29.10
    re.compile(
        r"(?<![A-Z0-9])(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)"
        r"[A-Z]*\.?\s?\d{1,2}(?:ST|ND|RD|TH)?,?\s?\d{0,4}(?![A-Z0-9])"
    ),                                                                       # NOV.15-2014, AUG 3
    re.compile(r"(?<![A-Z0-9])\d{1,2}(?::\d{2})?\s?[AP]M(?![A-Z0-9])"),      # 10:08 AM, 7PM
]

# --------------------------------------------------------------------------------------
# F4  ui_text -- platform UI chrome / capture artefacts / monetisation calls-to-action.
# Corpus df over 851: SUBSCRIBE 32, YOUTUBE 28, TWITTER 21, VIEWS 16, BITCHUTE 16,
# SHARE 16, COMMENTS 15, FACEBOOK 15, COMMENT 13, DONATE 9, TIKTOK 8, INSTAGRAM 6,
# DISCORD 5, TELEGRAM 5, SETTINGS 5, NOTIFICATIONS 5, SIGN IN 5, PATREON 4, TRENDING 4.
# Platform names are included because in this corpus they are screen-capture chrome
# (channel headers, share bars), i.e. provenance evidence.
# --------------------------------------------------------------------------------------
_UI_TERMS = [
    # engagement chrome
    "SUBSCRIBE", "SUBSCRIBED", "SUBSCRIBERS", "SUBSCRIPTIONS", "SUBSCRIBE NOW",
    "LIKE AND SUBSCRIBE", "LIKE, COMMENT", "LIKE COMMENT", "LIKES",
    "SHARE", "COMMENT", "COMMENTS", "VIEWS", "WATCH LATER", "ADD TO QUEUE",
    "ADD TO PLAYLIST", "PLAYLIST", "AUTOPLAY", "SHOW MORE", "SIGN IN",
    "TRENDING", "SETTINGS", "UPLOADED", "NOTIFICATION", "NOTIFICATIONS",
    "BELL ICON", "THE BELL", "HIT THE BELL", "RING THE BELL", "SMASH THAT",
    "THUMBS UP", "CLICK THE LINK", "LINK IN THE", "LINK IN BIO", "LINK BELOW",
    "FULL SCREEN", "LIVE CHAT", "SUPERCHAT", "STREAMED LIVE",
    # monetisation
    "PATREON", "PAYPAL", "SUBSCRIBESTAR", "CASHAPP", "CASH APP", "VENMO",
    "GOFUNDME", "BUY ME A COFFEE", "MERCH", "DONATE", "DONATION", "DONATIONS",
    "MEMBERSHIP",
    # platforms (screen-capture chrome)
    "YOUTUBE", "YOUTUBE.COM", "BITCHUTE", "TIKTOK", "TWITTER", "FACEBOOK",
    "INSTAGRAM", "TELEGRAM", "DISCORD", "REDDIT", "GAB.COM", "ODYSEE",
    "RUMBLE", "DLIVE", "TWITCH", "BITWAVE", "GOYIMTV", "PARLER", "VK.COM",
    "SNAPCHAT", "WHATSAPP", "OMEGLE",
]
_UI_RE = _terms_to_pattern(_UI_TERMS)

# --------------------------------------------------------------------------------------
# F5  handle_watermark -- @handles and bare domains / URLs.
# Corpus df over 851: handle 76 (.089), domain 148 (.174), http:// 36 (.042).
# "©HANDLE" (OCR mis-read of "@HANDLE") is routed here.
# --------------------------------------------------------------------------------------
_HANDLE_PATTERNS = [
    re.compile(r"(?<![A-Z0-9])[@©]\s?[A-Z][A-Z0-9_.]{2,}"),             # @gypsycrusader, ©CATTOYKAMI
    re.compile(r"HTTPS?://[^\s]+"),
    re.compile(r"(?<![A-Z0-9])WWW\.[A-Z0-9.\-]{2,}"),
    re.compile(
        r"(?<![A-Z0-9@.])[A-Z0-9\-]{2,}\.(?:COM|NET|ORG|TV|IO|CO|INFO|GOV|EDU|RU|DE|UK)"
        r"(?![A-Z])"
    ),
]

# --------------------------------------------------------------------------------------
# F6  copyright -- rights markers.
# Corpus df over 851: COPYRIGHT 6, "(C) 20xx" 8, ALL RIGHTS RESERVED 2.
# Bare "©" is NOT counted when immediately followed by letters (handle mis-read).
# --------------------------------------------------------------------------------------
_COPYRIGHT_PATTERNS = [
    re.compile(r"(?<![A-Z0-9])COPYRIGHT(?![A-Z0-9])"),
    re.compile(r"\(C\)\s?\d{4}"),
    re.compile(r"©\s?\d{4}"),                                           # ©1974
    re.compile(r"©(?![A-Z0-9])"),                                       # standalone ©
    re.compile(r"(?<![A-Z0-9])ALL\s?RIGHTS?\s?RESERVED(?![A-Z0-9])"),
    re.compile(r"(?<![A-Z0-9])RIGHTS?\s?RESERVED(?![A-Z0-9])"),
    re.compile(r"ALLRIGHTSRESERVED"),
    re.compile(r"(?<![A-Z0-9])®(?![A-Z0-9])"),                          # ®
]

VOCAB = {
    "stock_watermark": {
        "terms": sorted(set(_STOCK_TERMS)),
        "regexes": [_STOCK_RE.pattern],
    },
    "news_chyron": {
        "terms": sorted(set(_NEWS_TERMS)),
        "regexes": [_NEWS_RE.pattern, _NEWS_STATION_RE.pattern],
    },
    "date_stamp": {
        "terms": [],
        "regexes": [p.pattern for p in _DATE_PATTERNS],
    },
    "ui_text": {
        "terms": sorted(set(_UI_TERMS)),
        "regexes": [_UI_RE.pattern],
    },
    "handle_watermark": {
        "terms": [],
        "regexes": [p.pattern for p in _HANDLE_PATTERNS],
    },
    "copyright": {
        "terms": [],
        "regexes": [p.pattern for p in _COPYRIGHT_PATTERNS],
    },
}

_FAMILY_PATTERNS = {
    "stock_watermark": [_STOCK_RE],
    "news_chyron": [_NEWS_RE, _NEWS_STATION_RE],
    "date_stamp": list(_DATE_PATTERNS),
    "ui_text": [_UI_RE],
    "handle_watermark": list(_HANDLE_PATTERNS),
    "copyright": list(_COPYRIGHT_PATTERNS),
}


def extract(text):
    """Return {family: int match count} over the whitespace-normalised, uppercased text."""
    t = normalize(text)
    out = {}
    for name in FEATURE_NAMES:
        n = 0
        if t:
            for pat in _FAMILY_PATTERNS[name]:
                n += len(pat.findall(t))
        out[name] = n
    return out


def extract_binary(text):
    """Return [0/1] * 6 in FEATURE_NAMES order (1 iff the family count is > 0)."""
    c = extract(text)
    return [1 if c[n] > 0 else 0 for n in FEATURE_NAMES]


def extract_vector(text):
    """Return the raw counts as a list in FEATURE_NAMES order."""
    c = extract(text)
    return [c[n] for n in FEATURE_NAMES]
