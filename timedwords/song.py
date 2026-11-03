"""song.py: a song's words and when each word is sung, from its recording. The voice is lifted off the band, heard
by Whisper, set in lines as a lyric sheet sets them, and each word timed by the chant aligner's model.

    from timedwords import song
    song.song('recording.mp3', hints='the title, names Whisper mishears')   # {language, lines, held}

HOW, and what each step was measured to fix on the trial songs:

  1. The voice lifted off the band (demucs htdemucs, two stems). On the mix, Whisper heard
     "his fears" for "his face" and invented "Thank you." in the instrumental breaks.
  2. Whisper large-v3-turbo on the voice, with word times and no silence filter: the filter cut
     a six-minute song off at 1:35, its Hebrew blessing with it. It hears twice, plain and with
     hints (the title, names it mishears); only the names come from the hinted hearing.
  3. Lines as a lyric sheet sets them, never in the middle of a phrase: English Whisper heard without stops is
     punctuated by a local model; a sentence to a line; a long
     one cut at its comma or before a joining word, never on "the"; a capital starts a sentence unless it carries on
     a name. The gaps between sung words are nearly all zero, so time alone cannot find the line.
  4. A line the voice track is quiet under is dropped: that is Whisper hearing words in the band.
  5. Each word's start and end, for the singalong to light word by word: the chant aligner's MMS model
     aligns each line's own words inside that line's stretch of the voice (Whisper's times where it can't).
  6. A word or two left after a breath at a line's end moves to the next line (rehome).

WHAT IT NEEDS: torch (cpu), demucs, soundfile, faster-whisper, transformers; ffmpeg. About 2 GB of models download on
first run. Free and local.
"""
import json, os, re, shutil, subprocess, sys
from pathlib import Path

import numpy as np

BIN = Path(sys.executable).parent
SR = 16000
MODEL = 'large-v3-turbo'
HOW = f'whisper {MODEL} on the demucs voice track'
QUIET_DB = 30
THREADS = 6
QUIET = {**os.environ, 'OMP_NUM_THREADS': str(THREADS), 'MKL_NUM_THREADS': str(THREADS)}


def voice(wav, d):
    out = d / 'vocals.wav'
    if not out.exists():
        subprocess.run(['nice', '-n', '19', BIN / 'python', '-m', 'demucs', '--two-stems', 'vocals', '-n', 'htdemucs', '-o', d / 'sep', wav],
                       check=True, capture_output=True, env=QUIET)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', d / 'sep/htdemucs/audio/vocals.wav', '-ac', '1', '-ar', str(SR), out], check=True)
        shutil.rmtree(d / 'sep')
    return out


MODEL_ = None

def heard(vocals, d, hints):
    """Whisper's words for the voice, with or without hints; each hearing cached beside the song."""
    out = d / ('words-hinted.json' if hints else 'words.json')
    if out.exists() and json.loads(out.read_text()).get('hints', '') == hints:
        return json.loads(out.read_text())
    global MODEL_
    if MODEL_ is None:
        from faster_whisper import WhisperModel
        MODEL_ = WhisperModel(MODEL, device='cpu', compute_type='int8', cpu_threads=THREADS)
    segs, info = MODEL_.transcribe(str(vocals), vad_filter=False, word_timestamps=True, condition_on_previous_text=False,
                                   hotwords=hints or None)
    ws = [{'s': round(w.start, 2), 'e': round(w.end, 2), 'w': w.word, 'p': round(w.probability, 3), 'g': g}
          for g, s in enumerate(segs) for w in (s.words or [])]
    out.write_text(json.dumps({'language': info.language, 'words': ws, 'hints': hints}))
    return {'language': info.language, 'words': ws, 'hints': hints}


core = lambda w: re.sub(r'[^\w\']', '', w).lower()


def named(plain, hinted, hints):
    """The plain hearing, with a hint's name put in where the hinted hearing sang one at that moment.

    Hints fix names ("Sheena" becomes Shinar) but also bend other words that sound like them. So only a hinted
    word that is itself a hint replaces the
    plain words sung under it."""
    names = {core(x) for x in re.split(r'[^\w\']+', hints) if len(x) >= 3}
    out = [dict(w) for w in plain]
    for h in hinted:
        if core(h['w']) not in names or core(h['w']) in {core(w['w']) for w in plain}:
            continue
        under = [i for i, w in enumerate(out) if w and h['s'] <= (w['s'] + w['e']) / 2 <= h['e']]
        if under:
            stop = re.sub(r"[\w']", '', out[under[-1]]['w'].strip()[-1:])  # the plain words' own stop, if any
            out[under[0]] = {**out[under[0]], 'w': ' ' + re.sub(r"[^\w']", '', h['w']) + stop}
            for i in under[1:]:
                out[i] = None
    return [w for w in out if w]


def words_of(vocals, d, hints=''):
    plain = heard(vocals, d, '')
    if not hints:
        return plain
    return {**plain, 'words': named(plain['words'], heard(vocals, d, hints)['words'], hints)}


def loudness(vocals):
    """The voice track's level in dB, one value per 50 ms."""
    import soundfile as sf
    x, sr = sf.read(vocals, dtype='float32')
    hop = sr // 20
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)


DANGLING = {'a', 'an', 'the', 'to', 'of', 'with', 'and', 'my', 'your', 'our', 'his', 'her', 'their', 'in', 'on', 'for', 'at', 'from', 'by'}
LINE_MAX, LINE_MIN = 10, 3
JOINS = {'and', 'but', 'then', 'to', 'when', 'where', 'that', 'who', 'because', 'so', 'or', 'if', 'like', 'as'}
# ordinary words that start a sentence even if a song never sings them in lowercase
COMMON = {'but', 'so', 'or', 'now', 'then', 'we', 'you', 'he', 'she', 'they', 'it', 'this', 'that', 'there', 'when', 'where',
          'what', 'who', 'why', 'how', 'if', 'oh', 'yes', 'no', 'come', 'let', 'well'}
STOP = re.compile(r'[.?!]["”’)]*$')
PUNCT_ = None


def punctuate(words, language):
    """English that Whisper heard without stops or commas, punctuated the way the song is sung: Whisper's own mark
    where it put one, else the punctuation model's (oliverguhr/fullstop-punctuation-multilang-large, free and local).
    Marks only ever follow a word, so each word keeps its time. Other languages pass through as heard."""
    if language != 'en' or not words:
        return words
    global PUNCT_
    if PUNCT_ is None:
        from transformers import pipeline
        PUNCT_ = pipeline('token-classification', model='oliverguhr/fullstop-punctuation-multilang-large', aggregation_strategy='none')
    bare = [re.sub(r'[^\w\'’-]+$', '', w['w'].strip()) or w['w'].strip() for w in words]
    marks, spans, pos = {}, [], 0
    for b in bare:
        spans.append((pos, pos + len(b))); pos += len(b) + 1
    for k in range(0, len(bare), 200):  # the model reads 512 tokens at most: a song in pieces, the marks put back
        chunk = bare[k:k + 200]
        base = spans[k][0]
        for tok in PUNCT_(' '.join(chunk)):
            for i in range(k, k + len(chunk)):
                if spans[i][0] <= tok['start'] + base < spans[i][1]:
                    marks[i] = tok['entity']
    # only where Whisper wrote no punctuation in a whole phrase it heard: where it did, its own marks and capitals
    # are better than the model's guess ("And say giddy up right after me, of course")
    silent = {g for g in {w.get('g') for w in words} if not any(re.search(r'[.?!,]', w['w']) for w in words if w.get('g') == g)}
    out = []
    for i, w in enumerate(words):
        own = w['w'].strip()
        mark = {':': ',', ';': ','}.get(marks.get(i, '0'), marks.get(i, '0'))
        # a stop after "my" or "the" is the model guessing wrong: the sentence goes on
        if mark in ('.', '?') and own.lower().strip('"') in DANGLING:
            mark = '0'
        if w.get('g') in silent and not re.search(r'[.?!,]$', own) and mark in ('.', ',', '?'):
            own = own + mark
        out.append({**w, 'w': ' ' + own})
    return out


def dip(db, a, b):
    """How quiet the voice gets between two words, in dB: a breath between lines is a deep dip."""
    lo, hi = int(max(0, a - 0.1) * 20), int(b * 20 + 3)
    return float(np.min(db[lo:hi])) if hi > lo and lo < len(db) else 0.0


def split_long(ws, db):
    """A sentence too long for one sung line, cut where a lyric sheet would: at the comma nearest its middle, else at
    the deepest breath; never leaving a line on a word that leads into the next ("with", "the")."""
    if len(ws) <= LINE_MAX:
        return [ws]
    cuts = [i for i in range(LINE_MIN, len(ws) - LINE_MIN + 1) if ws[i - 1]['w'].strip().rstrip('.,?!:;').lower() not in DANGLING]
    if not cuts:
        cuts = list(range(LINE_MIN, len(ws) - LINE_MIN + 1)) or [len(ws) // 2]
    mid = len(ws) / 2
    commas = [i for i in range(2, len(ws) - 1) if ws[i - 1]['w'].strip().endswith(',')]
    if commas:
        at = min(commas, key=lambda i: abs(i - mid))
    else:
        # a breath, near the middle, before a joining word ("the wallet / to the police station")
        at = min(cuts, key=lambda i: dip(db, ws[i - 1]['e'], ws[i]['s']) + 2 * abs(i - mid)
                 - (8 if ws[i]['w'].strip().lower() in JOINS else 0))
    return split_long(ws[:at], db) + split_long(ws[at:], db)


def lines_of(words, db, language='en'):
    """Words into sung lines, as a lyric sheet sets them: a sentence to a line, a long one cut at a comma or a breath,
    a breath of two seconds always a new line, each line starting with a capital. A line the voice is quiet under
    goes (Whisper hearing words in the band)."""
    # a mark Whisper heard alone ("." over an instrumental intro) is not a word: it joins the word before it, or goes
    kept = []
    for w in words:
        if not re.search(r'\w', w['w']):
            if kept and not re.search(r'[.,?!]$', kept[-1]['w'].strip()):
                kept[-1] = {**kept[-1], 'w': kept[-1]['w'] + w['w'].strip()[:1], 'e': w['e']}
            continue
        kept.append(w)
    words = punctuate(kept, language)
    # a word the song also sings in lowercase is an ordinary word; one it only ever capitalises is a name
    plain = {w['w'].strip().strip('.,?!:;"').lower() for w in words if w['w'].strip()[:1].islower()} | COMMON | DANGLING
    def begins(w, prev, nxt):
        """Whisper capitalises the first word of a sentence it hears: a capital after no stop starts one when the song
        also sings that word in lowercase. One it never does is a name ("her name is Lola", "Mister Rogers")."""
        word, last = w['w'].strip(), prev['w'].strip()
        bare = word.strip('.,?!:;"').lower()
        if not word[:1].isupper() or word.split("'")[0] == 'I' or re.search(r'[,.?!:;]$', last):
            return False
        if bare in COMMON:
            return True  # "Lola / She was a showgirl"
        if last[:1].isupper():
            return False  # a capital after a capital carries on a name: "He's Mister Rogers"
        if nxt and nxt['w'].strip()[:1].isupper() and nxt['w'].strip().strip('.,?!:;"').lower() not in COMMON:
            return False  # two capitals in a row are a name: "and he's Mister Rogers"
        return bare in plain
    sentences, cur = [], []
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        if cur and (w['s'] - cur[-1]['e'] >= 2 or (len(cur) >= LINE_MIN - 1 and begins(w, cur[-1], nxt))):
            sentences.append(cur); cur = []
        cur.append(w)
        # a sentence ends at a stop, or where one of Whisper's own heard phrases ends on a comma or a stop
        last_of_phrase = i + 1 < len(words) and words[i + 1].get('g') != w.get('g')
        if STOP.search(w['w'].strip()) or (last_of_phrase and re.search(r'[,.?!]$', w['w'].strip()) and len(cur) >= LINE_MIN):
            sentences.append(cur); cur = []
    if cur:
        sentences.append(cur)
    groups = [g for s in sentences for g in split_long(s, db)]
    peak = np.percentile(db, 95)
    out = []
    for g in groups:
        span = db[int(g[0]['s'] * 20): max(int(g[-1]['e'] * 20), int(g[0]['s'] * 20) + 1)]
        if not len(span) or np.median(span) < peak - QUIET_DB:
            continue
        toks = [w['w'].strip() for w in g]
        if toks and toks[0][:1].islower():
            toks[0] = toks[0][0].upper() + toks[0][1:]
        if not all(toks):
            continue
        line = {'start': round(g[0]['s'], 1), 'text': ' '.join(toks), 'say': None,
                'words': [{'start': round(w['s'], 2), 'end': round(w['e'], 2), 'text': x} for w, x in zip(g, toks)]}
        out.append(line)
    return out


# A word or two the singer holds back after a breath belongs to the next line. Measured on the aligned word times,
# which see a breath Whisper's own times
# stretch a word across. A pause over 3 s is a break in the band, where those times wander; a sentence's last words
# stay ("No, I won't / be afraid." is a held note, not a breath).
REHOME = (1.0, 3.0)


def rehome(lines):
    """Lines with a word or two after a breath at their end moved to the start of the next line, when no breath
    parts them from it."""
    out = [dict(l) for l in lines]
    full = lambda l: bool(l.get('words')) and len(l['words']) == len(l['text'].split())
    for i in range(len(out) - 1):
        l, n = out[i], out[i + 1]
        if not full(l) or not full(n):
            continue
        ws = l['words']
        k = next((k for k in range(len(ws) - 1, 0, -1) if REHOME[0] <= ws[k]['start'] - ws[k - 1]['end'] <= REHOME[1]), None)
        if k is None or k < LINE_MIN or len(ws) - k >= LINE_MIN or STOP.search(ws[-1]['text']) or not ws[-1]['start'] <= n['words'][0]['start'] < ws[-1]['end'] + REHOME[0]:
            continue
        moved = ws[k:] + n['words']
        moved = [{**moved[0], 'text': moved[0]['text'][:1].upper() + moved[0]['text'][1:]}] + moved[1:]
        out[i] = {**l, 'text': ' '.join(w['text'] for w in ws[:k]), 'words': ws[:k]}
        out[i + 1] = {**n, 'start': round(moved[0]['start'], 1), 'text': ' '.join(w['text'] for w in moved), 'words': moved}
    return out


MMS_ = None


def timed_words(lines, vocals):
    """Each line's words with their start and end, by forced alignment of the line's own words inside its
    stretch of the voice track: the chant aligner's MMS model (align.py).
    A line the aligner cannot place (a word with no letters, a stretch too short) keeps Whisper's own word times."""
    global MMS_
    import soundfile as sf
    if MMS_ is None:
        from . import align
        MMS_ = align.Mms()
    wav, sr = sf.read(vocals, dtype='float32')
    end = len(wav) / sr
    out = []
    for i, l in enumerate(lines):
        words = l['text'].split()
        t0 = max(0.0, l['start'] - 0.3)
        t1 = min(end, lines[i + 1]['start'] if i + 1 < len(lines) else l['start'] + 1.2 * len(words) + 2)
        l = dict(l)
        try:
            if t1 - t0 < 0.15 * len(words) or not all(MMS_.token(w) for w in words):
                raise ValueError('no room or no letters')
            spans = MMS_(wav[int(t0 * sr):int(t1 * sr)], words)
            ws = [{'start': round(t0 + s, 2), 'end': round(t0 + e, 2), 'text': w} for w, (s, e) in zip(words, spans)]
            if any(a['start'] > b['start'] for a, b in zip(ws, ws[1:])):
                raise ValueError('out of order')
            l['words'] = ws
        except Exception:  # noqa: BLE001 — Whisper's word times stand
            pass
        out.append(l)
    return out


KNOWN_BAR = 0.75  # the share of words both of the chant aligner's models agree on within 200 ms; below it, held

def said(word):
    """How a written word is sung: the divine name (יְהֹוָה, יְהוָה, יְיָ) is sung Adonai."""
    return 'אֲדֹנָי' + re.sub('[\u0590-\u05FF]', '', word) if re.sub('[^\u05D0-\u05EA]', '', word) in ('יהוה', 'יי') else word


# what Whisper writes over silence or an instrumental, not a sung line
FILLER = {'thank you.', 'thank you', 'thanks for watching!', 'amen.', 'amen', 'you', 'תודה רבה.', 'תודה רבה'}
# what Whisper writes over a choir or a long instrumental: captions it learned from video, never sung here
NOT_SUNG = re.compile(r"satsang with mooji|thanks for watching|subscribe to|subtitles by|transcribed by|^(thank you[,.!]?\s*)+$"
                      r"|^to be continued|^(תודה( רבה)?[,.!]?\s*)+$", re.I)   # its Hebrew "thank you" too, once or over and over


def sung(entry):
    """A heard song without the lines Whisper made up: they are dropped, so a singalong never lights them."""
    return {**entry, 'lines': [l for l in entry['lines'] if not NOT_SUNG.search(l['text'].strip())]}


def held(entry):
    """Why a result is not a singalong, or None: the aligner's models disagree, or nothing was sung."""
    if 'agree' in entry:  # timed on a known text: a blessing of one line is honest
        return None if entry['agree'] >= KNOWN_BAR else (
            f"the aligner's two models agree on {entry['agree']:.0%} of its words, under {KNOWN_BAR:.0%}: the recording repeats or changes the text")
    sung = [l for l in entry['lines'] if l['text'].strip().lower() not in FILLER]
    if len(sung) < 3:
        return 'no singing heard: an instrumental, or talk over silence'
    return None


def song(path, hints=''):
    """A recording, heard: its words in lines, each word timed, and why it is not a singalong (held), or None."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        wav = d / 'audio.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-threads', str(THREADS), '-y', '-i', path, '-ac', '2', '-ar', '44100', wav], check=True)
        v = voice(wav, d)
        heard = words_of(v, d, hints)
        entry = sung({'how': HOW, 'language': heard['language'], 'lines': rehome(timed_words(lines_of(heard['words'], loudness(v), heard['language']), v))})
        return {**entry, 'held': held(entry)}


if __name__ == '__main__':
    print(json.dumps(song(sys.argv[1], ' '.join(sys.argv[2:])), ensure_ascii=False, indent=1))
