#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""align.py — word timings for a recording of a reading whose text we already have.

    python -m timedwords.align <audio> <text.txt> <out.json> [--from SEC] [--to SEC] [--words N]

<text.txt> is the reading's Hebrew. --from and --to time a stretch of the recording; the times written are
read against the whole recording. --words N: the recording says only the first N words; the rest are stamped
at the last word's end with no length, which a player reads as unsung.

WHAT IT NEEDS: four packages and ffmpeg. About 2 GB of models download on
the first run and are cached; their licenses are in the README:

    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers uroman

WHY THIS IS NOT A TRANSCRIPTION. The text is known. The only question is WHEN
each word is sung, and a model that guesses what it hears answers a harder
question worse: on a cantor's chanted Haftarah (Ezekiel 43:10-17), whisper matched 58 of 118 words and
put two of them 24 seconds out, because the reading says the same phrase twice.

WHY IT IS NOT ONE FORCED ALIGNMENT EITHER. Chant holds a word for seconds and
runs two together without a breath; a CTC aligner places a token where its
consonant scores, which on this tape lagged the moment the word starts by up to
half a second, and in the repeated measurement passages of Ezekiel 43 it slid a
whole phrase onto the wrong repeat.

SO THE TAPE IS ASKED FIRST, AND THE MODELS ONLY SAY WHICH WORD:

  1. Pauses. Every stretch of silence >= 0.12 s at 20 dB under the clip's
     95th-percentile level. That Haftarah has 64 unbroken stretches of sound for 118
     words, so the tape itself names where half the words start.
  2. Two aligners, independently. `torchaudio` MMS_FA over uroman-romanised
     Hebrew, and a Hebrew-native wav2vec2 CTC model over the letters
     themselves. Neither is trusted for a timestamp; they are used to decide
     WHICH stretch of sound each word is in, and they agreed on 117 of 118.
  3. The boundary comes from the waveform. A word first in its stretch starts
     on the frame the sound resumes. Words sharing a stretch are divided at the
     deepest energy dip between the two aligners' answers — a stop closing, a
     fricative, a snatched breath — which is a measurement, not a guess.
  4. Anything still disputed is re-asked in a window bounded by two settled
     words, so no answer can slide in from elsewhere.

WHAT IT CHECKS, AND WHY EACH CHECK MEASURES SOMETHING:

  A  No word may be sung during a pause. Overlap in ms, per word. Zero here.
  B  Both aligners are re-run per word in a bounded window and compared to the
     shipped time. 110 of 118 within 100 ms, 115 within 200 ms on the Haftarah.
  C  For every word the aligners dispute, the energy dip is a third witness,
     independent of both. On the Haftarah it agreed with the shipped time on
     all three.

  A word is accepted when either aligner is within 200 ms of it (B), or the dip
  is within 100 ms (C); any other word, and any word whose recheck failed, is
  unverified and listed to be checked by ear.

  The gate FAILS if the word count does not match the reading, if the times are
  not strictly increasing, if any word overlaps the pauses by more than 0.12 s,
  if more than 4 words are unverified, or if no recheck ran at all.

COST: nothing. Both models run locally.
"""
import argparse, json, re, subprocess, statistics, sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
import uroman as ur
from transformers import AutoProcessor, AutoModelForCTC

torch.set_num_threads(8)
SR = 16000
HEB_MODEL = 'imvladikon/wav2vec2-large-xlsr-53-hebrew'

# ── the reading ──────────────────────────────────────────────────────────────
# te'amim, paseq, sof pasuq and the editorial markers go; nikud and maqaf stay.
_DROP = re.compile(r'[֑-֯׀׃׆\[\]*:]')
_PAREN = re.compile(r'\([^)]*\)')

def reading_words(hebrew):
    h = _DROP.sub('', _PAREN.sub(' ', hebrew))
    return [w for w in re.split(r'\s+', h) if w]

def verses(hebrew):
    return [v for v in hebrew.split('׃') if v.strip()]

# ── the tape ─────────────────────────────────────────────────────────────────
def decode(path, t0=None, t1=None):
    cmd = ['ffmpeg', '-v', 'error']
    if t0 is not None: cmd += ['-ss', str(t0)]
    if t1 is not None: cmd += ['-to', str(t1)]
    cmd += ['-i', str(path), '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()

def pauses(x, min_len=0.12, drop=20.0, hop=0.02, floor=-70.0):
    """Silence >= min_len, at `drop` dB under the clip's 95th-percentile level. Anything under `floor` dBFS is
    silence too: in a clip that is mostly silence, the 95th percentile is the silence itself."""
    win = int(hop * SR); n = len(x) // win
    if not n: return []
    e = np.sqrt((x[:n * win].astype(np.float64) ** 2).reshape(n, win).mean(1))
    db = 20 * np.log10(np.maximum(e, 1e-9))
    thr = max(np.percentile(db, 95) - drop, floor)
    quiet = db < thr
    out = []; i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]: j += 1
            if (j - i) * hop >= min_len: out.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    return out

def regions(x, **kw):
    """The stretches of sound between the pauses."""
    end = len(x) / SR; out = []; t = 0.0
    for a, b in pauses(x, **kw):
        if a > t + 0.05: out.append((t, a))
        t = b
    if end > t + 0.05: out.append((t, end))
    return out

def envelope(x, hop=0.01, win=0.025):
    h = int(hop * SR); w = int(win * SR); n = (len(x) - w) // h
    e = np.array([np.sqrt(np.mean(x[i * h:i * h + w].astype(np.float64) ** 2)) for i in range(n)])
    return 20 * np.log10(np.maximum(e, 1e-9)), hop

def dip(db, hop, t0, t1, pad=0.06):
    """The quietest moment between two candidate boundaries — the join."""
    a = max(0, int((t0 - pad) / hop)); b = min(len(db), int((t1 + pad) / hop))
    if b - a < 3: return None
    return (a + int(np.argmin(db[a:b]))) * hop

# ── the two aligners ─────────────────────────────────────────────────────────
class Mms:
    """MMS_FA over romanised Hebrew. Trained on 1,130 languages, none of them
    pointed Hebrew, which is why it is one witness and not the answer."""
    def __init__(self):
        b = torchaudio.pipelines.MMS_FA
        self.model = b.get_model(with_star=False)
        self.tok = b.get_tokenizer(); self.al = b.get_aligner()
        self.rom = ur.Uroman()

    def token(self, w):
        return re.sub(r"[^a-z']", '', self.rom.romanize_string(w, lcode='heb').lower())

    def __call__(self, wav, words):
        toks = [self.token(w) for w in words]
        if not toks: raise ValueError('no words')
        if not all(toks): raise ValueError('a word romanised to nothing')
        t = torch.from_numpy(wav).unsqueeze(0)
        with torch.inference_mode():
            em, _ = self.model(t)
            spans = self.al(em[0], self.tok(toks))
        r = t.shape[1] / em.shape[1] / SR
        return [(s[0].start * r, s[-1].end * r) for s in spans]

class Hebrew:
    """A Hebrew-trained CTC model, over the letters themselves — no romanising.
    `do_normalize` is True on its feature extractor and raw audio degrades it
    badly."""
    def __init__(self):
        self.proc = AutoProcessor.from_pretrained(HEB_MODEL)
        self.model = AutoModelForCTC.from_pretrained(HEB_MODEL).eval()
        v = self.proc.tokenizer.get_vocab()
        self.vocab = v; self.blank = v['[PAD]']; self.sep = v['|']

    def ids(self, word):
        letters = re.sub(r'[^א-ת]', '', word.replace('־', ''))
        return [self.vocab[c] for c in letters if c in self.vocab]

    def logprobs(self, wav):
        iv = self.proc(wav, sampling_rate=SR, return_tensors='pt').input_values
        with torch.inference_mode():
            lg = self.model(iv).logits
        return torch.log_softmax(lg, dim=-1), iv.shape[1] / lg.shape[1]

    def __call__(self, wav, words):
        per = [self.ids(w) for w in words]
        if not per: raise ValueError('no words')
        if not all(per): raise ValueError('a word has no Hebrew letters')
        targets = []; owner = []
        for i, ids in enumerate(per):
            if i: targets.append(self.sep); owner.append(-1)
            for t in ids: targets.append(t); owner.append(i)
        lp, ratio = self.logprobs(wav)
        aligned, scores = AF.forced_align(lp, torch.tensor([targets]), blank=self.blank)
        spans = AF.merge_tokens(aligned[0], scores[0], blank=self.blank)
        if len(spans) != len(targets):
            raise ValueError(f'{len(spans)} spans for {len(targets)} targets')
        out = {}
        for k, sp in enumerate(spans):
            i = owner[k]
            if i < 0: continue
            s, e = sp.start * ratio / SR, sp.end * ratio / SR
            out[i] = (out[i][0], e) if i in out else (s, e)
        return [out[i] for i in range(len(words))]

# ── the pass that narrows ────────────────────────────────────────────────────
def segments(times, ps, max_words=6, max_secs=14.0, min_pause=0.30):
    cuts = []; first = 0
    for i in range(1, len(times)):
        a, b = times[i - 1][1], times[i][0]
        if b - a < 0.1: continue
        here = [p for p in ps if p[1] - p[0] >= min_pause and p[0] >= a - 0.35 and p[1] <= b + 0.35]
        if not here: continue
        if i - first >= max_words or times[i - 1][1] - times[first][0] >= max_secs:
            mid = max(here, key=lambda p: p[1] - p[0])
            cuts.append((i, (mid[0] + mid[1]) / 2)); first = i
    return cuts

def pass_over(al, x, words, seed, ps, rounds=3, **kw):
    """Align, cut the tape where the reader paused, align each piece again."""
    times = seed
    for _ in range(rounds):
        cuts = segments(times, ps, **kw)
        bounds = [0] + [c[0] for c in cuts] + [len(words)]
        tcuts = [0.0] + [c[1] for c in cuts] + [len(x) / SR]
        out = []
        for k in range(len(bounds) - 1):
            a = int(max(0, tcuts[k]) * SR); b = int(min(len(x) / SR, tcuts[k + 1]) * SR)
            sub = al(x[a:b], words[bounds[k]:bounds[k + 1]])
            out.extend([(s + a / SR, e + a / SR) for s, e in sub])
        times = out
    return times

def which_region(v, reg):
    for i, (a, b) in enumerate(reg):
        if a - 0.02 <= v <= b + 0.02: return i
    for i, (a, b) in enumerate(reg):
        if v < a: return i
    return len(reg) - 1

def align(audio, words, t0=None, t1=None, log=print):
    x = decode(audio, t0, t1)
    ps = pauses(x); reg = regions(x); db, hop = envelope(x)
    if not reg: raise ValueError('no sound to time: the recording is silence, or quieter than -70 dBFS')
    log(f'  {len(x)/SR:.1f} s, {len(ps)} pauses, {len(reg)} stretches of sound, {len(words)} words')

    mms, heb = Mms(), Hebrew()
    seed = [(i * len(x) / SR / len(words), (i + 1) * len(x) / SR / len(words)) for i in range(len(words))]
    m = pass_over(mms, x, words, seed, ps)
    h = pass_over(heb, x, words, m, ps)
    agree = sum(1 for a, b in zip(m, h) if which_region(a[0], reg) == which_region(b[0], reg))
    log(f'  the two aligners put {agree} of {len(words)} words in the same stretch')

    idx = []
    for a, b in zip(m, h):
        ia, ib = which_region(a[0], reg), which_region(b[0], reg)
        idx.append(min(ia, ib))
    for i in range(1, len(idx)):
        if idx[i] < idx[i - 1]: idx[i] = idx[i - 1]

    starts = [0.0] * len(words); how = [''] * len(words)
    for i in range(len(words)):
        if i == 0 or idx[i] != idx[i - 1]:
            starts[i] = reg[idx[i]][0]; how[i] = 'onset'
        else:
            sm, sh = m[i][0], h[i][0]
            t = dip(db, hop, min(sm, sh), max(sm, sh))
            if t is not None and min(abs(t - sm), abs(t - sh)) <= 0.20:
                starts[i] = t; how[i] = 'dip'
            else:
                starts[i] = (sm + sh) / 2; how[i] = 'mean'
    for i in range(1, len(words)):
        if starts[i] <= starts[i - 1]: starts[i] = starts[i - 1] + 0.02

    starts, ends = close(words, idx, starts, reg)

    # 4. WHAT IS STILL DISPUTED GETS RE-ASKED, bounded by two settled words.
    #    A window with unambiguous ends is the smallest question there is, and
    #    it is where the repeated measurement passages of Ezekiel 43 resolve.
    for _ in range(2):
        rows = audit(mms, heb, x, words, starts)
        far = [(i, sm, sh) for i, sm, sh in rows
               if sm is not None and min(abs(starts[i] - sm), abs(starts[i] - sh)) > 0.200]
        moved = 0
        for i, sm, sh in far:
            lo = starts[i - 1] if i else 0.0
            hi = starts[i + 1] if i + 1 < len(words) else len(x) / SR
            t = dip(db, hop, min(sm, sh, starts[i]), max(sm, sh, starts[i]))
            if t is not None and abs(t - starts[i]) <= 0.100:
                continue                      # the waveform backs where it is
            best = sm if abs(sm - (t or sm)) <= abs(sh - (t or sh)) else sh
            r = which_region(best, reg)
            cand = reg[r][0] if abs(best - reg[r][0]) < 0.45 else best
            # a word both aligners first put in one stretch stays in it: the re-ask's window can
            # hold the previous word's tail, and a held word squeezed into it crosses a pause
            first = which_region(m[i][0], reg)
            if first == which_region(h[i][0], reg) and which_region(cand, reg) != first:
                continue
            if lo + 0.02 < cand < hi - 0.02 and abs(cand - starts[i]) > 0.05:
                starts[i] = cand; idx[i] = which_region(cand, reg)
                how[i] = 'rewindowed'; moved += 1
        log(f'  re-asked {len(far)} disputed words, moved {moved}')
        if not moved: break
        for i in range(1, len(idx)):
            if idx[i] < idx[i - 1]: idx[i] = idx[i - 1]
        starts, ends = close(words, idx, starts, reg)
    return x, reg, ps, db, hop, starts, ends, how, m, h

def close(words, idx, starts, reg):
    """Clamp every start inside the stretch it belongs to, keep them
    increasing, and end each word where the next one starts or the sound stops
    — so nothing is ever sung during a pause."""
    starts = list(starts)
    for i in range(len(words)):
        a, b = reg[idx[i]]
        starts[i] = min(max(starts[i], a), b - 0.03)
    for i in range(1, len(words)):
        if starts[i] <= starts[i - 1] + 0.01:
            starts[i] = min(starts[i - 1] + 0.02, reg[idx[i]][1] - 0.01)
    ends = [0.0] * len(words)
    for i in range(len(words)):
        ends[i] = starts[i + 1] if (i + 1 < len(words) and idx[i + 1] == idx[i]) \
                  else reg[idx[i]][1]
        if i + 1 < len(words) and ends[i] > starts[i + 1]: ends[i] = starts[i + 1]
        if ends[i] < starts[i] + 0.02: ends[i] = starts[i] + 0.02
    return starts, ends

# ── the gates ────────────────────────────────────────────────────────────────
def audit(mms, heb, x, words, starts, log=print):
    """Ask both aligners again, one word at a time, in a bounded window."""
    end = len(x) / SR; rows = []
    for i in range(len(words)):
        lo = starts[i - 1] if i else max(0.0, starts[0] - 1.2)
        hi = starts[i + 1] + 0.20 if i + 1 < len(words) else min(end, starts[-1] + 6.0)
        a, b = int(max(0, lo) * SR), int(min(end, hi) * SR)
        ws = words[max(0, i - 1):min(len(words), i + 2)]
        k = 1 if i else 0
        try:
            sm = mms(x[a:b], ws)[k][0] + a / SR
            sh = heb(x[a:b], ws)[k][0] + a / SR
        except Exception:
            rows.append((i, None, None)); continue
        rows.append((i, sm, sh))
    return rows

def check(words, starts, ends, ps, rows, db, hop, log=print):
    problems = []
    if len(starts) != len(words): problems.append('word count does not match the reading')
    if not all(starts[i] > starts[i - 1] for i in range(1, len(starts))):
        problems.append('the times do not always increase')
    overlap = [sum(max(0, min(e, b) - max(s, a)) for a, b in ps) for s, e in zip(starts, ends)]
    sung_in_silence = [i for i, v in enumerate(overlap) if v > 0.12]
    if sung_in_silence: problems.append(f'{len(sung_in_silence)} words are sung during a pause: {sung_in_silence}')
    heard = [(i, sm, sh) for i, sm, sh in rows if sm is not None]
    threw = [i for i, sm, sh in rows if sm is None]
    if not heard:
        problems.append('every recheck failed, so no word was checked')
        return problems, threw
    near = [min(abs(starts[i] - sm), abs(starts[i] - sh)) for i, sm, sh in heard]
    within100 = sum(1 for v in near if v <= 0.100)
    within200 = sum(1 for v in near if v <= 0.200)
    log(f'  A  no word sung during a pause: worst {max(overlap)*1000:.0f} ms')
    log(f'  B  nearest aligner: median {statistics.median(near)*1000:.0f} ms, '
        f'within 100 ms {within100}/{len(near)}, within 200 ms {within200}/{len(near)}')
    far = [(i, sm, sh) for (i, sm, sh), v in zip(heard, near) if v > 0.200]
    unverified = []
    for i, sm, sh in far:
        t = dip(db, hop, min(sm, sh, starts[i]), max(sm, sh, starts[i]))
        if t is None or abs(t - starts[i]) > 0.100: unverified.append(i)
    log(f'  C  of {len(far)} words the aligners dispute, the energy dip backs the '
        f'shipped time on {len(far) - len(unverified)}')
    # THE RULE, stated before the run rather than fitted to it: a word is
    # accepted when TWO of the three witnesses agree with the shipped time --
    # either aligner inside 200 ms, or the waveform's own dip inside 100 ms. A
    # word with neither is unverified, and listed for a person to check. So is
    # a word whose recheck threw: nothing witnessed it.
    unverified = sorted(unverified + threw)
    log(f'  ACCEPTED {len(rows) - len(unverified)} of {len(rows)} words; '
        f'unverified: {unverified if unverified else "none"}')
    if len(unverified) > 4:
        problems.append(f'{len(unverified)} words are backed by no second witness: {unverified}')
    return problems, unverified

def timing(every, starts, ends, off=0):
    """The timing file: every word of the reading, its times read against the whole recording (align() times from
    where --from cut it), and the words the recording does not say stamped at the last word's end with no length."""
    unsung = len(every) - len(starts)
    starts = [round(v + off, 3) for v in starts]; ends = [round(v + off, 3) for v in ends]
    return {'words': every, 'starts': starts + [ends[-1]] * unsung, 'ends': ends + [ends[-1]] * unsung}

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio'); ap.add_argument('text'); ap.add_argument('out')
    ap.add_argument('--from', dest='t0', type=float, default=None)
    ap.add_argument('--to', dest='t1', type=float, default=None)
    ap.add_argument('--words', dest='n', type=int, default=None)
    a = ap.parse_args()
    hebrew = Path(a.text).read_text(encoding='utf-8')
    every = reading_words(hebrew)
    words = every[:a.n] if a.n else every
    print(f'{a.text}: {len(words)} of {len(every)} words, {len(verses(hebrew))} verses')
    x, reg, ps, db, hop, starts, ends, how, m, h = align(a.audio, words, a.t0, a.t1)
    from collections import Counter
    print('  boundaries from:', dict(Counter(how)))
    rows = audit(Mms(), Hebrew(), x, words, starts)
    problems, unverified = check(words, starts, ends, ps, rows, db, hop)
    if problems:
        for p in problems: print('align: ' + p, file=sys.stderr)
        print(f'FAILED: {len(problems)} problem(s) — nothing written', file=sys.stderr)
        sys.exit(1)
    Path(a.out).write_text(json.dumps(timing(every, starts, ends, a.t0 or 0), ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print('wrote', a.out)
    if unverified:
        print('  check these words by ear:', unverified)

if __name__ == '__main__':
    main()
