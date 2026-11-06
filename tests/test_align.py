"""The aligner's parts that need no model: the reading's words, its pauses, and the timing file. python3 tests/test_align.py"""
import sys, types
from pathlib import Path

import numpy as np

# the models are not loaded by these parts; stand-ins keep the import cheap where torch is not installed
for name in ('torch', 'torchaudio', 'torchaudio.functional', 'uroman', 'transformers'):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            m = types.ModuleType(name); m.set_num_threads = lambda n: None
            m.AutoModelForCTC = m.Wav2Vec2FeatureExtractor = m.Wav2Vec2Processor = object
            sys.modules[name] = m
sys.modules['torchaudio'].functional = sys.modules['torchaudio.functional']
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timedwords import align as A  # noqa: E402
real_decode = A.decode

assert A.reading_words('אַתָּ֣ה בֶן־אָדָ֗ם הַגֵּ֤ד׃ (פ) וּמָדְד֖וּ') == ['אַתָּה', 'בֶן־אָדָם', 'הַגֵּד', 'וּמָדְדוּ'], 'cantillation goes, maqaf stays, an editorial note goes'
assert len(A.verses('א ב׃ ג ד׃ ')) == 2

sr = A.SR
tone = lambda s: 0.5 * np.sin(2 * np.pi * 220 * np.arange(int(s * sr)) / sr).astype(np.float32)
x = np.concatenate([tone(1.0), np.zeros(int(0.5 * sr), np.float32), tone(1.0)])
ps = A.pauses(x)
assert len(ps) == 1 and abs(ps[0][0] - 1.0) < 0.05 and abs(ps[0][1] - 1.5) < 0.05, ps

t = A.timing(['a', 'b', 'c'], [0.5, 1.25], [1.0, 2.0], off=10)
assert t == {'words': ['a', 'b', 'c'], 'starts': [10.5, 11.25, 12.0], 'ends': [11.0, 12.0, 12.0]}, t
# a recording shorter than one 20 ms frame has no pauses, and does not crash
assert A.pauses(np.zeros(100, np.float32)) == []

# mostly silence with one click: the silence is still a pause
y = np.zeros(2 * sr, np.float32); y[sr:sr + 160] = 0.8
assert any(b - a >= 0.5 for a, b in A.pauses(y)), A.pauses(y)

# no words is an error, never a pass
class Rom: romanize_string = lambda self, w, lcode: w
mms = A.Mms.__new__(A.Mms); mms.rom = Rom()
try:
    mms(np.zeros(sr, np.float32), []); raise SystemExit('Mms passed an empty word list')
except ValueError:
    pass

# a recheck that threw counts as unverified, and never shifts another word's score onto it
db = np.full(200, -20.0); hop = 0.01
starts = [0.1, 0.5, 1.0]; ends = [0.4, 0.9, 1.5]
rows = [(0, None, None), (1, 0.5, 0.5), (2, 1.6, 1.6)]
problems, unverified = A.check(['a', 'b', 'c'], starts, ends, [], rows, db, hop, log=lambda *a: None)
assert 0 in unverified and 1 not in unverified, unverified

# every recheck threw: fail loudly
problems, unverified = A.check(['a', 'b'], [0.1, 0.5], [0.4, 0.9], [], [(0, None, None), (1, None, None)], db, hop, log=lambda *a: None)
assert problems, 'every recheck threw, and the gate passed'
# a recording with no sound in it fails with a reason, never an IndexError
A.decode = lambda *a: np.full(sr, 1e-4, np.float32)
try:
    A.align('x', ['a'], log=lambda *a: None); raise SystemExit('align timed a silent recording')
except ValueError as e:
    assert 'no sound' in str(e), e
finally:
    A.decode = real_decode

# a word both aligners first put after a pause stays after it, though its re-ask lands before it
full = np.concatenate([tone(1.0), np.zeros(int(0.2 * sr), np.float32), tone(3.8)])
class Stub:
    def __call__(self, wav, words):
        if len(wav) == len(full): return [(0.1, 0.9), (1.4, 3.0), (3.1, 4.9)][:len(words)]
        return [(0.2, 0.4), (0.5, 0.8), (0.85, 0.95)][:len(words)]
real = A.decode, A.Mms, A.Hebrew
A.decode, A.Mms, A.Hebrew = (lambda *a: full), Stub, Stub
try:
    out = A.align('x', ['a', 'b', 'c'], log=lambda *a: None)
finally:
    A.decode, A.Mms, A.Hebrew = real
starts = out[5]
assert starts[1] >= 1.2, f'b crossed the pause back to {starts[1]:.2f}'

# romanised Hebrew becomes the phoneme model's symbols: digraphs first, the glottal stop, the uvular r
assert A.sounds("hari'u") == ['h', 'a', 'ʁ', 'i', 'ʔ', 'u'], A.sounds("hari'u")
assert A.sounds('beshimechah') == ['b', 'e', 'ʃ', 'i', 'm', 'e', 'χ', 'a', 'h']
assert A.sounds('vetson') == ['v', 'e', 'ts', 'o', 'n'] and A.sounds('kl-haarets') == ['k', 'l', 'h', 'a', 'a', 'ʁ', 'e', 'ts']

print('align test: every part holds')
