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
            m = types.ModuleType(name); m.set_num_threads = lambda n: None; m.AutoProcessor = m.AutoModelForCTC = object
            sys.modules[name] = m
sys.modules['torchaudio'].functional = sys.modules['torchaudio.functional']
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timedwords import align as A  # noqa: E402

assert A.reading_words('אַתָּ֣ה בֶן־אָדָ֗ם הַגֵּ֤ד׃ (פ) וּמָדְד֖וּ') == ['אַתָּה', 'בֶן־אָדָם', 'הַגֵּד', 'וּמָדְדוּ'], 'cantillation goes, maqaf stays, an editorial note goes'
assert len(A.verses('א ב׃ ג ד׃ ')) == 2

sr = A.SR
tone = lambda s: 0.5 * np.sin(2 * np.pi * 220 * np.arange(int(s * sr)) / sr).astype(np.float32)
x = np.concatenate([tone(1.0), np.zeros(int(0.5 * sr), np.float32), tone(1.0)])
ps = A.pauses(x)
assert len(ps) == 1 and abs(ps[0][0] - 1.0) < 0.05 and abs(ps[0][1] - 1.5) < 0.05, ps

t = A.timing(['a', 'b', 'c'], [0.5, 1.25], [1.0, 2.0], off=10)
assert t == {'words': ['a', 'b', 'c'], 'starts': [10.5, 11.25, 12.0], 'ends': [11.0, 12.0, 12.0]}, t
print('align test: every part holds')
