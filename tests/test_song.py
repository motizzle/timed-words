"""song.py's parts that need no model. python3 tests/test_song.py"""
import subprocess, sys, tempfile, types
from pathlib import Path

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
from timedwords import song  # noqa: E402

# demucs runs under the Python that runs this, so a machine with only python3 works
ran = []
real = subprocess.run
subprocess.run = lambda cmd, **kw: ran.append(cmd)
try:
    with tempfile.TemporaryDirectory() as d:
        try:
            song.voice(Path(d) / 'audio.wav', Path(d))
        except FileNotFoundError:
            pass
finally:
    subprocess.run = real
assert ran and ran[0][3] == sys.executable, ran[:1]
# English songs are timed by the English model, any other language by the phoneme model, in its own romanisation
from timedwords import align
made = []
class Stub:
    def __init__(self, *a): made.append(type(self).__name__); self.lcode = None
class English(Stub): pass
class Phonemes(Stub): pass
real = align.English, align.Phonemes
align.English, align.Phonemes = English, Phonemes
try:
    song.ALIGNERS_.clear()
    assert type(song.aligner('en')).__name__ == 'English'
    he = song.aligner('he'); assert type(he).__name__ == 'Phonemes' and he.lcode == 'heb'
    assert song.aligner('es') is he and he.lcode == 'spa', 'one phoneme model, its romanisation per song'
    assert made == ['English', 'Phonemes']
finally:
    align.English, align.Phonemes = real; song.ALIGNERS_.clear()

print('song test: every part holds')
