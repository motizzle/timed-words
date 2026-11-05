"""song.py's parts that need no model. python3 tests/test_song.py"""
import subprocess, sys, tempfile
from pathlib import Path

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
print('song test: every part holds')
