# timed-words

When each word is said or sung in a recording, so a player can light each word as it is heard.

**Try it:** https://motizzle.github.io/timed-words/ plays three public-domain recordings with their words lit: a song, a
Hebrew reading and a talk.

| Module | Give it | You get |
|---|---|---|
| `timedwords.song` | a song whose words you do not have | its words set in lines as a lyric sheet sets them, each word timed, or why it is not a song |
| `timedwords.align` | a Hebrew reading or chant, and its text | each word's start and end, checked three ways; it fails loudly rather than guess |
| `timedwords.talk` | a talk or a sermon | a transcript in paragraphs of about a minute, each with its start time |

Everything runs locally on a CPU with free models (Whisper, Demucs, two wav2vec2 aligners). About 2 GB of models
download on the first run.

## Install

    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install git+https://github.com/motizzle/timed-words

You also need `ffmpeg`.

## Use

    python -m timedwords.song song.mp3 "Amazing Grace"        # hints: the title, names it mishears
    python -m timedwords.talk talk.mp3
    python -m timedwords.align reading.mp3 text.txt out.json

Or from Python:

```python
from timedwords import song, talk
heard = song.song('song.mp3', hints='Amazing Grace')   # {language, lines: [{start, text, words}], held}
body, paragraphs = talk.transcribe('talk.mp3')         # paragraphs: [{start, text}]
```

How each one works, and what each step was measured to fix, is at the top of its file.

## Test

    python tests/test_align.py

## The samples

In `docs/samples/`, all public domain:

- **Song:** "Amazing Grace", Original Sacred Harp Choir, 1922 ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Amazing_grace_1922.ogg)).
- **Hebrew reading:** Psalm 100, read by Rafi Metz ([Internet Archive](https://archive.org/details/PsalmsOfDavidInHebrew)).
- **Talk:** the Gettysburg Address, read by John Greenman for LibriVox ([Internet Archive](https://archive.org/details/gettysburg_johng_librivox)), 13.1 s to 95.0 s.

## License

MIT
