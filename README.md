# timed-words

When each word is said or sung in a recording, so a player can light each word as it is heard.

**Try it:** https://motizzle.github.io/timed-words/ plays three public-domain recordings with their words lit: a song, a
Hebrew reading and a talk.

| Module | Give it | You get |
|---|---|---|
| `timedwords.song` | a song whose words you do not have | its words set in lines as a lyric sheet sets them, each word timed, or why it is not a song |
| `timedwords.align` | a Hebrew reading or chant, and its text | each word's start and end, checked three ways; it fails loudly rather than guess |
| `timedwords.talk` | a talk or a sermon | a transcript in paragraphs of about a minute, each with its start time, and each word's times if you ask |

Everything runs locally on a CPU. About 2 GB of models download on the first run; each has its own license, below.

## Install

    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install git+https://github.com/motizzle/timed-words

You also need `ffmpeg`.

## Use

    python -m timedwords.song song.mp3 "Amazing Grace"        # hints: the title, names it mishears
    python -m timedwords.talk talk.mp3 --words               # --words: each word's start and end too
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
- **Talk:** the Gettysburg Address, read by John Greenman for LibriVox ([Internet Archive](https://archive.org/details/gettysburg_johng_librivox)), 12.75 s to 147.57 s: the whole address, without LibriVox's opening and closing words.

## License

The code is MIT.

The models are not part of this repository. Each downloads from its maker on the first run, under its own license:

| Model | Used by | License |
|---|---|---|
| Whisper `small` and `large-v3-turbo`, as converted for faster-whisper | `talk`, `song` | MIT |
| Demucs `htdemucs` | `song` | MIT |
| `oliverguhr/fullstop-punctuation-multilang-large` | `song` | MIT |
| Meta's MMS aligner (`torchaudio.pipelines.MMS_FA`) | `song`, `align` | CC-BY-NC 4.0: not for commercial use |
| `imvladikon/wav2vec2-large-xlsr-53-hebrew` | `align` | Apache 2.0 |

So `talk` can be used for anything; `song` and `align` rely on a model that is not for commercial use.

This project uses the universal romanizer software 'uroman' written by Ulf Hermjakob, USC Information Sciences
Institute (2015-2020).
