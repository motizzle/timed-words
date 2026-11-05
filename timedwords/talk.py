"""talk.py: a talk's words, in paragraphs of about a minute, each with its start time.

    from timedwords import talk
    body, paragraphs = talk.transcribe('sermon.mp3')   # paragraphs: [{start, text}]
    body, paragraphs = talk.transcribe('sermon.mp3', words=True)   # each paragraph also has [{start, end, text}]

WHAT IT NEEDS: faster-whisper. Runs locally; the model's license is in the README.
"""
MODEL = None  # faster-whisper, loaded on the first recording


def transcribe(f, words=False):
    """A transcript in paragraphs of about a minute, each with its start time; with words, each word's times too."""
    global MODEL
    if MODEL is None:
        from faster_whisper import WhisperModel
        MODEL = WhisperModel('small', device='cpu', compute_type='int8')
    segs, _ = MODEL.transcribe(str(f), vad_filter=True, word_timestamps=words)
    paras, cur, start, ws = [], [], None, []
    for s in segs:
        if start is None:
            start = s.start
        cur.append(s.text.strip())
        ws += [{'start': round(w.start, 2), 'end': round(w.end, 2), 'text': w.word.strip()} for w in (s.words or [])]
        if s.end - start >= 60 and s.text.rstrip().endswith(('.', '?', '!')):
            paras.append({'start': round(start, 1), 'text': ' '.join(cur), **({'words': ws} if words else {})})
            cur, start, ws = [], None, []
    if cur:
        paras.append({'start': round(start or 0, 1), 'text': ' '.join(cur), **({'words': ws} if words else {})})
    return '\n\n'.join(p['text'] for p in paras), paras


if __name__ == '__main__':
    import json, sys
    body, paragraphs = transcribe(sys.argv[1], words='--words' in sys.argv)
    print(json.dumps(paragraphs, ensure_ascii=False, indent=1))
