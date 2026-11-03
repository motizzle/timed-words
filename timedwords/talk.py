"""talk.py: a talk's words, in paragraphs of about a minute, each with its start time.

    from timedwords import talk
    body, paragraphs = talk.transcribe('sermon.mp3')   # paragraphs: [{start, text}]

WHAT IT NEEDS: faster-whisper. Free and local.
"""
MODEL = None  # faster-whisper, loaded on the first recording


def transcribe(f):
    """A transcript in paragraphs of about a minute, each with its start time."""
    global MODEL
    if MODEL is None:
        from faster_whisper import WhisperModel
        MODEL = WhisperModel('small', device='cpu', compute_type='int8')
    segs, _ = MODEL.transcribe(str(f), vad_filter=True)
    paras, cur, start = [], [], None
    for s in segs:
        if start is None:
            start = s.start
        cur.append(s.text.strip())
        if s.end - start >= 60 and s.text.rstrip().endswith(('.', '?', '!')):
            paras.append({'start': round(start, 1), 'text': ' '.join(cur)})
            cur, start = [], None
    if cur:
        paras.append({'start': round(start or 0, 1), 'text': ' '.join(cur)})
    return '\n\n'.join(p['text'] for p in paras), paras


if __name__ == '__main__':
    import json, sys
    body, paragraphs = transcribe(sys.argv[1])
    print(json.dumps(paragraphs, ensure_ascii=False, indent=1))
