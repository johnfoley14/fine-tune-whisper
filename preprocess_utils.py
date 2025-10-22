
import re

def segment_words(word_items, min_len=5.0, max_len=30.0):
    """
    Split list of word items [{'startTime', 'endTime', 'value'}] into 5-30s segments.
    Prefers sentence or comma breaks, falls back to receptive windows.
    """
    segments = []
    current = []
    seg_start = None

    for w in word_items:
        start, end, text = float(w["startTime"]), float(w["endTime"]), w["value"]
        if seg_start is None:
            seg_start = start
        current.append((start, end, text))

        duration = end - seg_start
        # ensure each segment is over 5 secs
        if duration >= min_len and re.search(r'[.?!]', text):
            # 
            if duration <= max_len:
                seg_text = ''.join(t for _, _, t in current).strip()
                segments.append((seg_start, end, seg_text))
                current, seg_start = [], None
            else:
                # too long — try to break at commas
                seg_text = ''.join(t for _, _, t in current).strip()
                parts = re.split(r'[,;]', seg_text)
                t0 = seg_start
                chunk_len = (end - seg_start) / len(parts)
                for p in parts:
                    if not p.strip():
                        continue
                    t1 = min(t0 + chunk_len, end)
                    if t1 - t0 >= min_len:
                        segments.append((t0, t1, p.strip()))
                    t0 = t1
                current, seg_start = [], None

    # handle leftover words
    if current:
        start, end = current[0][0], current[-1][1]
        seg_text = ''.join(t for _, _, t in current).strip()
        if end - start > max_len:
            # fallback receptive windows
            t0 = start
            while t0 < end:
                t1 = min(t0 + max_len, end)
                segments.append((t0, t1, seg_text))
                if t1 == end:
                    break
                t0 += max_len * 0.8  # 20% overlap
        elif end - start >= min_len:
            segments.append((start, end, seg_text))

    return segments
