"""Share links: a page state stored under a short content hash. URL form: /#/<page>?s=<id>."""
import hashlib
import json
import re

ID_RE = re.compile(r"^[0-9a-f]{10}$")
MAX_BYTES = 200_000


def share_id(page, state):
    blob = json.dumps(dict(page=page, state=state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:10], blob


def save_share(cache, page, state):
    sid, blob = share_id(page, state)
    if len(blob) > MAX_BYTES:
        raise ValueError("state too large to share")
    cache.set_setting(f"share:{sid}", dict(page=page, state=state))
    return sid


def load_share(cache, sid):
    if not ID_RE.match(sid or ""):
        return None
    return cache.get_setting(f"share:{sid}")
