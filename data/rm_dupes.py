import os, sys, hashlib
from collections import defaultdict

root = sys.argv[1]
by_size = defaultdict(list)

for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f)
        try: by_size[os.path.getsize(p)].append(p)
        except OSError: pass

seen, dups = {}, []
for _, paths in by_size.items():
    if len(paths) < 2: continue
    for p in paths:
        try:
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        except OSError:
            continue
        if h in seen: dups.append(p)
        else: seen[h] = p

for p in dups:
    try: os.remove(p)
    except OSError: pass