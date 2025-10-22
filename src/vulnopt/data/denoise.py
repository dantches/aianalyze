from pathlib import Path
import json
def denoise_jsonl(inp:Path, out:Path):
    rows=[json.loads(l) for l in inp.read_text(encoding='utf8').splitlines() if l.strip()]
    # naive: keep unique codes
    seen=set()
    keep=[]
    for r in rows:
        s=r.get('code','')
        if s in seen: continue
        seen.add(s)
        keep.append(r)
    out.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in keep))
