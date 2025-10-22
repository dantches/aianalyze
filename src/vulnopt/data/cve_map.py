from pathlib import Path
import orjson, re
GH_COMMIT = re.compile(r"https?://github.com/([^/]+/[^/]+)/commit/([0-9a-f]{7,40})")
def map_cve_references(nvd_json:Path, out:Path):
    items=orjson.loads(nvd_json.read_bytes())
    rows=[]
    for obj in items:
        cve = obj.get('cve',{}).get('id') or obj.get('cve',{}).get('CVE')
        refs = obj.get('cve',{}).get('references',[]) or []
        for ref in refs:
            url = ref.get('url') if isinstance(ref,dict) else ref
            if not url: continue
            m = GH_COMMIT.match(url)
            if m:
                repo=m.group(1)
                sha=m.group(2)
                rows.append({'cve':cve,'repo':repo,'commit':sha})
    out.write_text('\n'.join(orjson.dumps(r) for r in rows))
