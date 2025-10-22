from pathlib import Path
import json
from git import Repo
def mine_repos(map_csv:Path, out_jsonl:Path, cache_dir:Path):
    rows = [l.strip() for l in map_csv.read_text().splitlines() if l.strip()]
    out=open(out_jsonl,'w',encoding='utf8')
    for r in rows:
        try:
            d = json.loads(r)
        except:
            continue
        repo = d.get('repo')
        if not repo: continue
        local = cache_dir / repo.replace('/','__')
        if not local.exists():
            Repo.clone_from(f'https://github.com/{repo}', local)
        # simple: walk py files and emit file-level samples
        for p in local.rglob('*.py'):
            code = p.read_text(errors='ignore')
            rec = {'id':str(p),'repo':repo,'file':str(p.relative_to(local)),'code':code,'label':0}
            out.write(json.dumps(rec, ensure_ascii=False)+'\n')
    out.close()
