import typer
from pathlib import Path
from vulnopt.data import nvd_fetch, cve_map, repo_miner, denoise
from vulnopt import train, evaluate, infer

app = typer.Typer()

@app.command()
def fetch_nvd(years: str, out: Path):
    start, end = [int(x) for x in years.split('-')]
    out.parent.mkdir(parents=True, exist_ok=True)
    nvd_fetch.fetch_nvd(start, end, out)

@app.command()
def map_cve(nvd: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    cve_map.map_cve_references(nvd, out)

@app.command()
def mine_repos(map: Path, out: Path, cache: Path = Path('repos/')):
    out.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    repo_miner.mine_repos(map, out, cache)

@app.command()
def denoise_cmd(inp: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    denoise.denoise_jsonl(inp, out)

@app.command()
def train_cmd(data: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    train.train_main(data, out)

@app.command()
def evaluate_cmd(ckpt: Path, data: Path):
    evaluate.eval_main(ckpt, data)

@app.command()
def infer_cmd(ckpt: Path, path: Path, out: Path):
    infer.infer_main(ckpt, path, out, None)

if __name__ == '__main__':
    app()
