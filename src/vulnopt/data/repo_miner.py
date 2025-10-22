"""Utilities for building a labelled dataset from mined repositories.

The previous implementation emitted only negative (non vulnerable) samples.
To train a detector we need positive examples as well.  The CVE map generated
by :func:`vulnopt.data.cve_map.map_cve_references` contains links to GitHub
commits that fix known vulnerabilities.  By pulling those commits we can
recover the vulnerable version of the files from the parent commit and the
patched version from the fixing commit.  This function now emits both versions
labelled respectively ``1`` (vulnerable) and ``0`` (patched).

Only Python files are processed because the current feature extraction logic
is Python specific.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

from git import Repo, GitCommandError


def _ensure_repo(repo_name: str, cache_dir: Path) -> Repo | None:
    """Clone the repository if necessary and return the :class:`Repo` object."""

    local = cache_dir / repo_name.replace("/", "__")
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            Repo.clone_from(f"https://github.com/{repo_name}", local)
        except GitCommandError:
            return None
    repo = Repo(local)
    try:
        repo.git.fetch("--all", prune=True)
    except GitCommandError:
        # Fetch errors are not fatal; the commit might already exist locally.
        pass
    return repo


def _iter_commit_samples(repo: Repo, repo_name: str, commit_sha: str, cve: str | None) -> Iterable[dict]:
    """Yield vulnerable and fixed samples for a CVE fixing commit."""

    try:
        commit = repo.commit(commit_sha)
    except (ValueError, GitCommandError):
        return

    if not commit.parents:
        return

    parent = commit.parents[0]
    diff_index = parent.diff(commit, create_patch=False)

    for diff in diff_index:
        a_path = diff.a_path or diff.b_path
        b_path = diff.b_path or diff.a_path
        # Focus on Python source files
        if not a_path or not a_path.endswith(".py") and (not b_path or not b_path.endswith(".py")):
            continue

        vuln_code = None
        patched_code = None

        if diff.a_blob is not None and diff.a_path:
            try:
                vuln_code = repo.git.show(f"{parent.hexsha}:{diff.a_path}")
            except GitCommandError:
                vuln_code = None
        if diff.b_blob is not None and diff.b_path:
            try:
                patched_code = repo.git.show(f"{commit.hexsha}:{diff.b_path}")
            except GitCommandError:
                patched_code = None

        if vuln_code:
            yield {
                "id": f"{repo_name}@{commit.hexsha}:{diff.a_path}:vuln",
                "repo": repo_name,
                "commit": commit.hexsha,
                "file": diff.a_path,
                "code": vuln_code,
                "label": 1,
                "cve": cve,
            }
        if patched_code:
            yield {
                "id": f"{repo_name}@{commit.hexsha}:{diff.b_path}:fix",
                "repo": repo_name,
                "commit": commit.hexsha,
                "file": diff.b_path,
                "code": patched_code,
                "label": 0,
                "cve": cve,
            }


def mine_repos(map_csv: Path, out_jsonl: Path, cache_dir: Path):
    """Build a JSONL dataset from the CVE/repository mapping."""

    rows = [l.strip() for l in map_csv.read_text().splitlines() if l.strip()]
    with open(out_jsonl, "w", encoding="utf8") as out:
        for r in rows:
            try:
                record = json.loads(r)
            except json.JSONDecodeError:
                continue

            repo_name = record.get("repo")
            commit_sha = record.get("commit")
            if not repo_name or not commit_sha:
                continue

            repo = _ensure_repo(repo_name, cache_dir)
            if repo is None:
                continue

            for sample in _iter_commit_samples(repo, repo_name, commit_sha, record.get("cve")):
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
