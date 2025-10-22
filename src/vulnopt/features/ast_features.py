"""Helpers for converting Python ASTs into numeric features."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List

from .ast_graph import python_ast_graph


def ast_node_counter(code: str) -> Counter[str]:
    """Return a counter of AST node types present in ``code``."""

    G = python_ast_graph(code)
    counter: Counter[str] = Counter()
    for _, data in G.nodes(data=True):
        node_type = data.get("type")
        if node_type:
            counter[node_type] += 1
    return counter


def build_vocab(counters: Iterable[Counter[str]], limit: int = 256) -> List[str]:
    """Create a vocabulary of the most common node types."""

    global_counts: Counter[str] = Counter()
    for counter in counters:
        global_counts.update(counter)

    vocab = [node for node, _ in global_counts.most_common(limit)]
    if not vocab:
        vocab = ["Module"]  # sensible default for empty/invalid code snippets
    return vocab


def vectorize(counter: Counter[str], vocab: Iterable[str]) -> List[float]:
    """Convert a node counter into a dense feature vector."""

    vocab_list = list(vocab)
    if not vocab_list:
        return [0.0]
    return [float(counter.get(token, 0)) for token in vocab_list]
