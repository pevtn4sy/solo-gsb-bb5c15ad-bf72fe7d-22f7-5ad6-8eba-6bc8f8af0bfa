"""Shortest Markdown reference-chain queries over one import batch.

The cross-document reference graph is produced by ``analyzer.analyze``, so a
query reuses the exact same Markdown parsing and relative-link resolution
rules as validation (code-block skipping, inline-code masking, reference
definitions only counted when used, percent-decoding, fragment-only links
not treated as edges). The query only reads the graph: it never emits or
persists diagnostics, so validation results are unaffected.
"""
from __future__ import annotations

from .analyzer import analyze


def shortest_reference_chain(
    documents: dict[str, str], source: str, target: str
) -> dict:
    """Find the shortest reference chain from ``source`` to ``target``.

    Breadth-first search over the document graph, expanded one distance
    level at a time. When several chains have the same length, the one whose
    full path sequence is lexicographically smallest is selected; the choice
    does not depend on the order in which a level is expanded. Nodes settled
    at a shorter distance bound the traversal, so cycles (including self
    loops) cannot make the search non-terminating.

    Returns ``source``/``target`` echoes plus ``found``, ``path`` and
    ``length``. An unreachable target yields ``found=False``, an empty path
    and ``length=None``.
    """
    graph = analyze(documents).graph

    # best[v] is the lexicographically smallest shortest path reaching v.
    best: dict[str, list[str]] = {source: [source]}
    frontier = [source]
    while frontier and target not in best:
        # Candidates reachable at the next distance level. Equal-length
        # prefixes compete here; nodes already in `best` are settled at a
        # strictly shorter distance and must not be re-entered.
        candidates: dict[str, list[str]] = {}
        for node in frontier:
            prefix = best[node]
            for nxt in graph.get(node, ()):  # graph edges target imported docs
                if nxt in best:
                    continue
                candidate = prefix + [nxt]
                current = candidates.get(nxt)
                if current is None or candidate < current:
                    candidates[nxt] = candidate
        if not candidates:
            break
        best.update(candidates)
        frontier = sorted(candidates)  # deterministic expansion order

    chain = best.get(target)
    if chain is None:
        return {
            "source": source,
            "target": target,
            "found": False,
            "path": [],
            "length": None,
        }
    return {
        "source": source,
        "target": target,
        "found": True,
        "path": chain,
        "length": len(chain) - 1,
    }
