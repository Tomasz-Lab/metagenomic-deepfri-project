from functools import lru_cache
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class WangSemantic:
    """
    Wang (2007) semantic similarity on a GO DAG.

    - godag: GO DAG object (e.g. goatools.obo_parser.GODag)
    - w_is_a: weight for 'is_a' edges
    - w_part_of: weight for 'part_of' edges

    S-values:
      S(term) is a dict: ancestor_term -> contribution in [0,1],
      computed by propagating upward with weights along all paths and taking
      the maximum contribution per ancestor.

    Similarity:
      sim(t1, t2) = sum_{a in A∩B} (S1[a] + S2[a]) / (sum(S1) + sum(S2))
    """

    def __init__(self, godag, w_is_a: float = 0.8, w_part_of: float = 0.6):
        self.godag = godag
        self.w_edge: Dict[str, float] = {
            "is_a": float(w_is_a),
            "part_of": float(w_part_of),
        }
        # cache: term -> dict(ancestor -> S-value)
        self._s_cache: Dict[str, Dict[str, float]] = {}
        # cache: term -> sum of S-values
        self._s_sum_cache: Dict[str, float] = {}

    # ---------- internal helpers ----------

    def _get_term(self, go_id: str) -> Any:
        """Return GO term object or None if not in DAG / obsolete."""
        term = self.godag.get(go_id)
        if term is None:
            return None
        # goatools terms often have .is_obsolete flag
        if getattr(term, "is_obsolete", False):
            return None
        return term

    def _parents_with_types(self, go_id: str) -> List[Tuple[str, str]]:
        """
        Return list of (edge_type, parent_id) for a GO term.

        Only 'is_a' and 'part_of' are used, with weights in self.w_edge.
        """
        term = self._get_term(go_id)
        if term is None:
            return []

        out: List[Tuple[str, str]] = []

        # is_a parents (goatools: .parents is a set of GO term objects)
        for p in getattr(term, "parents", []):
            pid = getattr(p, "item_id", None) or str(p)
            out.append(("is_a", pid))

        # part_of via relationships (if present in OBO)
        rel = getattr(term, "relationships", None)
        if rel and "part_of" in rel:
            for p in rel["part_of"]:
                pid = getattr(p, "item_id", None) or str(p)
                out.append(("part_of", pid))

        # deduplicate: prefer explicit 'part_of' weight if duplicated
        dedup: Dict[str, str] = {}
        for etype, pid in out:
            if pid not in dedup or etype == "part_of":
                dedup[pid] = etype

        return [
            (etype, pid)
            for pid, etype in dedup.items()
            if etype in self.w_edge
        ]

    # ---------- S-values ----------

    def s_values(self, go_id: str) -> Dict[str, float]:
        """
        Return S-value map for a term: ancestor -> S in [0,1].

        Includes the term itself with S=1.0.
        Cached for reuse.
        """
        if go_id in self._s_cache:
            return self._s_cache[go_id]

        term = self._get_term(go_id)
        if term is None:
            self._s_cache[go_id] = {}
            self._s_sum_cache[go_id] = 0.0
            return self._s_cache[go_id]

        # upward propagation (DFS/BFS) with max-product aggregation
        Sv: Dict[str, float] = {go_id: 1.0}
        stack = [go_id]

        while stack:
            cur = stack.pop()
            cur_val = Sv[cur]
            for etype, parent in self._parents_with_types(cur):
                w = self.w_edge.get(etype, 0.0)
                if w <= 0.0:
                    continue
                val = cur_val * w
                if val <= 0.0:
                    continue
                # keep max contribution per ancestor
                if val > Sv.get(parent, 0.0):
                    Sv[parent] = val
                    stack.append(parent)

        self._s_cache[go_id] = Sv
        self._s_sum_cache[go_id] = float(sum(Sv.values()))
        return Sv

    def _s_sum(self, go_id: str) -> float:
        """Sum of S-values for a term (cached)."""
        if go_id not in self._s_sum_cache:
            _ = self.s_values(go_id)
        return self._s_sum_cache.get(go_id, 0.0)

    # ---------- similarity ----------

    @lru_cache(maxsize=200_000)
    def sim(self, t1: str, t2: str) -> float:
        """
        Wang similarity between two GO terms in [0,1].

        - 0.0 if either term is missing/obsolete
        - 1.0 if terms are identical
        """
        # enforce symmetry in cache: sim(t1, t2) == sim(t2, t1)
        if t2 < t1:
            t1, t2 = t2, t1

        if t1 == t2:
            # ensure it's a valid term
            if self._get_term(t1) is None:
                return 0.0
            return 1.0

        # compute S-values
        s1 = self.s_values(t1)
        s2 = self.s_values(t2)
        if not s1 or not s2:
            return 0.0

        inter = set(s1).intersection(s2)
        if not inter:
            return 0.0

        num = sum(s1[a] + s2[a] for a in inter)
        den = self._s_sum(t1) + self._s_sum(t2)
        return float(num / den) if den > 0.0 else 0.0

def semantic_cohesion_for_terms(go_list, wang):
    """
    go_list: iterable of GO IDs
    wang:    WangSemantic instance
    """
    terms = list(dict.fromkeys(go_list))
    n = len(terms)
    if n < 2:
        return np.nan

    total = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            total += wang.sim(terms[i], terms[j])
            count += 1

    return total / count

def per_protein_cohesion(
    df,
    aspect,
    wang,
    aspect_col=None,
    protein_col="Protein",
    go_col="go_term",
):
    _asp_norm = str(aspect).lower()

    # detect aspect column (case-insensitive)
    if aspect_col is None:
        for _col in df.columns:
            if _col.lower() == "aspect":
                aspect_col = _col
                break
    if aspect_col is None:
        raise ValueError("No aspect column found (case-insensitive match for 'aspect').")

    _sub = df[df[aspect_col].astype(str).str.lower() == _asp_norm]

    _groups = (
        _sub.groupby(protein_col)[go_col]
        .apply(lambda s: list(s.dropna().unique()))
    )

    _result = _groups.apply(lambda _lst: semantic_cohesion_for_terms(_lst, wang))
    _result.name = f"cohesion_{_asp_norm}"
    return _result

def n_go_per_protein(
    df,
    aspect,
    aspect_col=None,
    protein_col="Protein",
    go_col="go_term",
):
    """
    Number of distinct GO terms per protein for a given aspect.
    """
    _asp_norm = str(aspect).lower()

    if aspect_col is None:
        for _col in df.columns:
            if _col.lower() == "aspect":
                aspect_col = _col
                break
    if aspect_col is None:
        raise ValueError("No aspect column found (case-insensitive match for 'aspect').")

    _sub = df[df[aspect_col].astype(str).str.lower() == _asp_norm]
    return _sub.groupby(protein_col)[go_col].nunique()


def boxplot_cohesion(_ax, _coh_dict, _title, _ylabel=None):
    """
    _ax:      matplotlib axis to draw on
    _coh_dict: dict label -> Series (per-protein cohesion)
    _title:   subplot title
    _ylabel:  optional y-axis label
    """
    _labels = list(_coh_dict.keys())
    _data   = [_coh_dict[_lab].dropna() for _lab in _labels]

    _bp = _ax.boxplot(_data, tick_labels=_labels, sym=".")

    # annotate medians
    for _i, _median_line in enumerate(_bp["medians"], start=1):
        _x, _y = _median_line.get_xdata(), _median_line.get_ydata()
        _median_val = _y.mean()

        _ax.text(
            _i,
            _median_val,
            f"{_median_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _ax.set_title(_title)
    if _ylabel is not None:
        _ax.set_ylabel(_ylabel)
    _ax.set_ylim(0.0, 1.0)

def scatter_terms_vs_cohesion(df, coh_series, aspect, method_label):
    n_terms = n_go_per_protein(df, aspect)
    # align index
    common_proteins = coh_series.index.intersection(n_terms.index)
    x = n_terms.loc[common_proteins]
    y = coh_series.loc[common_proteins]

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=5, alpha=0.3)
    plt.xlabel("#GO terms for protein")
    plt.ylabel("Semantic cohesion")
    plt.ylim(0, 1)
    plt.title(f"{method_label} – {aspect}: terms vs cohesion")
    plt.show()

def go_terms_per_protein(
    df,
    aspect=None,
    protein_col="Protein",
    go_col="go_term",
    aspect_col=None,
):
    sub = df

    # Find aspect column if needed
    if aspect is not None:
        if aspect_col is None:
            for col in df.columns:
                if col.lower() == "aspect":
                    aspect_col = col
                    break

        aspect_norm = str(aspect).lower()

        sub = sub[sub[aspect_col].astype(str).str.lower() == aspect_norm]

    # group GO terms per protein
    grouped = (
        sub.groupby(protein_col)[go_col]
        .apply(lambda s: set(s.dropna().unique()))
    )
    return grouped.to_dict()

def core_and_unique_per_protein(**Gdicts):

    names = list(Gdicts.keys())
    dicts = list(Gdicts.values())

    # union of all proteins across all sources
    proteins = set()
    for G in dicts:
        proteins.update(G.keys())

    core = {}
    uniques = {name: {} for name in names}

    for p in proteins:
        # gather per-source sets for this protein
        sets_for_p = {
            name: Gdicts[name].get(p, set())
            for name in names
        }

        # core: intersection across all sources
        all_sets = list(sets_for_p.values())
        if len(all_sets) == 1:
            c = set(all_sets[0])  # copy
        else:
            # intersection over all sets (if any non-empty)
            c = set.intersection(*[s for s in all_sets]) if all_sets else set()
        core[p] = c

        # uniques per source: S_i - union(other S_j)
        for name in names:
            S_i = sets_for_p[name]
            others_union = set()
            for other_name, S_j in sets_for_p.items():
                if other_name != name:
                    others_union |= S_j
            uniques[name][p] = S_i - others_union

    return core, uniques

def mean_sim_unique_to_core(unique_terms, core_terms, wang):
    """
    unique_terms: set of GO IDs (for one method, one protein)
    core_terms  : set of GO IDs (intersection baseline for that protein)
    Returns: mean of max similarity per unique term, or np.nan if empty or no core.
    """
    if not unique_terms or not core_terms:
        return np.nan

    core_list = list(core_terms)
    vals = []

    for u in unique_terms:
        # best similarity of unique term u to any core term
        max_sim = max(wang.sim(u, c) for c in core_list)
        vals.append(max_sim)

    return float(np.mean(vals)) if vals else np.nan

def unique_to_core_similarity(core, unique_dict, wang):
    """
    core       : dict Protein -> set(core_go_terms)
    unique_dict: dict Protein -> set(unique_go_terms) for one method
    Returns: Series indexed by Protein with mean similarity of unique terms to core.
    """
    scores = {}
    for p, uniq in unique_dict.items():
        c = core.get(p, set())
        scores[p] = mean_sim_unique_to_core(uniq, c, wang)

    s = pd.Series(scores, name="unique_sim_to_core")
    return s