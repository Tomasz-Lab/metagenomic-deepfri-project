"""GO-term propagation.

Public surface:
- ``parse_go_obo(obo_path)`` — minimal dependency-free OBO parser.
- ``GOPropagator`` — class that parses OBO + builds the ancestor closure once,
  then propagates many DataFrames efficiently. Use this when you have multiple
  inputs (e.g. several predictors) sharing the same ontology.
- ``propagate_go_annotations(df, obo_path, ...)`` — pandas-in / pandas-out
  back-compat wrapper preserving the original output schema.

The core does propagation as a polars join + group_by, so it scales to tens
of millions of rows without the Python-dict accumulator of the original
implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

import polars as pl


GO_ROOTS = {"bp": "GO:0008150", "mf": "GO:0003674", "cc": "GO:0005575"}
ROOT_IDS: Set[str] = set(GO_ROOTS.values())

ScoreAgg = Literal["max", "noisy_or", "sum", "mean"]
ObsoleteMode = Literal["drop", "map"]
ObsoletePrefer = Literal["replaced_by", "consider"]

AllowTerms = Union[str, Path, Iterable[str], pl.Series, None]


@dataclass(frozen=True)
class GOTerm:
    go_id: str
    name: str
    namespace: str
    parents_is_a: Tuple[str, ...]
    parents_part_of: Tuple[str, ...]
    is_obsolete: bool
    replaced_by: Tuple[str, ...]
    consider: Tuple[str, ...]


def _ns_to_aspect(ns: str) -> Optional[str]:
    return {
        "biological_process": "bp",
        "molecular_function": "mf",
        "cellular_component": "cc",
    }.get(ns)


def parse_go_obo(obo_path: Union[str, Path]) -> Dict[str, GOTerm]:
    terms: Dict[str, GOTerm] = {}
    current: Dict[str, object] = {}
    in_term = False

    def flush():
        nonlocal current, in_term
        if not in_term:
            return
        go_id = current.get("id")
        if not go_id:
            current = {}
            in_term = False
            return
        terms[str(go_id)] = GOTerm(
            go_id=str(go_id),
            name=str(current.get("name", "")),
            namespace=str(current.get("namespace", "")),
            parents_is_a=tuple(current.get("is_a", ())),
            parents_part_of=tuple(current.get("part_of", ())),
            is_obsolete=bool(current.get("is_obsolete", False)),
            replaced_by=tuple(current.get("replaced_by", ())),
            consider=tuple(current.get("consider", ())),
        )
        current = {}
        in_term = False

    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line == "[Term]":
                flush()
                in_term = True
                current = {"is_a": [], "part_of": [], "replaced_by": [], "consider": []}
                continue
            if not in_term:
                continue

            if line.startswith("id: "):
                current["id"] = line.split("id: ", 1)[1].strip()
            elif line.startswith("name: "):
                current["name"] = line.split("name: ", 1)[1].strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line.split("namespace: ", 1)[1].strip()
            elif line.startswith("is_obsolete: "):
                current["is_obsolete"] = (
                    line.split("is_obsolete: ", 1)[1].strip().lower() == "true"
                )
            elif line.startswith("is_a: "):
                parent = line.split("is_a: ", 1)[1].split("!", 1)[0].strip()
                current["is_a"].append(parent)
            elif line.startswith("relationship: "):
                rest = line.split("relationship: ", 1)[1].strip()
                if rest.startswith("part_of "):
                    parent = rest.split("part_of ", 1)[1].split("!", 1)[0].strip()
                    current["part_of"].append(parent)
            elif line.startswith("replaced_by: "):
                rid = line.split("replaced_by: ", 1)[1].strip()
                current["replaced_by"].append(rid)
            elif line.startswith("consider: "):
                cid = line.split("consider: ", 1)[1].strip()
                current["consider"].append(cid)

    flush()
    return terms


def _normalize_relations(relations: Union[str, Iterable[str]]) -> Tuple[str, ...]:
    if isinstance(relations, str):
        relations = (relations,)
    rels = tuple(relations)
    valid = {"is_a", "part_of"}
    bad = [r for r in rels if r not in valid]
    if bad:
        raise ValueError(f"Unknown relation(s): {bad}. Must be subset of {valid}.")
    if not rels:
        raise ValueError("relations must contain at least one of 'is_a', 'part_of'.")
    return rels


def _load_allow_terms(
    allow_terms: AllowTerms,
    *,
    column: str = "go_term",
) -> Optional[Set[str]]:
    """Accept a path (csv/tsv/txt), an iterable of strings, or a polars Series."""
    if allow_terms is None:
        return None
    if isinstance(allow_terms, (str, Path)):
        path = Path(allow_terms)
        if not path.exists():
            raise FileNotFoundError(f"Allow-terms file not found: {path}")
        if path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pl.read_csv(path, separator=sep)
            if column not in df.columns:
                raise ValueError(
                    f"Allow-terms file {path} missing column {column!r}. "
                    f"Found: {df.columns}"
                )
            return set(df[column].cast(pl.Utf8).str.strip_chars().to_list())
        # plain text: one term per line
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    if isinstance(allow_terms, pl.Series):
        return set(allow_terms.cast(pl.Utf8).str.strip_chars().to_list())
    return {str(t).strip() for t in allow_terms}


class GOPropagator:
    """Parse OBO once, reuse closure + meta tables across many DataFrames."""

    def __init__(
        self,
        obo_path: Union[str, Path],
        *,
        relations: Union[str, Iterable[str]] = ("is_a",),
        exclude_roots: bool = True,
        obsolete_mode: ObsoleteMode = "drop",
        obsolete_prefer: ObsoletePrefer = "replaced_by",
    ):
        if obsolete_mode not in ("drop", "map"):
            raise ValueError("obsolete_mode must be 'drop' or 'map'.")
        if obsolete_prefer not in ("replaced_by", "consider"):
            raise ValueError("obsolete_prefer must be 'replaced_by' or 'consider'.")

        self.terms: Dict[str, GOTerm] = parse_go_obo(obo_path)
        self.relations = _normalize_relations(relations)
        self.exclude_roots = exclude_roots
        self.obsolete_mode: ObsoleteMode = obsolete_mode
        self.obsolete_prefer: ObsoletePrefer = obsolete_prefer

        self._closure: pl.DataFrame = self._build_closure()
        self._meta: pl.DataFrame = self._build_meta()
        self._obs_map: pl.DataFrame = self._build_obsolete_map()
        self._known_terms: Set[str] = {
            tid for tid, t in self.terms.items() if not t.is_obsolete
        }
        self._obsolete_terms: Set[str] = {
            tid for tid, t in self.terms.items() if t.is_obsolete
        }

    def _ancestors_of(self, term_id: str) -> Set[str]:
        """All non-obsolete ancestors of a term under the configured relations."""
        term = self.terms.get(term_id)
        if term is None or term.is_obsolete:
            return set()
        use_is_a = "is_a" in self.relations
        use_part_of = "part_of" in self.relations

        seen: Set[str] = set()
        stack: List[str] = []
        if use_is_a:
            stack.extend(term.parents_is_a)
        if use_part_of:
            stack.extend(term.parents_part_of)

        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            if self.exclude_roots and pid in ROOT_IDS:
                continue
            parent = self.terms.get(pid)
            if parent is None or parent.is_obsolete:
                continue
            seen.add(pid)
            if use_is_a:
                stack.extend(parent.parents_is_a)
            if use_part_of:
                stack.extend(parent.parents_part_of)
        return seen

    def _build_closure(self) -> pl.DataFrame:
        terms_col: List[str] = []
        ancestors_col: List[str] = []
        for tid, term in self.terms.items():
            if term.is_obsolete:
                continue
            for anc in self._ancestors_of(tid):
                terms_col.append(tid)
                ancestors_col.append(anc)
        return pl.DataFrame(
            {"_seed_term": terms_col, "_anc_term": ancestors_col},
            schema={"_seed_term": pl.Utf8, "_anc_term": pl.Utf8},
        )

    def _build_meta(self) -> pl.DataFrame:
        tids: List[str] = []
        names: List[str] = []
        aspects: List[Optional[str]] = []
        for tid, term in self.terms.items():
            if term.is_obsolete:
                continue
            tids.append(tid)
            names.append(term.name)
            aspects.append(_ns_to_aspect(term.namespace))
        return pl.DataFrame(
            {"_term": tids, "_name": names, "_aspect": aspects},
            schema={"_term": pl.Utf8, "_name": pl.Utf8, "_aspect": pl.Utf8},
        )

    def _build_obsolete_map(self) -> pl.DataFrame:
        """Obsolete → best replacement (only when obsolete_mode='map'). Each obsolete
        term contributes ONE row pointing to its preferred non-obsolete replacement."""
        if self.obsolete_mode != "map":
            return pl.DataFrame(
                {"_obsolete": [], "_replacement": []},
                schema={"_obsolete": pl.Utf8, "_replacement": pl.Utf8},
            )
        obsolete: List[str] = []
        replacement: List[str] = []
        for tid, term in self.terms.items():
            if not term.is_obsolete:
                continue
            primary = (
                term.replaced_by
                if self.obsolete_prefer == "replaced_by"
                else term.consider
            )
            secondary = (
                term.consider
                if self.obsolete_prefer == "replaced_by"
                else term.replaced_by
            )
            for cand in list(primary) + list(secondary):
                c = self.terms.get(cand)
                if c is not None and not c.is_obsolete:
                    obsolete.append(tid)
                    replacement.append(cand)
                    break
        return pl.DataFrame(
            {"_obsolete": obsolete, "_replacement": replacement},
            schema={"_obsolete": pl.Utf8, "_replacement": pl.Utf8},
        )

    def _aggregate_pairs(
        self,
        sub: pl.DataFrame,
        *,
        has_score: bool,
        score_agg: ScoreAgg,
    ) -> pl.DataFrame:
        """Deduplicate (_p, _t) pairs, aggregating duplicate scores via score_agg."""
        if not has_score:
            return sub.unique(["_p", "_t"])
        if score_agg == "max":
            score_expr = pl.col("_s").max()
        elif score_agg == "sum":
            score_expr = pl.col("_s").sum()
        elif score_agg == "mean":
            score_expr = pl.col("_s").mean()
        elif score_agg == "noisy_or":
            score_expr = 1.0 - (1.0 - pl.col("_s")).log().sum().exp()
        else:  # pragma: no cover -- validated upstream
            raise ValueError(score_agg)
        return sub.group_by(["_p", "_t"]).agg(score_expr.alias("_s"))

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def propagate(
        self,
        df: pl.DataFrame,
        *,
        protein_col: str = "protein",
        term_col: str = "go_term",
        score_col: Optional[str] = "score",
        keep_original_terms: bool = True,
        score_min: Optional[float] = None,
        score_agg: ScoreAgg = "max",
        allow_terms: AllowTerms = None,
        allow_terms_col: str = "go_term",
        ic_df: Optional[pl.DataFrame] = None,
        ic_term_col: str = "go_term",
        ic_value_col: str = "IC",
        round_decimals: Optional[int] = 3,
        leaf_only: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Return (propagated_df, stats_df).

        Output columns: ``protein``, ``go_term``, optional ``score``, ``name``,
        ``aspect``, ``ic``, ``propagated``.

        When ``leaf_only=True``, propagation is skipped and the result is the
        per-protein leaf set: any term that is an ancestor of another term in
        the same protein's annotation set is dropped. The ``propagated`` column
        is False everywhere in this mode.
        """
        if score_agg not in ("max", "noisy_or", "sum", "mean"):
            raise ValueError(
                "score_agg must be one of: 'max', 'noisy_or', 'sum', 'mean'."
            )

        has_score = score_col is not None and score_col in df.columns
        if score_min is not None and not has_score:
            raise ValueError(
                f"score_min provided but score_col {score_col!r} missing."
            )
        if score_agg in ("noisy_or", "sum", "mean") and not has_score:
            raise ValueError(
                f"score_agg={score_agg!r} requires a score column."
            )
        if leaf_only and not keep_original_terms:
            raise ValueError(
                "leaf_only=True requires keep_original_terms=True "
                "(otherwise the result is empty)."
            )

        # --- 1. select, auto-explode list-typed columns, drop nulls ---
        # If term/score arrive as parallel List columns (e.g. from a
        # per-protein groupby upstream), explode them together first.
        cols = [protein_col, term_col] + ([score_col] if has_score else [])
        sub = df.select(cols).rename({protein_col: "_p", term_col: "_t"})
        if has_score:
            sub = sub.rename({score_col: "_s"})

        schema = sub.schema
        explode_cols: List[str] = []
        if isinstance(schema["_t"], pl.List):
            explode_cols.append("_t")
        if has_score and isinstance(schema["_s"], pl.List):
            explode_cols.append("_s")
        if explode_cols:
            sub = sub.explode(explode_cols)

        sub = sub.drop_nulls()
        sub = sub.with_columns(pl.col("_t").cast(pl.Utf8))
        if has_score:
            sub = sub.with_columns(pl.col("_s").cast(pl.Float64, strict=False))
            sub = sub.drop_nulls("_s")
        n_input_rows = sub.height
        n_rows_nonnull = sub.height

        # --- 2. score filter -----------------------------------------
        n_rows_dropped_score = 0
        if has_score and score_min is not None:
            before = sub.height
            sub = sub.filter(pl.col("_s") >= float(score_min))
            n_rows_dropped_score = before - sub.height

        # --- 3. allow-list filter -----------------------------------
        allow_set = _load_allow_terms(allow_terms, column=allow_terms_col)
        n_rows_dropped_allow = 0
        if allow_set is not None:
            before = sub.height
            sub = sub.filter(pl.col("_t").is_in(list(allow_set)))
            n_rows_dropped_allow = before - sub.height
        n_rows_after_filters = sub.height
        input_unique_terms = sub["_t"].n_unique()

        # --- 4. classify rows for stats -----------------------------
        is_go_like = sub.with_columns(pl.col("_t").str.starts_with("GO:").alias("_go"))
        n_rows_go_like = int(is_go_like["_go"].sum())
        go_like = is_go_like.filter(pl.col("_go")).drop("_go")
        n_rows_known = int(go_like["_t"].is_in(list(self.terms.keys())).sum())
        n_rows_unknown = go_like.height - n_rows_known
        unknown_terms = go_like.filter(~pl.col("_t").is_in(list(self.terms.keys())))
        n_unique_unknown_terms = unknown_terms["_t"].n_unique()

        known = go_like.filter(pl.col("_t").is_in(list(self.terms.keys())))
        obsolete_mask = known["_t"].is_in(list(self._obsolete_terms))
        n_rows_obsolete = int(obsolete_mask.sum())
        n_rows_nonobsolete = known.height - n_rows_obsolete
        n_unique_obsolete_terms = (
            known.filter(obsolete_mask)["_t"].n_unique() if n_rows_obsolete else 0
        )

        # --- 5. resolve obsoletes -----------------------------------
        # Keep only GO-like rows from here on.
        sub = sub.filter(pl.col("_t").str.starts_with("GO:"))

        n_rows_obsolete_mapped = 0
        if self.obsolete_mode == "map" and self._obs_map.height > 0:
            sub = sub.join(
                self._obs_map, left_on="_t", right_on="_obsolete", how="left"
            )
            mapped_mask = sub["_replacement"].is_not_null()
            n_rows_obsolete_mapped = int(
                (
                    sub["_t"].is_in(list(self._obsolete_terms)) & mapped_mask
                ).sum()
            )
            sub = sub.with_columns(
                pl.when(pl.col("_replacement").is_not_null())
                .then(pl.col("_replacement"))
                .otherwise(pl.col("_t"))
                .alias("_t")
            ).drop("_replacement")

        # Drop unknown + remaining obsolete terms
        sub = sub.filter(pl.col("_t").is_in(list(self._known_terms)))
        if self.exclude_roots:
            sub = sub.filter(~pl.col("_t").is_in(list(ROOT_IDS)))

        n_leaf_only_dropped = 0
        if leaf_only:
            # --- 6/7 (leaf-only): aggregate duplicates, then drop non-leaves
            sub_agg = self._aggregate_pairs(sub, has_score=has_score, score_agg=score_agg)
            non_leaves = sub_agg.join(
                self._closure, left_on="_t", right_on="_seed_term", how="inner"
            ).select(pl.col("_p"), pl.col("_anc_term").alias("_t"))
            before_leaf = sub_agg.height
            leaves = sub_agg.join(non_leaves, on=["_p", "_t"], how="anti")
            n_leaf_only_dropped = before_leaf - leaves.height

            select_cols = [
                pl.col("_p"),
                pl.col("_t").alias("_out_t"),
                pl.lit(False).alias("_propagated"),
            ]
            if has_score:
                select_cols.insert(2, pl.col("_s"))
            grouped = leaves.select(select_cols)
        else:
            # --- 6. originals + propagated --------------------------
            if has_score:
                originals = sub.select(
                    pl.col("_p"),
                    pl.col("_t").alias("_out_t"),
                    pl.col("_s"),
                    pl.lit(False).alias("_propagated"),
                )
                propagated = sub.join(
                    self._closure, left_on="_t", right_on="_seed_term", how="inner"
                ).select(
                    pl.col("_p"),
                    pl.col("_anc_term").alias("_out_t"),
                    pl.col("_s"),
                    pl.lit(True).alias("_propagated"),
                )
            else:
                originals = sub.select(
                    pl.col("_p"),
                    pl.col("_t").alias("_out_t"),
                    pl.lit(False).alias("_propagated"),
                )
                propagated = sub.join(
                    self._closure, left_on="_t", right_on="_seed_term", how="inner"
                ).select(
                    pl.col("_p"),
                    pl.col("_anc_term").alias("_out_t"),
                    pl.lit(True).alias("_propagated"),
                )

            if keep_original_terms:
                both = pl.concat([originals, propagated], how="vertical_relaxed")
            else:
                both = propagated

            # --- 7. groupby aggregate -------------------------------
            if has_score:
                agg_exprs = [pl.col("_propagated").min().alias("_propagated")]
                if score_agg == "max":
                    agg_exprs.append(pl.col("_s").max().alias("_s"))
                elif score_agg == "sum":
                    agg_exprs.append(pl.col("_s").sum().alias("_s"))
                elif score_agg == "mean":
                    agg_exprs.append(pl.col("_s").mean().alias("_s"))
                elif score_agg == "noisy_or":
                    # 1 - prod(1 - s) via log/exp; safe for s in [0, 1].
                    agg_exprs.append(
                        (
                            1.0
                            - (1.0 - pl.col("_s")).log().sum().exp()
                        ).alias("_s")
                    )
                grouped = both.group_by(["_p", "_out_t"]).agg(agg_exprs)
            else:
                grouped = both.group_by(["_p", "_out_t"]).agg(
                    pl.col("_propagated").min().alias("_propagated")
                )

        # --- 8. decorate with name/aspect/IC ------------------------
        out = grouped.join(self._meta, left_on="_out_t", right_on="_term", how="left")
        if ic_df is not None:
            if ic_term_col not in ic_df.columns or ic_value_col not in ic_df.columns:
                raise ValueError(
                    f"ic_df must have columns {ic_term_col!r}, {ic_value_col!r}."
                )
            ic = (
                ic_df.select(
                    pl.col(ic_term_col).cast(pl.Utf8).alias("_term"),
                    pl.col(ic_value_col).cast(pl.Float64).alias("_ic"),
                )
                .drop_nulls()
                .group_by("_term")
                .agg(pl.col("_ic").max())
            )
            out = out.join(ic, left_on="_out_t", right_on="_term", how="left")
        else:
            out = out.with_columns(pl.lit(None).cast(pl.Float64).alias("_ic"))

        # --- 9. rename + sort + round -------------------------------
        rename = {
            "_p": "protein",
            "_out_t": "go_term",
            "_propagated": "propagated",
            "_name": "name",
            "_aspect": "aspect",
            "_ic": "ic",
        }
        if has_score:
            rename["_s"] = "score"
        out = out.rename(rename)

        sort_cols = ["protein", "propagated"]
        sort_desc = [False, False]
        if has_score:
            sort_cols.append("score")
            sort_desc.append(True)
        sort_cols.append("go_term")
        sort_desc.append(False)
        out = out.sort(sort_cols, descending=sort_desc)

        if round_decimals is not None:
            if has_score:
                out = out.with_columns(pl.col("score").round(round_decimals))
            out = out.with_columns(pl.col("ic").round(round_decimals))

        # column order
        cols = ["protein", "go_term"]
        if has_score:
            cols.append("score")
        cols += ["name", "aspect", "ic", "propagated"]
        out = out.select(cols)

        # --- 10. stats ----------------------------------------------
        stats_df = self._build_stats(
            n_input_rows=n_input_rows,
            n_rows_nonnull=n_rows_nonnull,
            n_rows_dropped_score=n_rows_dropped_score,
            n_rows_dropped_allow=n_rows_dropped_allow,
            n_rows_after_filters=n_rows_after_filters,
            input_unique_terms=input_unique_terms,
            n_rows_go_like=n_rows_go_like,
            n_rows_known=n_rows_known,
            n_rows_unknown=n_rows_unknown,
            n_unique_unknown_terms=n_unique_unknown_terms,
            n_rows_obsolete=n_rows_obsolete,
            n_unique_obsolete_terms=n_unique_obsolete_terms,
            n_rows_nonobsolete=n_rows_nonobsolete,
            n_rows_obsolete_mapped=n_rows_obsolete_mapped,
            score_min=score_min,
            allow_active=allow_set is not None,
            out=out,
            round_decimals=round_decimals,
            leaf_only=leaf_only,
            n_leaf_only_dropped=n_leaf_only_dropped,
        )
        return out, stats_df

    def _build_stats(
        self,
        *,
        n_input_rows: int,
        n_rows_nonnull: int,
        n_rows_dropped_score: int,
        n_rows_dropped_allow: int,
        n_rows_after_filters: int,
        input_unique_terms: int,
        n_rows_go_like: int,
        n_rows_known: int,
        n_rows_unknown: int,
        n_unique_unknown_terms: int,
        n_rows_obsolete: int,
        n_unique_obsolete_terms: int,
        n_rows_nonobsolete: int,
        n_rows_obsolete_mapped: int,
        score_min: Optional[float],
        allow_active: bool,
        out: pl.DataFrame,
        round_decimals: Optional[int],
        leaf_only: bool = False,
        n_leaf_only_dropped: int = 0,
    ) -> pl.DataFrame:
        n_output_rows = out.height
        n_output_unique_terms = out["go_term"].n_unique() if n_output_rows else 0
        n_output_original_rows = (
            int((~out["propagated"]).sum()) if n_output_rows else 0
        )
        n_output_propagated_rows = (
            int(out["propagated"].sum()) if n_output_rows else 0
        )
        new_unique_terms = (
            out.filter(pl.col("propagated"))["go_term"].n_unique()
            if n_output_rows
            else 0
        )

        def pct(x: int, denom: int) -> float:
            return (100.0 * x / denom) if denom > 0 else 0.0

        rows_after_score = (
            n_rows_nonnull - n_rows_dropped_score
            if score_min is not None
            else n_rows_nonnull
        )

        rows: List[Tuple[str, int, float]] = [
            ("input_rows_total", n_input_rows, 100.0),
            ("input_rows_nonnull_cols", n_rows_nonnull, pct(n_rows_nonnull, n_input_rows)),
        ]
        if score_min is not None:
            rows.append(
                ("input_rows_dropped_by_score", n_rows_dropped_score,
                 pct(n_rows_dropped_score, n_rows_nonnull))
            )
        if allow_active:
            rows.extend([
                ("input_rows_dropped_by_allow_list", n_rows_dropped_allow,
                 pct(n_rows_dropped_allow, rows_after_score)),
                ("input_rows_after_allow_list", n_rows_after_filters,
                 pct(n_rows_after_filters, rows_after_score)),
            ])
        elif score_min is not None:
            rows.append(
                ("input_rows_after_score_filter", n_rows_after_filters,
                 pct(n_rows_after_filters, n_rows_nonnull))
            )

        rows.extend([
            ("input_unique_terms_raw_after_filters", input_unique_terms, 100.0),
            ("input_rows_GO_like", n_rows_go_like,
             pct(n_rows_go_like, n_rows_after_filters)),
            ("input_rows_GO_known", n_rows_known,
             pct(n_rows_known, n_rows_go_like)),
            ("input_rows_GO_unknown_dropped", n_rows_unknown,
             pct(n_rows_unknown, n_rows_go_like)),
            ("input_unique_GO_terms_unknown", n_unique_unknown_terms,
             pct(n_unique_unknown_terms, input_unique_terms)),
            ("input_rows_GO_obsolete", n_rows_obsolete,
             pct(n_rows_obsolete, n_rows_known)),
            ("input_unique_GO_terms_obsolete", n_unique_obsolete_terms,
             pct(n_unique_obsolete_terms, input_unique_terms)),
            ("input_rows_GO_nonobsolete_used", n_rows_nonobsolete,
             pct(n_rows_nonobsolete, n_rows_known)),
        ])
        if self.obsolete_mode == "map":
            rows.append(
                ("input_rows_GO_obsolete_mapped_to_replacement",
                 n_rows_obsolete_mapped,
                 pct(n_rows_obsolete_mapped, n_rows_obsolete))
            )
        if leaf_only:
            denom = n_output_rows + n_leaf_only_dropped
            rows.append(
                ("output_rows_dropped_as_non_leaf", n_leaf_only_dropped,
                 pct(n_leaf_only_dropped, denom))
            )
        rows.extend([
            ("output_rows_total", n_output_rows, 100.0),
            ("output_unique_terms", n_output_unique_terms, 100.0 if n_output_unique_terms else 0.0),
            ("output_rows_original", n_output_original_rows,
             pct(n_output_original_rows, n_output_rows)),
            ("output_rows_propagated", n_output_propagated_rows,
             pct(n_output_propagated_rows, n_output_rows)),
            ("output_unique_terms_new_from_propagation", new_unique_terms,
             pct(new_unique_terms, n_output_unique_terms)),
        ])
        stats_df = pl.DataFrame(
            rows,
            schema=["metric", "count", "percent"],
            orient="row",
        )
        if round_decimals is not None:
            stats_df = stats_df.with_columns(pl.col("percent").round(round_decimals))
        return stats_df


# ----------------------------------------------------------------------
# Pandas back-compat wrapper
# ----------------------------------------------------------------------


def propagate_go_annotations(
    df,
    obo_path: Union[str, Path],
    protein_col: str = "Protein",
    term_col: str = "GO_term/EC",
    score_col: Optional[str] = "Score",
    ic_df=None,
    ic_term_col: str = "go_term",
    ic_value_col: str = "IC",
    keep_original_terms: bool = True,
    relations: Union[str, Iterable[str]] = ("is_a",),
    exclude_roots: bool = True,
    obsolete_mode: ObsoleteMode = "drop",
    obsolete_prefer: ObsoletePrefer = "replaced_by",
    score_min: Optional[float] = None,
    round_decimals: int = 3,
    filter_predictable_terms: bool = False,
    predictable_terms_path: Optional[Union[str, Path]] = None,
    score_agg: ScoreAgg = "max",
    leaf_only: bool = False,
):
    """Pandas-in / pandas-out wrapper preserving the original output schema.

    Output columns: ``Protein``, ``GO_term``, ``Score`` (if present), ``Name``,
    ``Aspect``, ``IC``, ``Propagated``. Returns ``(out_df, stats_df)``.
    """
    import pandas as pd  # local import: keep top-level free of pandas

    allow_terms: AllowTerms = None
    if filter_predictable_terms:
        if predictable_terms_path is None:
            raise ValueError(
                "filter_predictable_terms=True requires predictable_terms_path."
            )
        allow_terms = predictable_terms_path

    propagator = GOPropagator(
        obo_path,
        relations=relations,
        exclude_roots=exclude_roots,
        obsolete_mode=obsolete_mode,
        obsolete_prefer=obsolete_prefer,
    )

    df_pl = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
    ic_pl = (
        pl.from_pandas(ic_df)
        if (ic_df is not None and isinstance(ic_df, pd.DataFrame))
        else ic_df
    )

    out_pl, stats_pl = propagator.propagate(
        df_pl,
        protein_col=protein_col,
        term_col=term_col,
        score_col=score_col,
        keep_original_terms=keep_original_terms,
        score_min=score_min,
        score_agg=score_agg,
        allow_terms=allow_terms,
        allow_terms_col="go_term",
        ic_df=ic_pl,
        ic_term_col=ic_term_col,
        ic_value_col=ic_value_col,
        round_decimals=round_decimals,
        leaf_only=leaf_only,
    )

    # Map to the original (capitalized) output schema.
    out = out_pl.to_pandas()
    out = out.rename(
        columns={
            "protein": "Protein",
            "go_term": "GO_term",
            "score": "Score",
            "name": "Name",
            "aspect": "Aspect",
            "ic": "IC",
            "propagated": "Propagated",
        }
    )
    return out, stats_pl.to_pandas()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _read_table(path: Path, separator: Optional[str] = None) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".tsv", ".csv", ".txt"} or separator is not None:
        sep = separator or ("\t" if suffix == ".tsv" else ",")
        return pl.read_csv(path, separator=sep)
    raise ValueError(
        f"Cannot infer format from extension {suffix!r}. "
        f"Use .parquet, .tsv, or .csv, or pass --separator."
    )


def _write_table(df: pl.DataFrame, path: Path, separator: Optional[str] = None) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".parquet":
        df.write_parquet(path)
        return
    if suffix in {".tsv", ".csv", ".txt"} or separator is not None:
        sep = separator or ("\t" if suffix == ".tsv" else ",")
        df.write_csv(path, separator=sep)
        return
    raise ValueError(
        f"Cannot infer output format from extension {suffix!r}. "
        f"Use .parquet, .tsv, or .csv, or pass --output-separator."
    )


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="goterm_propagator",
        description=(
            "Propagate leaf GO-term annotations to all ancestors under a "
            "configurable subset of relations. Input/output formats are "
            "inferred from extension (.tsv, .csv, .parquet)."
        ),
    )
    p.add_argument("-i", "--input", type=Path, required=True,
                   help="Input file (.tsv/.csv/.parquet).")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="Output file (.tsv/.csv/.parquet).")
    p.add_argument("--obo", type=Path, required=True,
                   help="Path to go-basic.obo (or any OBO).")

    p.add_argument("--protein-col", default="protein",
                   help="Input column with protein ID. Default: 'protein'.")
    p.add_argument("--term-col", default="go_term",
                   help="Input column with GO term. Default: 'go_term'.")
    p.add_argument("--score-col", default="score",
                   help=("Input column with prediction score. "
                         "Use '' (empty) or --no-score to disable. Default: 'score'."))
    p.add_argument("--no-score", action="store_true",
                   help="Treat input as scoreless (presence/absence).")

    p.add_argument("--separator", default=None,
                   help="Input field separator (auto-detected from extension if omitted).")
    p.add_argument("--output-separator", default=None,
                   help="Output field separator (auto-detected from extension if omitted).")

    p.add_argument("--ic", type=Path, default=None,
                   help="Optional CSV/TSV/Parquet with IC values.")
    p.add_argument("--ic-term-col", default="go_term",
                   help="IC table: term column. Default: 'go_term'.")
    p.add_argument("--ic-value-col", default="IC",
                   help="IC table: value column. Default: 'IC'.")

    p.add_argument("--allow-terms", type=Path, default=None,
                   help=("Optional allow-list of GO terms (CSV/TSV/TXT). "
                         "Filters input BEFORE propagation."))
    p.add_argument("--allow-terms-col", default="go_term",
                   help="Column in --allow-terms file. Default: 'go_term'.")

    p.add_argument("--score-min", type=float, default=None,
                   help="Drop input rows with score < score-min before propagation.")
    p.add_argument("--score-agg", default="max",
                   choices=("max", "noisy_or", "sum", "mean"),
                   help="Aggregator for combining scores at a shared ancestor. Default: max.")

    p.add_argument("--relations", default="is_a",
                   help=("Comma-separated subset of {is_a, part_of}. "
                         "Default: is_a."))
    p.add_argument("--include-roots", action="store_true",
                   help="Keep the three GO roots in the output.")
    p.add_argument("--no-keep-originals", action="store_true",
                   help="Output ancestors only; drop the original (non-propagated) rows.")
    p.add_argument("--leaf-only", action="store_true",
                   help=("Skip propagation; instead keep only the most-specific "
                         "terms per protein (drop terms that are ancestors of "
                         "another term in the same protein's set). Useful for "
                         "compressing storage."))

    p.add_argument("--obsolete-mode", default="drop", choices=("drop", "map"),
                   help="Drop obsolete terms, or map to replaced_by/consider. Default: drop.")
    p.add_argument("--obsolete-prefer", default="replaced_by",
                   choices=("replaced_by", "consider"),
                   help="Preferred field when mapping obsoletes. Default: replaced_by.")

    p.add_argument("--round-decimals", type=int, default=3,
                   help="Decimal places for score / IC / stats percent. Default: 3.")

    p.add_argument("--stats", type=Path, default=None,
                   help="Optional path to write the stats table (same format rules as --output).")
    p.add_argument("--quiet", action="store_true", help="Suppress progress messages.")

    p.add_argument("--chunk-size", type=int, default=0,
                   help=("If > 0, process the input in chunks of N rows. "
                         "Required for inputs too large to propagate in one shot. "
                         "Parquet input/output only."))

    return p


def _recompute_chunked_stats(
    metric_order: List[str],
    counts: Dict[str, int],
    *,
    n_output_unique: int,
    n_propagated_unique: int,
    round_decimals: Optional[int] = 3,
) -> pl.DataFrame:
    """Rebuild stats DataFrame from summed per-chunk counts.

    Row-count metrics sum correctly across chunks. Output unique-term counts
    are overridden with the actual cross-chunk set cardinalities. Input-side
    unique counts remain as sum-across-chunks approximations.
    """
    def pct(num: int, denom: int) -> float:
        return (100.0 * num / denom) if denom > 0 else 0.0

    # Override output uniques with true set cardinalities
    counts = dict(counts)
    counts["output_unique_terms"] = n_output_unique
    counts["output_unique_terms_new_from_propagation"] = n_propagated_unique

    # Compute denominators from summed counts
    n_total = counts.get("input_rows_total", 0)
    n_nonnull = counts.get("input_rows_nonnull_cols", n_total)
    n_after_score = n_nonnull - counts.get("input_rows_dropped_by_score", 0)
    n_after_filters = counts.get(
        "input_rows_after_allow_list",
        counts.get("input_rows_after_score_filter", n_nonnull),
    )
    n_go_like = counts.get("input_rows_GO_like", 0)
    n_known = counts.get("input_rows_GO_known", 0)
    n_obsolete = counts.get("input_rows_GO_obsolete", 0)
    n_unique_terms = counts.get("input_unique_terms_raw_after_filters", 0)
    n_out_total = counts.get("output_rows_total", 0)
    n_dropped_leaf = counts.get("output_rows_dropped_as_non_leaf", 0)

    denom: Dict[str, Optional[int]] = {
        "input_rows_total": None,
        "input_rows_nonnull_cols": n_total,
        "input_rows_dropped_by_score": n_nonnull,
        "input_rows_after_score_filter": n_nonnull,
        "input_rows_dropped_by_allow_list": n_after_score,
        "input_rows_after_allow_list": n_after_score,
        "input_unique_terms_raw_after_filters": None,
        "input_rows_GO_like": n_after_filters,
        "input_rows_GO_known": n_go_like,
        "input_rows_GO_unknown_dropped": n_go_like,
        "input_unique_GO_terms_unknown": n_unique_terms,
        "input_rows_GO_obsolete": n_known,
        "input_unique_GO_terms_obsolete": n_unique_terms,
        "input_rows_GO_nonobsolete_used": n_known,
        "input_rows_GO_obsolete_mapped_to_replacement": n_obsolete,
        "output_rows_dropped_as_non_leaf": n_out_total + n_dropped_leaf,
        "output_rows_total": None,
        "output_unique_terms": None,
        "output_rows_original": n_out_total,
        "output_rows_propagated": n_out_total,
        "output_unique_terms_new_from_propagation": n_output_unique,
    }

    rows = []
    for m in metric_order:
        c = counts[m]
        d = denom.get(m)
        if d is None:
            p = 100.0 if c > 0 else 0.0
        else:
            p = pct(c, d)
        rows.append((m, c, p))
    df = pl.DataFrame(rows, schema=["metric", "count", "percent"], orient="row")
    if round_decimals is not None:
        df = df.with_columns(pl.col("percent").round(round_decimals))
    return df


def _run_chunked(args, propagator: "GOPropagator", ic_df: Optional[pl.DataFrame],
                 score_col: Optional[str], log) -> None:
    import time
    import pyarrow.parquet as pq

    if args.input.suffix.lower() != ".parquet":
        raise SystemExit("--chunk-size requires .parquet input.")
    if args.output.suffix.lower() != ".parquet":
        raise SystemExit("--chunk-size requires .parquet output.")

    n_total = pl.scan_parquet(args.input).select(pl.len()).collect().item()

    # Score aggregation happens per chunk, so a protein split across a chunk
    # boundary would yield partial, duplicated (protein, term) outputs whose
    # scores are aggregated over only a subset of the protein's parents.
    # Sort by protein and snap chunk boundaries to protein edges so every
    # protein is processed whole, in exactly one chunk.
    sorted_input = args.output.parent / (args.output.stem + ".sorted_input.tmp.parquet")
    (pl.scan_parquet(args.input)
       .sort(args.protein_col)
       .sink_parquet(sorted_input, compression="zstd"))

    _pcol = (pl.scan_parquet(sorted_input)
             .select(args.protein_col).with_row_index("_ri").collect())
    _grp = (_pcol.with_columns((pl.col("_ri") // args.chunk_size).alias("_cid"))
                 .group_by(args.protein_col, maintain_order=True)
                 .agg(pl.col("_cid").first().alias("_cid"), pl.len().alias("_n")))
    _ends = (_grp.with_columns(pl.col("_n").cum_sum().alias("_end"))
                 .group_by("_cid", maintain_order=True)
                 .agg(pl.col("_end").last())["_end"].to_list())
    chunk_ranges: List[Tuple[int, int]] = []
    _prev = 0
    for _e in _ends:
        chunk_ranges.append((_prev, _e - _prev))
        _prev = _e
    del _pcol, _grp

    n_chunks = len(chunk_ranges)
    log(f"[chunks] {n_total:,} input rows → {n_chunks} protein-aligned "
        f"chunk(s) of up to ~{args.chunk_size:,} rows")

    summed_counts: Dict[str, int] = {}
    metric_order: List[str] = []
    output_terms_seen: Set[str] = set()
    propagated_terms_seen: Set[str] = set()

    writer = None
    t_load_total = 0.0
    t_prop_total = 0.0
    n_in_total = 0
    n_out_total = 0

    propagate_kwargs = dict(
        protein_col=args.protein_col,
        term_col=args.term_col,
        score_col=score_col,
        keep_original_terms=not args.no_keep_originals,
        score_min=args.score_min,
        score_agg=args.score_agg,
        allow_terms=args.allow_terms,
        allow_terms_col=args.allow_terms_col,
        ic_df=ic_df,
        ic_term_col=args.ic_term_col,
        ic_value_col=args.ic_value_col,
        round_decimals=args.round_decimals,
        leaf_only=args.leaf_only,
    )

    try:
        for ci, (offset, length) in enumerate(chunk_ranges):
            t0 = time.perf_counter()
            chunk = (
                pl.scan_parquet(sorted_input)
                .slice(offset, length)
                .collect()
            )
            t_load = time.perf_counter() - t0
            t_load_total += t_load
            n_in_total += chunk.height

            t0 = time.perf_counter()
            out, stats = propagator.propagate(chunk, **propagate_kwargs)
            t_prop = time.perf_counter() - t0
            t_prop_total += t_prop
            n_out_total += out.height

            # Accumulate stats
            for m, c in zip(stats["metric"].to_list(), stats["count"].to_list()):
                if m not in summed_counts:
                    summed_counts[m] = 0
                    metric_order.append(m)
                summed_counts[m] += int(c)

            # Track output unique terms
            if out.height > 0:
                output_terms_seen.update(out["go_term"].to_list())
                propagated_terms_seen.update(
                    out.filter(pl.col("propagated"))["go_term"].to_list()
                )

            # Stream chunk to parquet
            arrow_table = out.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    args.output, arrow_table.schema, compression="snappy"
                )
            writer.write_table(arrow_table)

            log(f"[chunk {ci+1}/{n_chunks}] load={t_load:.2f}s "
                f"prop={t_prop:.2f}s  in={chunk.height:,}  out={out.height:,}  "
                f"running_out={n_out_total:,}")
            del chunk, out, stats, arrow_table
    finally:
        if writer is not None:
            writer.close()
        sorted_input.unlink(missing_ok=True)

    log(f"[done]  total: load={t_load_total:.1f}s  prop={t_prop_total:.1f}s  "
        f"in={n_in_total:,} → out={n_out_total:,}")

    final_stats = _recompute_chunked_stats(
        metric_order,
        summed_counts,
        n_output_unique=len(output_terms_seen),
        n_propagated_unique=len(propagated_terms_seen),
        round_decimals=args.round_decimals,
    )

    if args.stats is not None:
        _write_table(final_stats, args.stats, separator=args.output_separator)
        log(f"[stats] {args.stats}")
    elif not args.quiet:
        log("--- stats (chunked totals; input-side unique counts are "
            "sum-across-chunks approximations) ---")
        with pl.Config(tbl_rows=-1, tbl_width_chars=120):
            import sys
            print(final_stats, file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    import sys
    import time

    args = _build_arg_parser().parse_args(argv)

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)

    score_col: Optional[str] = None if (args.no_score or args.score_col == "") else args.score_col
    relations = tuple(r.strip() for r in args.relations.split(",") if r.strip())

    t0 = time.perf_counter()
    propagator = GOPropagator(
        args.obo,
        relations=relations,
        exclude_roots=not args.include_roots,
        obsolete_mode=args.obsolete_mode,
        obsolete_prefer=args.obsolete_prefer,
    )
    log(f"[obo]   parsed in {time.perf_counter()-t0:.2f}s "
        f"({len(propagator.terms):,} terms, closure {propagator._closure.height:,} rows)")

    ic_df: Optional[pl.DataFrame] = None
    if args.ic is not None:
        ic_df = _read_table(args.ic)
        log(f"[ic]    {args.ic} → {ic_df.height:,} rows")

    if args.chunk_size > 0:
        _run_chunked(args, propagator, ic_df, score_col, log)
        return 0

    t0 = time.perf_counter()
    df = _read_table(args.input, separator=args.separator)
    log(f"[load]  {args.input} → {df.height:,} rows in {time.perf_counter()-t0:.2f}s")

    # Heads-up if the propagator will auto-explode list columns
    list_msgs = []
    if args.term_col in df.columns and isinstance(df.schema[args.term_col], pl.List):
        list_msgs.append(args.term_col)
    if (score_col is not None and score_col in df.columns
            and isinstance(df.schema[score_col], pl.List)):
        list_msgs.append(score_col)
    if list_msgs:
        log(f"[note]  list-typed column(s) detected: {list_msgs} — will auto-explode "
            f"(use --chunk-size if this OOMs)")

    t0 = time.perf_counter()
    out, stats = propagator.propagate(
        df,
        protein_col=args.protein_col,
        term_col=args.term_col,
        score_col=score_col,
        keep_original_terms=not args.no_keep_originals,
        score_min=args.score_min,
        score_agg=args.score_agg,
        allow_terms=args.allow_terms,
        allow_terms_col=args.allow_terms_col,
        ic_df=ic_df,
        ic_term_col=args.ic_term_col,
        ic_value_col=args.ic_value_col,
        round_decimals=args.round_decimals,
        leaf_only=args.leaf_only,
    )
    log(f"[prop]  {out.height:,} output rows in {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    _write_table(out, args.output, separator=args.output_separator)
    log(f"[write] {args.output} ({time.perf_counter()-t0:.2f}s)")

    if args.stats is not None:
        _write_table(stats, args.stats, separator=args.output_separator)
        log(f"[stats] {args.stats}")
    elif not args.quiet:
        log("--- stats ---")
        with pl.Config(tbl_rows=-1, tbl_width_chars=120):
            print(stats, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
