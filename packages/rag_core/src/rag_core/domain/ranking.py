# pyright: standard
"""
Ranking policy after the reranker: recency bonus,
relevance floor (abstain), minimum source score and a per-document cap.
Operates on plain metadata dicts; standard library only. There is no
audience filter: every source may be shown to every user.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

YEAR_PATTERN = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")
# The years a question names (a bare year, a school year "2022/2023", the year of a date "31.12.2012"),
# never the digits of an amount ("1.500", "2.019,50") or a phone number.
DATE_YEAR = re.compile(r"(?<!\d)\d{1,2}\.\d{1,2}\.((?:19[5-9]|20[0-4])\d)(?!\d)")
BARE_YEAR = re.compile(r"(?<![\d.,])((?:19[5-9]|20[0-4])\d)(?!\d)(?![.,]\d)")


@dataclass
class RankingPolicy:
    relevance_floor: float = 0.0  # best score below this -> abstain
    source_min_score: float = 0.0  # single sources below this are dropped
    max_per_document: int = 0  # 0 = no cap
    recency_types: Tuple[str, ...] = ()  # newer wins among these document types; none by default
    recency_bonus_per_year: float = 0.0
    recency_max_bonus: float = 0.0
    date_from: Optional[str] = None  # ISO dates from the query analysis
    date_to: Optional[str] = None
    in_range_bonus: float = 0.0  # added to dated sources inside [date_from, date_to]
    today: Optional[date] = None
    # figure and time questions; each is off by default, which leaves the ranking above unchanged
    in_range_by_year: bool = False  # a source known only by its year (doc_year) counts as dated for the bonus
    question_years: bool = False  # no dates from the analysis: the period spans the years the question names
    no_recency_in_range: bool = False  # inside a named period the period decides, not the age of the source


def question_period(question):
    """(date_from, date_to) spanning the years the question names, else (None, None)."""
    text = question or ""
    years = sorted({int(y) for y in DATE_YEAR.findall(text) + BARE_YEAR.findall(text)})
    return (f"{years[0]}-01-01", f"{years[-1]}-12-31") if years else (None, None)


def _year(meta, max_year):
    """doc_date, doc_year, else the last year in the decoded URL or title; years after
    max_year count as unknown (e.g. "%2045", "STEP 2040" in a URL or title)."""
    year = None
    for value in (str(meta.get("doc_date") or "")[:4], meta.get("doc_year")):
        try:
            year = int(value) if value else None
        except (TypeError, ValueError):
            year = None
        if year:
            break
    if not year:
        url = unquote(str(meta.get("download_url") or meta.get("url") or ""))
        found = [int(y) for y in YEAR_PATTERN.findall(url + " " + (meta.get("title") or "")) if int(y) <= max_year]
        year = found[-1] if found else None
    return year if year and year <= max_year else None


def _in_years(doc_year, date_from, date_to):
    try:
        return int(date_from[:4]) <= int(doc_year) <= int((date_to or date_from)[:4])
    except (TypeError, ValueError):
        return False


def rank(
    metas: List[dict[str, Any]], question: str, policy: RankingPolicy
) -> Tuple[List[int], Dict[int, float], Dict[str, Any]]:
    """Returns (kept_indices_in_rank_order, adjusted_scores_by_index, report).
    kept is empty when the best adjusted score is below the relevance floor."""
    today = policy.today or date.today()
    date_from, date_to = policy.date_from, policy.date_to
    if policy.question_years and not date_from:
        date_from, date_to = question_period(question)
    report = {
        "candidates": len(metas),
        "abstain": False,
        "best": 0.0,
        "below_min": 0,
        "capped": 0,
    }
    rerank_failed = any(m.get("rerank_failed") for m in metas)
    report["rerank_failed"] = rerank_failed
    scored = []
    for i, m in enumerate(metas):
        if rerank_failed:
            scored.append((0.0, i))  # no scores to adjust: keep the retrieval order
            continue
        score = float(m.get("cross_encoder_score") or 0.0)
        doc_type = m.get("doc_type")
        year = _year(m, today.year + 1)
        if (
            year
            and doc_type in policy.recency_types
            and policy.recency_bonus_per_year
            and not (policy.no_recency_in_range and date_from)
        ):
            age = max(0, today.year - year)
            score += max(0.0, policy.recency_max_bonus - age * policy.recency_bonus_per_year)
        doc_date = str(m.get("doc_date") or "")
        if policy.in_range_bonus and date_from:
            if doc_date:
                if date_from <= doc_date[:10] <= (date_to or date_from):
                    score += policy.in_range_bonus
            elif policy.in_range_by_year and _in_years(m.get("doc_year"), date_from, date_to):
                score += policy.in_range_bonus
        scored.append((score, i))
    scored.sort(key=lambda t: t[0], reverse=True)  # stable: equal scores keep the retrieval order
    adjusted = {i: s for s, i in scored}
    report["best"] = round(scored[0][0], 3) if scored else 0.0
    if not scored or (not rerank_failed and scored[0][0] < policy.relevance_floor):
        report["abstain"] = True
        return [], adjusted, report
    kept, per_doc = [], {}
    for s, i in scored:
        if not rerank_failed and s < policy.source_min_score:
            report["below_min"] += 1
            continue
        key = metas[i].get("doc_key") or metas[i].get("download_url") or metas[i].get("url") or str(i)
        if policy.max_per_document and per_doc.get(key, 0) >= policy.max_per_document:
            report["capped"] += 1
            continue
        per_doc[key] = per_doc.get(key, 0) + 1
        kept.append(i)
    return kept, adjusted, report
