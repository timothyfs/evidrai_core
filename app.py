from __future__ import annotations

import json
import os
import re
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type
from urllib.parse import urlparse

import requests
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

from prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_claim_analysis_messages,
    build_reasoning_messages,
    build_source_summary_messages,
    load_json,
)


# -----------------------------
# Models
# -----------------------------

@dataclass
class SubClaim:
    id: str
    text: str
    claim_type: str
    entities: List[str] = field(default_factory=list)
    jurisdiction: Optional[str] = None
    time_sensitivity: str = "medium"
    verification_requirements: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)


@dataclass
class EvidenceSource:
    title: str
    url: str
    domain: str
    source_type: str
    snippet: str = ""
    content: str = ""
    published_date: Optional[str] = None
    authority_score: float = 0.0
    relevance_score: float = 0.0
    directness_score: float = 0.0
    recency_score: float = 0.0
    bias_risk_score: float = 2.5
    weighted_score: float = 0.0
    claim_support: str = "irrelevant"
    evidence_category: str = "irrelevant"
    source_role: str = "context"
    narrative_cluster: str = ""




class EvidenceTypeModel(BaseModel):
    type: str = "Unknown"
    weight: str = "Low"
    impact: str = "neutral"
    note: str = ""


class LegacyAssessmentModel(BaseModel):
    verdict: str = "Unverified"
    confidence: str = "Low"
    tldr: str = ""
    summary: str = ""
    why_convincing: str = ""
    interpretation_note: str = ""
    interpretation_confidence: str = "Low"
    explicit_vs_inferred: Dict[str, str] = Field(default_factory=dict)
    evidence_access_note: str = ""
    evidence_types: List[EvidenceTypeModel] = Field(default_factory=list)
    what_would_change_verdict: str = ""
    user_takeaway: str = ""
    caution_flags: List[str] = Field(default_factory=list)


class ClaimAnalysisItemModel(BaseModel):
    id: str = "sc_1"
    text: str
    claim_type: str = "other"
    entities: List[str] = Field(default_factory=list)
    jurisdiction: Optional[str] = None
    time_sensitivity: str = "medium"
    verification_requirements: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class ClaimAnalysisModel(BaseModel):
    normalized_claim: str = ""
    subclaims: List[ClaimAnalysisItemModel] = Field(default_factory=list)
    overall_notes: List[str] = Field(default_factory=list)


class SourceSummaryModel(BaseModel):
    summary: str = ""
    claim_support: str = "irrelevant"
    evidence_category: str = "irrelevant"
    source_role: str = "context"
    narrative_cluster: str = ""
    key_points: List[str] = Field(default_factory=list)
    quoted_or_precise_points: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)


class ReasoningSummaryModel(BaseModel):
    supported_points: List[str] = Field(default_factory=list)
    contradicted_points: List[str] = Field(default_factory=list)
    uncertain_points: List[str] = Field(default_factory=list)


class EvidenceAssessmentModel(BaseModel):
    primary_sources_used: List[str] = Field(default_factory=list)
    secondary_sources_used: List[str] = Field(default_factory=list)
    source_conflicts: List[str] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)
    rumor_drivers: List[str] = Field(default_factory=list)
    actual_evidence: List[str] = Field(default_factory=list)


class VerifiedAssessmentModel(BaseModel):
    verified_verdict: str = "Unverified"
    verified_confidence: str = "Low"
    consensus_strength: str = "No clear consensus"
    consensus_summary: str = ""
    pendulum_band: str = ""
    pendulum_explanation: str = ""
    tldr: str = ""
    one_line_correction: str = ""
    reasoning_summary: ReasoningSummaryModel = Field(default_factory=ReasoningSummaryModel)
    evidence_assessment: EvidenceAssessmentModel = Field(default_factory=EvidenceAssessmentModel)
    misinformation_patterns: List[str] = Field(default_factory=list)
    why_this_claim_spreads: List[str] = Field(default_factory=list)
    final_explanation: str = ""


@dataclass(frozen=True)
class ScoringConfig:
    authority_weight: float = 0.30
    relevance_weight: float = 0.25
    directness_weight: float = 0.20
    recency_weight: float = 0.15
    bias_weight: float = 0.10
    max_source_summaries: int = 8
    max_summary_workers: int = 4
    max_retries: int = 3
    retry_base_sleep: float = 1.0
    term_pattern: str = r"\b{term}\b"


SCORING_CONFIG = ScoringConfig()

# -----------------------------
# Utility
# -----------------------------

PRIMARY_DOMAINS = (
    ".gov",
    ".gouv.fr",
    ".parliament.uk",
    ".legislation.gov.uk",
    ".judiciary.uk",
    ".edu",
    "who.int",
    "nih.gov",
    "nhs.uk",
    "oecd.org",
)
SECONDARY_DOMAINS = (
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "ft.com",
    "nytimes.com",
    "theguardian.com",
    "lemonde.fr",
    "france24.com",
)


def is_probable_url(value: str) -> bool:
    return bool(re.match(r"^https?://", value or "", flags=re.I))


def build_analysis_input(claim: str, source_url: str) -> str:
    claim = (claim or "").strip()
    source_url = (source_url or "").strip()
    if claim and source_url:
        return f"Claim or content to assess:\n{claim}\n\nOptional source URL:\n{source_url}"
    return claim or source_url


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def classify_source_type(domain: str) -> str:
    if any(d in domain for d in PRIMARY_DOMAINS):
        return "primary"
    if any(d in domain for d in SECONDARY_DOMAINS):
        return "secondary"
    return "contextual"


def parse_iso_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def recency_score(date_str: Optional[str]) -> float:
    dt = parse_iso_date(date_str)
    if not dt:
        return 2.5
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    days = max(0, (now - dt).days)
    if days <= 7:
        return 5.0
    if days <= 30:
        return 4.0
    if days <= 180:
        return 3.0
    if days <= 365:
        return 2.0
    return 1.0




def stable_request_key(*parts: Any) -> str:
    joined = "||".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def validate_model(payload: Dict[str, Any], model_cls: Type[BaseModel]) -> Dict[str, Any]:
    model = model_cls.model_validate(payload)
    return model.model_dump()

# -----------------------------
# LLM integrations
# -----------------------------

class OpenAICompatibleClient:
    def __init__(self) -> None:
        secrets = getattr(st, "secrets", {})
        self.api_key = secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        self.base_url = secrets.get("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        self.model = secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> Dict[str, Any]:
        if not self.configured:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        last_exc: Optional[Exception] = None
        for attempt in range(SCORING_CONFIG.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "temperature": temperature,
                        "response_format": {"type": "json_object"},
                        "messages": messages,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                parsed = load_json(content)
                if not isinstance(parsed, dict):
                    raise ValueError("Model returned non-object JSON.")
                return parsed
            except (requests.RequestException, ValueError, KeyError, ValidationError, TypeError) as exc:
                last_exc = exc
                if attempt == SCORING_CONFIG.max_retries - 1:
                    break
                time.sleep(SCORING_CONFIG.retry_base_sleep * (2 ** attempt))
        raise RuntimeError(f"LLM request failed after retries: {last_exc}")


class TavilySearchClient:
    def __init__(self) -> None:
        secrets = getattr(st, "secrets", {})
        self.api_key = secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.configured:
            return []
        last_exc: Optional[Exception] = None
        for attempt in range(SCORING_CONFIG.max_retries):
            try:
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                        "include_raw_content": True,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                out: List[Dict[str, Any]] = []
                for item in data.get("results", []):
                    out.append(
                        {
                            "title": item.get("title", "Untitled"),
                            "url": item.get("url", ""),
                            "snippet": item.get("content", "")[:500],
                            "content": item.get("raw_content") or item.get("content") or "",
                            "published_date": item.get("published_date"),
                        }
                    )
                return out
            except (requests.RequestException, ValueError, TypeError) as exc:
                last_exc = exc
                if attempt == SCORING_CONFIG.max_retries - 1:
                    break
                time.sleep(SCORING_CONFIG.retry_base_sleep * (2 ** attempt))
        raise RuntimeError(f"Search request failed after retries: {last_exc}")


# -----------------------------
# Legacy single-pass analysis
# -----------------------------


def call_legacy_model(claim: str, category: str, detail_mode: str, llm: OpenAICompatibleClient) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(claim, category, detail_mode)},
    ]
    payload = llm.complete_json(messages, temperature=0.1)
    return validate_model(payload, LegacyAssessmentModel)


# -----------------------------
# Fast provisional pass
# -----------------------------


def run_quick_pass(user_input: str, category: str, llm: OpenAICompatibleClient) -> Dict[str, Any]:
    """Fast first-pass assessment without external retrieval."""
    try:
        data = call_legacy_model(user_input, category or "auto-detect", "Simple", llm)
    except Exception:
        # Fallback minimal payload so the UI can still stage the response cleanly.
        data = {}
    return {
        "verdict": data.get("verdict", "Unverified"),
        "confidence": data.get("confidence", "Low"),
        "tldr": data.get("tldr") or data.get("summary") or "Initial assessment generated.",
        "one_line_correction": data.get("user_takeaway") or data.get("what_would_change_verdict") or "Deep verification may refine this answer.",
        "summary": data.get("summary", ""),
        "why_convincing": data.get("why_convincing", ""),
        "evidence_access_note": data.get("evidence_access_note", ""),
        "what_would_change_verdict": data.get("what_would_change_verdict", ""),
        "user_takeaway": data.get("user_takeaway", ""),
        "evidence_types": data.get("evidence_types", []) or [],
    }


# -----------------------------
# Multi-step pipeline
# -----------------------------


def parse_claim_analysis(payload: Dict[str, Any], user_input: str) -> List[SubClaim]:
    validated = validate_model(payload, ClaimAnalysisModel)
    subclaims = []
    for i, item in enumerate(validated.get("subclaims", []) or []):
        subclaims.append(
            SubClaim(
                id=str(item.get("id", f"sc_{i+1}")),
                text=(item.get("text") or "").strip(),
                claim_type=item.get("claim_type", "other"),
                entities=list(item.get("entities", []) or []),
                jurisdiction=item.get("jurisdiction"),
                time_sensitivity=item.get("time_sensitivity", "medium"),
                verification_requirements=list(item.get("verification_requirements", []) or []),
                risk_flags=[x for x in item.get("risk_flags", []) if x],
            )
        )
    if not subclaims:
        subclaims = [SubClaim(id="sc_1", text=user_input.strip(), claim_type="other")]
    return subclaims


def build_search_queries(subclaims: List[SubClaim]) -> List[str]:
    queries: List[str] = []
    seen = set()
    for sub in subclaims:
        candidates = [
            sub.text,
            f'"{sub.text}"',
            f"{sub.text} official source",
            f"{sub.text} evidence",
            f"{sub.text} debunked OR disputed",
        ]
        if sub.claim_type == "legal":
            candidates.extend([
                f"site:gov.uk {sub.text}",
                f"site:legislation.gov.uk {sub.text}",
                f"site:judiciary.uk {sub.text}",
            ])
        for q in candidates:
            q = re.sub(r"\s+", " ", q).strip()
            if q and q not in seen:
                seen.add(q)
                queries.append(q)
    return queries[:12]


def score_source(item: Dict[str, Any], claim_text: str) -> EvidenceSource:
    url = item.get("url", "")
    domain = domain_from_url(url)
    source_type = classify_source_type(domain)
    title = item.get("title", "Untitled")
    snippet = item.get("snippet", "")
    content = item.get("content", "")
    haystack = f"{title} {snippet} {content}".lower()
    terms = [t for t in re.findall(r"[A-Za-z]{4,}", claim_text.lower())][:8]
    overlap = 0
    for t in terms:
        if re.search(SCORING_CONFIG.term_pattern.format(term=re.escape(t)), haystack):
            overlap += 1
    relevance = min(5.0, 1.0 + overlap)
    authority = 5.0 if source_type == "primary" else 3.8 if source_type == "secondary" else 2.2
    directness = 4.5 if any(re.search(SCORING_CONFIG.term_pattern.format(term=re.escape(t)), haystack) for t in terms[:3]) else 2.5
    recency = recency_score(item.get("published_date"))
    bias_risk = 1.5 if source_type == "primary" else 2.5 if source_type == "secondary" else 3.5
    weighted = authority * SCORING_CONFIG.authority_weight + relevance * SCORING_CONFIG.relevance_weight + directness * SCORING_CONFIG.directness_weight + recency * SCORING_CONFIG.recency_weight + (5 - bias_risk) * SCORING_CONFIG.bias_weight
    return EvidenceSource(
        title=title,
        url=url,
        domain=domain,
        source_type=source_type,
        snippet=snippet,
        content=content,
        published_date=item.get("published_date"),
        authority_score=authority,
        relevance_score=relevance,
        directness_score=directness,
        recency_score=recency,
        bias_risk_score=bias_risk,
        weighted_score=round(weighted, 2),
    )


def retrieve_sources(search: TavilySearchClient, queries: List[str], claim_text: str) -> List[EvidenceSource]:
    dedup: Dict[str, EvidenceSource] = {}
    for query in queries[:6]:
        for item in search.search(query, max_results=4):
            url = item.get("url") or ""
            if not url or url in dedup:
                continue
            dedup[url] = score_source(item, claim_text)
    return sorted(dedup.values(), key=lambda x: x.weighted_score, reverse=True)[:SCORING_CONFIG.max_source_summaries]


def _summarize_one_source(llm: OpenAICompatibleClient, subclaim_text: str, source: EvidenceSource) -> EvidenceSource:
    text = source.content or source.snippet
    if not text:
        return source
    payload = llm.complete_json(build_source_summary_messages(subclaim_text, source.title, source.url, text[:6000]))
    validated = validate_model(payload, SourceSummaryModel)
    source.claim_support = validated.get("claim_support", "irrelevant")
    source.evidence_category = validated.get("evidence_category", "irrelevant")
    source.source_role = validated.get("source_role", "context")
    source.narrative_cluster = validated.get("narrative_cluster", "")
    source.snippet = validated.get("summary", source.snippet)
    return source


def summarize_sources(llm: OpenAICompatibleClient, subclaim: SubClaim, sources: List[EvidenceSource]) -> List[EvidenceSource]:
    if not llm.configured or not sources:
        return sources

    indexed_sources = list(enumerate(sources))
    results: Dict[int, EvidenceSource] = {}
    max_workers = min(SCORING_CONFIG.max_summary_workers, len(indexed_sources))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_summarize_one_source, llm, subclaim.text, source): idx
            for idx, source in indexed_sources
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = sources[idx]

    return [results.get(i, source) for i, source in indexed_sources]


def compute_confidence(sources: List[EvidenceSource]) -> int:
    if not sources:
        return 35
    avg = sum(s.weighted_score for s in sources) / len(sources)
    primary_count = sum(1 for s in sources if s.source_type == "primary")
    contradictory = sum(1 for s in sources if s.claim_support == "contradicts")
    supporting = sum(1 for s in sources if s.claim_support == "supports")
    score = int(avg * 14 + primary_count * 4 - max(0, contradictory - supporting) * 4)
    return max(20, min(score, 96))


def provisional_verdict(sources: List[EvidenceSource]) -> str:
    supports = sum(1 for s in sources if s.claim_support == "supports")
    contradicts = sum(1 for s in sources if s.claim_support == "contradicts")
    mixed = sum(1 for s in sources if s.claim_support == "mixed")
    primary_support = any(s.source_type == "primary" and s.claim_support == "supports" for s in sources)
    primary_contradict = any(s.source_type == "primary" and s.claim_support == "contradicts" for s in sources)
    if primary_contradict and contradicts >= supports + mixed:
        return "false"
    if primary_support and supports >= contradicts + mixed:
        return "true"
    if mixed or (supports and contradicts):
        return "misleading"
    if not sources:
        return "unverifiable"
    return "unverifiable"


def run_claim_pipeline(user_input: str, llm: OpenAICompatibleClient, search: TavilySearchClient) -> Dict[str, Any]:
    claim_analysis = validate_model(llm.complete_json(build_claim_analysis_messages(user_input)), ClaimAnalysisModel)
    subclaims = parse_claim_analysis(claim_analysis, user_input)
    claim_text = claim_analysis.get("normalized_claim") or user_input
    queries = build_search_queries(subclaims)
    sources = retrieve_sources(search, queries, claim_text)
    sources = summarize_sources(llm, subclaims[0], sources)
    confidence = compute_confidence(sources)
    pre = provisional_verdict(sources)

    evidence_packet = {
        "claim": claim_text,
        "subclaims": [s.text for s in subclaims],
        "sources": [
            {
                "title": s.title,
                "url": s.url,
                "domain": s.domain,
                "source_type": s.source_type,
                "published_date": s.published_date,
                "summary": s.snippet,
                "claim_support": s.claim_support,
                "evidence_category": getattr(s, "evidence_category", "irrelevant"),
                "source_role": getattr(s, "source_role", "context"),
                "narrative_cluster": getattr(s, "narrative_cluster", ""),
                "weighted_score": s.weighted_score,
            }
            for s in sources
        ],
    }

    pendulum = evidence_pendulum(
        evidence_packet["sources"],
        subclaims[0].claim_type if subclaims else "other",
    )

    reasoning = validate_model(
        llm.complete_json(
            build_reasoning_messages(
                claim_text,
                evidence_packet,
                pre,
                confidence,
                pendulum["band"],
                pendulum["explanation"],
            )
        ),
        VerifiedAssessmentModel,
    )

    reasoning["claim"] = claim_text
    reasoning["subclaims"] = evidence_packet["subclaims"]
    reasoning["sources"] = evidence_packet["sources"]
    reasoning["queries"] = queries
    reasoning["risk_flags"] = sorted(collect_risk_flags(subclaims))
    reasoning["pendulum_band"] = reasoning.get("pendulum_band") or pendulum["band"]
    reasoning["pendulum_explanation"] = reasoning.get("pendulum_explanation") or pendulum["explanation"]
    reasoning["verified_verdict"] = reasoning.get("verified_verdict") or map_pendulum_to_verified_verdict(reasoning["pendulum_band"])
    reasoning["verified_confidence"] = reasoning.get("verified_confidence") or map_confidence_label(reasoning.get("confidence", confidence))

    rule_view = rule_based_verdict_from_evidence(claim_text, subclaims, evidence_packet["sources"], reasoning["pendulum_band"])
    reasoning = align_reasoning_with_rules(reasoning, rule_view)
    reasoning["rule_engine"] = {
        "verdict": rule_view["verdict"],
        "confidence": rule_view["confidence"],
        "rationale": rule_view["rationale"],
        "stats": rule_view["stats"],
        "risk_flags": rule_view["risk_flags"],
    }

    split_view = split_evidence_vs_rumor(evidence_packet["sources"])
    reasoning.setdefault("evidence_assessment", {})
    reasoning["evidence_assessment"]["actual_evidence"] = reasoning["evidence_assessment"].get("actual_evidence") or split_view["actual_evidence"]
    reasoning["evidence_assessment"]["rumor_drivers"] = reasoning["evidence_assessment"].get("rumor_drivers") or split_view["rumor_drivers"]

    if not reasoning.get("consensus_strength"):
        stats = rule_view["stats"]
        support_count = stats["supportive_evidence"]
        contradict_count = stats["contradictory_evidence"]
        primary_support = stats["primary_supportive"]
        context_count = stats["rumor_or_context"]
        if support_count >= 3 and contradict_count == 0 and primary_support >= 1:
            reasoning["consensus_strength"] = "Strong agreement"
        elif support_count >= 2 and contradict_count <= 1:
            reasoning["consensus_strength"] = "Moderate agreement"
        elif support_count == 0 and contradict_count >= 1 and context_count > 0:
            reasoning["consensus_strength"] = "Context-heavy / unsubstantiated"
        elif support_count and contradict_count:
            reasoning["consensus_strength"] = "Mixed evidence"
        elif support_count:
            reasoning["consensus_strength"] = "Weak agreement"
        elif context_count > 0:
            reasoning["consensus_strength"] = "Context-heavy / unsubstantiated"
        else:
            reasoning["consensus_strength"] = "No clear consensus"

    if not reasoning.get("consensus_summary"):
        reasoning["consensus_summary"] = "This assessment reflects the balance of the reviewed sources rather than a single outlet or internal score."

    return reasoning


# -----------------------------
# UI
# -----------------------------


def normalize_claim_support(value: str) -> str:
    v = (value or "").strip().lower()
    mapping = {
        "supports": "Supports",
        "contradicts": "Contradicts",
        "mixed": "Mixed",
        "neutral": "Context",
        "irrelevant": "Context",
    }
    return mapping.get(v, "Context")


def map_source_quality_label(score: Any) -> str:
    try:
        val = float(score)
    except (TypeError, ValueError):
        return "Unknown"
    if val >= 4.25:
        return "High"
    if val >= 3.2:
        return "Medium"
    return "Low"


def render_claim_under_review(result: Dict[str, Any]) -> None:
    claim = (result.get("claim") or "").strip()
    subclaims = result.get("subclaims", []) or []
    if not claim and not subclaims:
        return
    st.markdown("### Claim under review")
    if claim:
        st.write(claim)
    if subclaims:
        with st.expander("Subclaims extracted", expanded=False):
            for item in subclaims:
                st.write(f"- {item}")


def canonical_source_buckets(sources: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    supporting: List[Dict[str, Any]] = []
    contradicting: List[Dict[str, Any]] = []
    contextual: List[Dict[str, Any]] = []
    for s in sources or []:
        cat = normalize_evidence_category(s.get("evidence_category", ""))
        support = (s.get("claim_support") or "").strip().lower()
        if cat in {"direct_evidence", "credible_reporting", "expert_analysis"} and support == "supports":
            supporting.append(s)
        elif cat in {"credible_contradiction", "denial_or_rebuttal"} or (
            cat in {"direct_evidence", "credible_reporting", "expert_analysis"} and support == "contradicts"
        ):
            contradicting.append(s)
        else:
            contextual.append(s)
    return {
        "supporting": supporting,
        "contradicting": contradicting,
        "contextual": contextual,
    }


def render_evidence_snapshot(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return
    buckets = canonical_source_buckets(sources)
    supporting = buckets["supporting"]
    contradicting = buckets["contradicting"]
    contextual = buckets["contextual"]

    st.markdown("### Evidence snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Supporting sources", len(supporting))
    c2.metric("Contradicting sources", len(contradicting))
    c3.metric("Contextual or mixed", len(contextual))

    sections = [
        ("Evidence supporting the claim", supporting),
        ("Evidence contradicting the claim", contradicting),
        ("Neutral or contextual evidence", contextual),
    ]
    for title, bucket in sections:
        st.markdown(f"**{title}**")
        if not bucket:
            st.caption("None surfaced in the reviewed set.")
            continue
        for src in bucket[:3]:
            summary = src.get("summary") or src.get("snippet") or src.get("title") or "Untitled"
            st.write(f"- {summary}")


def render_assessment_metrics(result: Dict[str, Any]) -> None:
    sources = result.get("sources", []) or []
    stats = ((result.get("rule_engine") or {}).get("stats") or {})
    primary = sum(1 for s in sources if (s.get("source_type") or "").lower() == "primary")
    high_quality = sum(1 for s in sources if map_source_quality_label(s.get("weighted_score")) == "High")
    contradictions = int(stats.get("contradictory_evidence", 0) or 0)
    elapsed = result.get("elapsed_seconds")

    st.markdown("### Assessment quality")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sources reviewed", len(sources))
    c2.metric("Primary sources", primary)
    c3.metric("High-quality sources", high_quality)
    c4.metric("Contradiction signals", contradictions)
    if elapsed is not None:
        st.caption(f"Completed in {elapsed:.1f}s")


def render_feedback_controls(result_key: str) -> None:
    feedback_store = st.session_state.setdefault("feedback_log", {})
    st.markdown("### Was this useful?")
    c1, c2, c3 = st.columns(3)
    if c1.button("Useful", key=f"fb_useful_{result_key}", use_container_width=True):
        feedback_store[result_key] = "useful"
    if c2.button("Not useful", key=f"fb_not_useful_{result_key}", use_container_width=True):
        feedback_store[result_key] = "not_useful"
    if c3.button("Sources weak", key=f"fb_sources_weak_{result_key}", use_container_width=True):
        feedback_store[result_key] = "sources_weak"
    if result_key in feedback_store:
        st.caption(f"Feedback captured: {feedback_store[result_key].replace('_', ' ')}")


def render_methodology_note() -> None:
    with st.expander("How Evidrai reached this assessment", expanded=False):
        st.write("Claim extraction → query generation → source retrieval → source ranking → contradiction check → verdict and confidence.")
        st.write("A rule engine then checks whether the final verdict is too strong for the evidence packet and can downgrade it to stay aligned with Evidrai's verification policy.")
        st.write("Confidence reflects the quantity, quality, directness, recency, and agreement level of the reviewed evidence. It is not a claim of certainty.")


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return
    st.markdown("### Sources reviewed")
    for src in sources:
        title = src.get("title", "Untitled")
        url = src.get("url", "")
        quality = map_source_quality_label(src.get("weighted_score"))
        stance = normalize_claim_support(src.get("claim_support", "context"))
        meta = f"{src.get('source_type', 'unknown').title()} • quality {quality} • stance {stance}"
        if src.get("weighted_score") is not None:
            meta += f" • score {src.get('weighted_score')}"
        if src.get("published_date"):
            meta += f" • {src['published_date']}"
        st.markdown(f"**[{title}]({url})**")
        st.caption(meta)
        if src.get("summary"):
            st.write(src["summary"])
        st.markdown("---")


def render_topline_block(title: str, verdict: str, confidence: Any, tldr: str, correction: str, badge: Optional[str] = None) -> None:
    st.markdown(f"## {title}")
    if badge:
        st.caption(badge)
    c1, c2 = st.columns(2)
    c1.metric("Verdict", str(verdict).replace("_", " ").title())
    c2.metric("Confidence", str(confidence))
    st.write(tldr or "No summary returned.")
    if correction:
        st.info(correction)



def map_pipeline_verdict(verdict: str) -> str:
    value = (verdict or "").strip().lower()
    mapping = {
        "true": "Supported",
        "supported": "Supported",
        "likely supported": "Likely supported",
        "partially_true": "Misleading framing",
        "partially true": "Misleading framing",
        "misleading": "Misleading framing",
        "false": "Not supported by credible evidence",
        "not supported by credible evidence": "Not supported by credible evidence",
        "weakly supported / likely incorrect": "Weakly supported / likely incorrect",
        "unverifiable": "Unverified",
        "unverified": "Unverified",
    }
    return mapping.get(value, "Unverified")


def map_confidence_label(value: Any) -> str:
    if isinstance(value, (int, float)) or str(value).isdigit():
        score = int(float(value))
        if score >= 70:
            return "High"
        if score >= 45:
            return "Medium"
        return "Low"
    text = str(value).strip().title()
    return text if text in {"High", "Medium", "Low"} else "Medium"


SERIOUS_ALLEGATION_TYPES = {"criminal", "corruption", "espionage", "foreign_agent", "misconduct_named_person"}

VERDICT_ORDER = {
    "Supported": 5,
    "Likely supported": 4,
    "Misleading framing": 3,
    "Unverified": 2,
    "Weakly supported / likely incorrect": 1,
    "Not supported by credible evidence": 0,
}

SOFT_CLAIM_FLAGS = {
    "opinion",
    "prediction",
    "rhetorical",
    "ambiguity",
    "vague",
    "motive_attribution",
    "value_judgment",
    "non_falsifiable",
}


def _normalized_text_set(values: List[str]) -> set[str]:
    out: set[str] = set()
    for v in values or []:
        cleaned = (v or "").strip().lower()
        if cleaned:
            out.add(cleaned)
    return out


def collect_risk_flags(subclaims: List[SubClaim]) -> set[str]:
    flags: set[str] = set()
    for sub in subclaims or []:
        flags.update(_normalized_text_set(sub.risk_flags))
    return flags


def is_soft_or_hard_to_verify_claim(subclaims: List[SubClaim]) -> bool:
    flags = collect_risk_flags(subclaims)
    if flags & SOFT_CLAIM_FLAGS:
        return True
    if not subclaims:
        return False
    soft_types = {"opinion", "prediction", "rhetorical"}
    return all((sub.claim_type or "other").lower() in soft_types for sub in (subclaims or []))


def is_concrete_factual_claim_text(claim_text: str) -> bool:
    text = (claim_text or "").strip().lower()
    if not text:
        return False
    soft_markers = (
        "might ", "may ", "could ", "perhaps", "maybe", "seems", "appears",
        "i think", "i believe", "probably", "arguably", "is a shill", "wants",
        "intends", "trying to", "because of his motives", "motivated by"
    )
    if any(marker in text for marker in soft_markers):
        return False
    fact_markers = (
        "accepted", "received", "took", "paid", "funded", "donated", "took money",
        "is", "was", "did", "has", "had"
    )
    return any(marker in text for marker in fact_markers)


def compute_evidence_stats(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        "supportive_evidence": 0,
        "contradictory_evidence": 0,
        "mixed_sources": 0,
        "rumor_or_context": 0,
        "primary_supportive": 0,
        "primary_contradictory": 0,
        "supportive_reporting": 0,
        "contradictory_reporting": 0,
        "allegation_or_context": 0,
        "denial_or_rebuttal": 0,
        "evidentiary_sources": 0,
        "unique_clusters": set(),
    }
    for s in sources or []:
        category = normalize_evidence_category(s.get("evidence_category", "irrelevant"))
        support = (s.get("claim_support") or "").strip().lower()
        cluster = (s.get("narrative_cluster") or s.get("url") or s.get("title") or "").strip().lower()
        if cluster:
            stats["unique_clusters"].add(cluster)

        is_primaryish = (s.get("source_type") or "").lower() in {"primary", "official", "government", "court", "parliament", "document", "record"}

        if category in {"direct_evidence", "credible_reporting", "expert_analysis"}:
            if support == "supports":
                stats["supportive_evidence"] += 1
                stats["evidentiary_sources"] += 1
                if category == "credible_reporting":
                    stats["supportive_reporting"] += 1
                if is_primaryish or category == "direct_evidence":
                    stats["primary_supportive"] += 1
            elif support == "contradicts":
                stats["contradictory_evidence"] += 1
                stats["evidentiary_sources"] += 1
                if category == "credible_reporting":
                    stats["contradictory_reporting"] += 1
                if is_primaryish or category == "direct_evidence":
                    stats["primary_contradictory"] += 1
            else:
                # Credible material that does not cleanly support or contradict should not count as evidence for the claim.
                stats["mixed_sources"] += 1
                stats["rumor_or_context"] += 1
                stats["allegation_or_context"] += 1
        elif category == "credible_contradiction":
            stats["contradictory_evidence"] += 1
            stats["evidentiary_sources"] += 1
            if is_primaryish:
                stats["primary_contradictory"] += 1
        elif category == "denial_or_rebuttal":
            stats["contradictory_evidence"] += 1
            stats["denial_or_rebuttal"] += 1
            stats["rumor_or_context"] += 1
            if is_primaryish:
                stats["primary_contradictory"] += 1
        elif category in {"reported_allegation", "contextual_signal", "rumor_amplification", "irrelevant"}:
            stats["rumor_or_context"] += 1
            stats["allegation_or_context"] += 1
            if support == "mixed":
                stats["mixed_sources"] += 1

    stats["unique_clusters"] = len(stats["unique_clusters"])
    return stats


def rule_based_verdict_from_evidence(
    claim_text: str,
    subclaims: List[SubClaim],
    sources: List[Dict[str, Any]],
    pendulum_band: str,
) -> Dict[str, Any]:
    stats = compute_evidence_stats(sources)
    flags = collect_risk_flags(subclaims)
    concrete_factual_claim = is_concrete_factual_claim_text(claim_text)
    soft_claim = is_soft_or_hard_to_verify_claim(subclaims) and not concrete_factual_claim
    serious_allegation = any((sub.claim_type or "").lower() in SERIOUS_ALLEGATION_TYPES for sub in (subclaims or [])) or concrete_factual_claim

    supportive = stats["supportive_evidence"]
    contradictory = stats["contradictory_evidence"]
    primary_supportive = stats["primary_supportive"]
    primary_contradictory = stats["primary_contradictory"]
    rumorish = stats["rumor_or_context"]
    mixed_sources = stats["mixed_sources"]

    verdict = "Unverified"
    confidence = "Low"
    rationale = "Evidence is limited or mixed."

    if supportive >= 2 and contradictory == 0 and primary_supportive >= 1:
        verdict = "Supported"
        confidence = "High" if supportive >= 3 else "Medium"
        rationale = "The reviewed evidence includes direct or high-quality support without material contradiction."
    elif supportive >= 2 and contradictory <= 1:
        verdict = "Likely supported"
        confidence = "Medium"
        rationale = "The balance of credible evidence leans supportive, but it is not fully closed."
    elif supportive == 0 and stats["evidentiary_sources"] == 0 and rumorish > 0:
        verdict = "Not supported by credible evidence" if serious_allegation and not soft_claim else "Unverified"
        confidence = "Low"
        rationale = "The packet is dominated by allegation, context, adjacency, denial, or rumor signals rather than substantive evidence."
    elif contradictory >= 2 and supportive == 0:
        verdict = "Not supported by credible evidence"
        confidence = "High" if primary_contradictory >= 1 else "Medium"
        rationale = "Credible contradiction outweighs the available support."
    elif contradictory >= 1 and supportive >= 1:
        if supportive >= contradictory:
            verdict = "Misleading framing"
            confidence = "Medium"
            rationale = "There appears to be a kernel of truth, but the claim overreaches or omits important counter-evidence."
        else:
            verdict = "Weakly supported / likely incorrect"
            confidence = "Medium"
            rationale = "Some support exists, but stronger evidence points the other way."
    elif pendulum_band == "Strongly evidenced" and supportive > 0:
        verdict = "Supported"
        confidence = "Medium"
        rationale = "The evidence pattern is strongly supportive, even if not all sources are fully primary."
    elif pendulum_band == "Mostly supported" and supportive > 0:
        verdict = "Likely supported"
        confidence = "Medium"
        rationale = "The evidence pattern is supportive overall, with some remaining uncertainty."
    elif pendulum_band == "Contradicted by evidence":
        verdict = "Not supported by credible evidence"
        confidence = "Medium"
        rationale = "The evidence packet contains credible material that conflicts with the claim."
    elif pendulum_band == "Weakly supported" and supportive > 0:
        verdict = "Weakly supported / likely incorrect"
        confidence = "Low"
        rationale = "The available support is weak and does not carry the claim cleanly."
    elif mixed_sources > 0 and supportive > 0 and contradictory > 0:
        verdict = "Misleading framing"
        confidence = "Low"
        rationale = "The evidence packet is mixed or partly interpretive rather than cleanly confirmatory."

    if soft_claim and verdict in {"Supported", "Likely supported", "Not supported by credible evidence"}:
        verdict = "Unverified" if verdict != "Supported" else "Likely supported"
        confidence = "Low" if verdict == "Unverified" else "Medium"
        rationale = "The claim is partly interpretive, predictive, rhetorical, or too vague for a stronger factual verdict."

    if "motive_attribution" in flags and verdict not in {"Supported", "Likely supported"}:
        verdict = "Unverified"
        confidence = "Low"
        rationale = "This claim depends on motive attribution, which usually cannot be verified cleanly from public evidence alone."

    if serious_allegation and supportive == 0 and primary_supportive == 0 and verdict == "Unverified" and rumorish >= 1:
        verdict = "Not supported by credible evidence"
        rationale = "This is a concrete allegation, but the packet does not contain credible substantiating evidence."

    if concrete_factual_claim and supportive == 0 and primary_supportive == 0 and contradictory >= 1 and rumorish >= 1:
        verdict = "Not supported by credible evidence"
        confidence = "Medium" if contradictory >= 1 else confidence
        rationale = "The claim is concrete, but the reviewed packet contains no credible substantiating evidence and at least some rebuttal or contradiction."

    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": rationale,
        "stats": stats,
        "soft_claim": soft_claim,
        "serious_allegation": serious_allegation,
        "risk_flags": sorted(flags),
    }


def align_reasoning_with_rules(reasoning: Dict[str, Any], rule_view: Dict[str, Any]) -> Dict[str, Any]:
    model_verdict = map_pipeline_verdict(reasoning.get("verified_verdict") or "Unverified")
    model_confidence = map_confidence_label(reasoning.get("verified_confidence") or "Low")
    rule_verdict = rule_view["verdict"]
    rule_confidence = rule_view["confidence"]

    if VERDICT_ORDER.get(model_verdict, 2) > VERDICT_ORDER.get(rule_verdict, 2):
        reasoning["verified_verdict"] = rule_verdict
        reasoning["verified_confidence"] = rule_confidence
    else:
        reasoning["verified_verdict"] = model_verdict
        if VERDICT_ORDER.get(model_verdict, 2) == VERDICT_ORDER.get(rule_verdict, 2):
            confidence_rank = {"Low": 0, "Medium": 1, "High": 2}
            reasoning["verified_confidence"] = rule_confidence if confidence_rank.get(rule_confidence, 0) < confidence_rank.get(model_confidence, 0) else model_confidence
        else:
            reasoning["verified_confidence"] = model_confidence

    if rule_view["soft_claim"] and reasoning.get("verified_confidence") == "High":
        reasoning["verified_confidence"] = "Medium"
    if rule_view["soft_claim"] and reasoning.get("verified_verdict") in {"Supported", "Likely supported", "Not supported by credible evidence"}:
        if reasoning.get("verified_verdict") != "Supported":
            reasoning["verified_verdict"] = "Unverified"
            reasoning["verified_confidence"] = "Low"

    stats = rule_view["stats"]

    # Collapse all downstream labels onto one canonical evidence-state.
    if stats.get("supportive_evidence", 0) == 0 and stats.get("evidentiary_sources", 0) == 0 and stats.get("rumor_or_context", 0) > 0:
        contradiction_like = int(stats.get("contradictory_evidence", 0) or 0)
        reasoning["pendulum_band"] = "Context-heavy / unsubstantiated"
        reasoning["pendulum_explanation"] = f"0 evidentiary source(s), {contradiction_like} contradiction signal(s), {stats.get('rumor_or_context', 0)} rumor/context signal(s)"
        reasoning["consensus_strength"] = "Context-heavy / unsubstantiated"
        reasoning["consensus_summary"] = "The reviewed packet contains allegation, context, adjacency, denial, or rebuttal material, but no direct substantiating evidence for the claim as stated."
        if not rule_view["soft_claim"]:
            reasoning["verified_verdict"] = "Not supported by credible evidence"
            reasoning["verified_confidence"] = "Medium" if contradiction_like >= 1 else "Low"

    concrete_override = (
        not rule_view["soft_claim"]
        and stats.get("supportive_evidence", 0) == 0
        and stats.get("primary_supportive", 0) == 0
        and stats.get("contradictory_evidence", 0) >= 1
        and stats.get("rumor_or_context", 0) >= 1
        and (reasoning.get("pendulum_band") or "").strip() == "Context-heavy / unsubstantiated"
    )
    if concrete_override:
        reasoning["verified_verdict"] = "Not supported by credible evidence"
        if map_confidence_label(reasoning.get("verified_confidence") or "Low") == "Low":
            reasoning["verified_confidence"] = "Medium"

    evidence_assessment = reasoning.setdefault("evidence_assessment", {})
    evidence_assessment.setdefault("evidence_gaps", [])
    if stats["supportive_evidence"] == 0:
        evidence_assessment["evidence_gaps"].append("No direct or clearly supportive evidentiary source was identified in the reviewed packet.")
    if rule_view["soft_claim"]:
        evidence_assessment["evidence_gaps"].append("Part of the claim is interpretive, rhetorical, predictive, or otherwise difficult to verify directly.")

    explanation_note = rule_view["rationale"]
    final_explanation = (reasoning.get("final_explanation") or "").strip()
    if explanation_note and explanation_note not in final_explanation:
        reasoning["final_explanation"] = (final_explanation + "\n\nRule-based check: " + explanation_note).strip()

    summary = (reasoning.get("consensus_summary") or "").strip()
    if explanation_note and explanation_note not in summary:
        reasoning["consensus_summary"] = (summary + " " + explanation_note).strip()

    return reasoning

def normalize_evidence_category(category: str) -> str:
    c = (category or "").strip().lower()
    mapping = {
        "documented_support": "direct_evidence",
        "direct_evidence": "direct_evidence",
        "primary_evidence": "direct_evidence",
        "credible_reporting": "credible_reporting",
        "expert_analysis": "expert_analysis",
        "reported_allegation": "reported_allegation",
        "allegation": "reported_allegation",
        "context_only": "contextual_signal",
        "contextual_signal": "contextual_signal",
        "denial": "denial_or_rebuttal",
        "denial_or_rebuttal": "denial_or_rebuttal",
        "credible_contradiction": "credible_contradiction",
        "contradiction": "credible_contradiction",
        "rumor_amplification": "rumor_amplification",
        "irrelevant": "irrelevant",
    }
    return mapping.get(c, "irrelevant")


def source_bucket_multiplier(source_type: str, domain: str) -> float:
    stype = (source_type or "").lower()
    d = (domain or "").lower()
    if stype in {"official", "government", "court", "parliament"}:
        return 1.5
    if stype in {"primary", "document", "record"}:
        return 1.4
    if any(x in d for x in ["reuters.com", "apnews.com", "bbc.", "ft.com", "nytimes.com", "theguardian.com", "lemonde.fr", "france24.com"]):
        return 1.3
    if stype in {"expert_publication", "journal", "think_tank"}:
        return 1.2
    if stype in {"commentary", "analysis"}:
        return 1.0
    if stype in {"local_news"}:
        return 0.9
    if stype in {"forum", "reddit", "quora"}:
        return 0.3
    if stype in {"social", "social_media", "x", "twitter", "facebook", "instagram", "tiktok"}:
        return 0.4
    return 0.7


def evidence_pendulum(sources: List[Dict[str, Any]], claim_type: str = "other") -> Dict[str, Any]:
    weights = {
        "direct_evidence": 3.0,
        "credible_reporting": 2.0,
        "expert_analysis": 1.0,
        "reported_allegation": 0.0,
        "contextual_signal": 0.0,
        "denial_or_rebuttal": -1.0,
        "credible_contradiction": -3.0,
        "rumor_amplification": 0.0,
        "irrelevant": 0.0,
    }
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for s in sources:
        key = (s.get("narrative_cluster") or f"{s.get('domain','')}|{s.get('evidence_category','')}|{s.get('claim_support','')}").strip().lower()
        grouped.setdefault(key, []).append(s)

    score = 0.0
    support_count = contradiction_count = rumor_count = 0
    decay = [1.0, 0.7, 0.4, 0.2]
    for items in grouped.values():
        items = sorted(items, key=lambda x: source_bucket_multiplier(x.get("source_type",""), x.get("domain","")), reverse=True)
        for idx, s in enumerate(items):
            cat = normalize_evidence_category(s.get("evidence_category", "irrelevant"))
            mult = source_bucket_multiplier(x.get("source_type",""), x.get("domain","")) if False else source_bucket_multiplier(s.get("source_type",""), s.get("domain",""))
            contribution = weights.get(cat, 0.0) * mult * (decay[idx] if idx < len(decay) else 0.1)
            score += contribution
            if cat in {"direct_evidence","credible_reporting","expert_analysis"} and (s.get("claim_support") or "").strip().lower() == "supports" and contribution > 0:
                support_count += 1
            if cat in {"credible_contradiction","denial_or_rebuttal"} or (cat in {"direct_evidence","credible_reporting","expert_analysis"} and (s.get("claim_support") or "").strip().lower() == "contradicts"):
                contradiction_count += 1
            if cat in {"reported_allegation","contextual_signal","rumor_amplification","denial_or_rebuttal"}:
                rumor_count += 1

    adjusted = score
    if (claim_type or "").lower() in SERIOUS_ALLEGATION_TYPES and adjusted > 0:
        adjusted -= 3.0

    if support_count == 0 and rumor_count > 0:
        if contradiction_count >= 1:
            band = "Context-heavy / unsubstantiated"
        else:
            band = "Context-heavy / unsubstantiated"
    elif contradiction_count >= 2 and adjusted <= -6:
        band = "Contradicted by evidence"
    elif adjusted >= 8:
        band = "Strongly evidenced"
    elif adjusted >= 4:
        band = "Mostly supported"
    elif adjusted >= -3:
        band = "Mixed / uncertain"
    elif adjusted >= -7:
        band = "Weakly supported"
    else:
        band = "Unsubstantiated rumor"

    evidentiary_count = support_count if support_count > 0 else 0
    explanation = f"{evidentiary_count} evidentiary source(s), {contradiction_count} contradiction signal(s), {rumor_count} rumor/context signal(s)"
    return {"band": band, "score": round(adjusted, 2), "explanation": explanation}


def map_pendulum_to_verified_verdict(band: str) -> str:
    mapping = {
        "Strongly evidenced": "Supported",
        "Mostly supported": "Likely supported",
        "Mixed / uncertain": "Misleading framing",
        "Weakly supported": "Weakly supported / likely incorrect",
        "Unsubstantiated rumor": "Unverified",
        "Context-heavy / unsubstantiated": "Not supported by credible evidence",
        "Contradicted by evidence": "Not supported by credible evidence",
    }
    return mapping.get(band, "Unverified")


def split_evidence_vs_rumor(sources: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    actual_evidence: List[str] = []
    rumor_drivers: List[str] = []
    for s in sources:
        cat = normalize_evidence_category(s.get("evidence_category", ""))
        summary = s.get("summary") or s.get("snippet") or ""
        if not summary:
            continue
        if cat in {"direct_evidence", "credible_reporting", "expert_analysis", "credible_contradiction"}:
            actual_evidence.append(summary)
        elif cat in {"reported_allegation", "contextual_signal", "rumor_amplification", "denial_or_rebuttal"}:
            rumor_drivers.append(summary)
    return {"actual_evidence": actual_evidence[:6], "rumor_drivers": rumor_drivers[:6]}


def render_pendulum(band: str) -> None:
    labels = ["Unsubstantiated rumor", "Weakly supported", "Mixed / uncertain", "Context-heavy / unsubstantiated", "Strongly evidenced"]
    pos_map = {label: idx for idx, label in enumerate(labels)}
    # Map legacy mostly-supported displays into the nearest visible bucket only when needed.
    if band == "Mostly supported":
        pos = 3
    else:
        pos = pos_map.get(band, 2)
    cols = st.columns(5)
    for i, label in enumerate(labels):
        cols[i].markdown(f"**⬤ {label}**" if i == pos else f"◯ {label}")


def render_consensus_block(result: Dict[str, Any]) -> None:
    strength = (result.get("consensus_strength") or "").strip()
    summary = (result.get("consensus_summary") or "").strip()
    if not strength and not summary:
        return
    st.markdown("### Consensus across evidence")
    if strength:
        st.write(f"**{strength}**")
    if summary:
        st.write(summary)


def render_pipeline_result(result: Dict[str, Any]) -> None:
    verified_verdict = map_pipeline_verdict(result.get("verified_verdict") or result.get("verdict") or "unverifiable")
    verified_confidence = map_confidence_label(result.get("verified_confidence") or result.get("confidence", 0))
    render_topline_block(
        "Verified assessment",
        verified_verdict,
        verified_confidence,
        result.get("tldr", "No summary returned."),
        result.get("one_line_correction", ""),
        badge="Deep cross-evidence review completed.",
    )

    render_claim_under_review(result)
    render_assessment_metrics(result)
    rule_engine = result.get("rule_engine") or {}
    if rule_engine.get("rationale"):
        st.markdown("### Rule engine check")
        st.write(rule_engine.get("rationale"))
        if rule_engine.get("risk_flags"):
            st.caption("Risk flags: " + ", ".join(rule_engine.get("risk_flags")[:8]))
    render_evidence_snapshot(result.get("sources", []) or [])

    stats = (rule_engine.get("stats") or {})
    band = result.get("pendulum_band", "")
    explanation = result.get("pendulum_explanation", "")
    if stats.get("supportive_evidence", 0) == 0 and stats.get("evidentiary_sources", 0) == 0 and stats.get("rumor_or_context", 0) > 0:
        band = "Context-heavy / unsubstantiated"
        explanation = f"0 evidentiary source(s), {int(stats.get('contradictory_evidence', 0) or 0)} contradiction signal(s), {int(stats.get('rumor_or_context', 0) or 0)} rumor/context signal(s)"
    if band:
        st.markdown("### Evidence position")
        render_pendulum(band)
        if explanation:
            st.caption(explanation)

    render_consensus_block(result)

    rs = result.get("reasoning_summary", {}) or {}
    ea = result.get("evidence_assessment", {}) or {}
    with st.expander("Evidence summary", expanded=False):
        for label, key in [("Supported", "supported_points"), ("Contradicted", "contradicted_points"), ("Uncertain", "uncertain_points")]:
            vals = rs.get(key, []) or []
            if vals:
                st.markdown(f"**{label}**")
                for v in vals:
                    st.write(f"- {v}")
        actual_evidence = ea.get("actual_evidence", []) or []
        if actual_evidence:
            st.markdown("**What actual evidence exists**")
            for v in actual_evidence:
                st.write(f"- {v}")
        rumor_drivers = ea.get("rumor_drivers", []) or []
        if rumor_drivers:
            st.markdown("**Why some people may think this is true**")
            for v in rumor_drivers:
                st.write(f"- {v}")
        evidence_gaps = ea.get("evidence_gaps", []) or []
        if evidence_gaps:
            st.markdown("**Remaining evidence gaps**")
            for v in evidence_gaps:
                st.write(f"- {v}")

    tab1, tab2, tab3 = st.tabs(["Sources", "Reasoning", "Why it spreads"])
    with tab1:
        with st.expander("Sources and weighting", expanded=True):
            render_sources(result.get("sources", []))
        queries = result.get("queries", []) or []
        if queries:
            with st.expander("Search queries used", expanded=False):
                for q in queries:
                    st.write(f"- {q}")
    with tab2:
        if result.get("final_explanation"):
            st.markdown("### Assessment")
            st.write(result["final_explanation"])
        patterns = result.get("misinformation_patterns", []) or []
        if patterns:
            with st.expander("Misinformation patterns", expanded=False):
                st.write(" • ".join(patterns))
    with tab3:
        spreads = result.get("why_this_claim_spreads", []) or []
        if spreads:
            for item in spreads:
                st.write(f"- {item}")
        else:
            st.caption("No additional spread analysis returned.")

    render_feedback_controls(result.get("result_id", "latest"))
    render_methodology_note()


def render_provisional_result(data: Dict[str, Any], source_url: str) -> None:
    render_topline_block(
        "Provisional assessment",
        data.get("verdict", "Unverified"),
        data.get("confidence", "Low"),
        data.get("tldr") or data.get("summary") or "No summary returned.",
        data.get("one_line_correction") or data.get("user_takeaway") or "Deep verification may refine this answer.",
        badge="Fast first-pass assessment. Deep verification may update the verdict.",
    )

    if source_url:
        st.caption(f"Source link provided: {source_url}")

    with st.expander("Fast-pass notes", expanded=False):
        for heading, key in [
            ("Assessment", "summary"),
            ("Why this may seem convincing", "why_convincing"),
            ("Evidence access note", "evidence_access_note"),
            ("What would change the verdict", "what_would_change_verdict"),
        ]:
            if data.get(key):
                st.markdown(f"**{heading}**")
                st.write(data[key])

        evidence_types = data.get("evidence_types", []) or []
        if evidence_types:
            st.markdown("**Evidence breakdown**")
            for item in evidence_types:
                st.markdown(f"- **{item.get('type', 'Unknown')}** — {item.get('impact', '')}")
                if item.get("note"):
                    st.caption(item["note"])

    render_feedback_controls(data.get("result_id", "quick_latest"))


def render_legacy_result(data: Dict[str, Any], source_url: str) -> None:
    render_provisional_result(data, source_url)



def main() -> None:
    st.set_page_config(page_title="Evidrai", layout="wide")
    st.title("🔎 Evidrai — Claim Check")
    st.caption("Assess the evidence behind a claim, story, or post — not just how confidently it is repeated.")

    llm = OpenAICompatibleClient()
    search = TavilySearchClient()

    if "evidrai_cache" not in st.session_state:
        st.session_state["evidrai_cache"] = {}
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = None
    if "feedback_log" not in st.session_state:
        st.session_state["feedback_log"] = {}

    with st.sidebar:
        st.header("Settings")
        detail_mode = st.radio("Output mode", ["Simple", "Detailed"], index=0)
        category = st.selectbox(
            "Claim category",
            ["auto-detect", "politics", "celebrity", "health", "science", "finance", "history", "general"],
            index=0,
        )
        verification_mode = st.selectbox("Verification depth", ["Auto", "Fast", "Deep"], index=0)
        st.markdown("---")
        st.caption("Auto uses deep verification when search is configured, otherwise it falls back to the fast first-pass flow.")
        st.markdown("---")
        st.caption("The product is optimized around claim → evidence → verdict. Fast mode gives a quick first pass. Deep mode shows the evidence pipeline.")

    claim = st.text_area(
        "Paste a claim, link description, quote, or content to assess",
        placeholder="Paste a claim, rumor, headline, quote, or a short description of a video/post here...",
        height=140,
        key="claim_input",
    )
    source_url = st.text_input(
        "Optional source link",
        placeholder="Paste a YouTube, article, podcast, Facebook, or Instagram URL if helpful...",
        key="source_url_input",
    )
    st.caption("Tip: if you add a link, also paste the key quote or claim above. The app reasons better when the central claim is explicit.")
    with st.expander("Verdict scale", expanded=False):
        st.write("Supported, Likely supported, Misleading framing, Weakly supported or likely incorrect, Not supported by credible evidence, or Unverified.")

    if st.button("Check claim", type="primary", use_container_width=True):
        cleaned_claim = (claim or "").strip()
        cleaned_source_url = (source_url or "").strip()
        if not cleaned_claim and not cleaned_source_url:
            st.error("Please enter a claim, some content, or a source link first.")
            return
        if cleaned_source_url and not is_probable_url(cleaned_source_url):
            st.error("The source link does not look like a valid URL. Please include http:// or https://")
            return
        if not llm.configured:
            st.error("OPENAI_API_KEY is not configured in your app secrets or environment.")
            return

        analysis_input = build_analysis_input(cleaned_claim, cleaned_source_url)
        use_search = search.configured if verification_mode == "Auto" else verification_mode == "Deep"
        if verification_mode == "Deep" and not search.configured:
            st.error("Deep mode requires TAVILY_API_KEY to be configured.")
            return

        cache_key = stable_request_key(analysis_input, category, verification_mode, use_search, detail_mode)
        cache = st.session_state["evidrai_cache"]
        if cache_key in cache:
            st.session_state["last_results"] = cache[cache_key]
        else:
            try:
                started_at = time.time()
                status = st.status("Starting assessment...", expanded=True)
                with status:
                    st.write("Running fast first-pass assessment...")
                quick_result = run_quick_pass(analysis_input, category, llm)
                quick_result["result_id"] = f"quick_{cache_key}"

                full_result = None
                if use_search:
                    with status:
                        st.write("Extracting the core claim and subclaims...")
                        st.write("Retrieving and ranking external sources...")
                        st.write("Checking for contradiction, support, and uncertainty...")
                        st.write("Generating the verified assessment...")
                    full_result = run_claim_pipeline(analysis_input, llm, search)
                    full_result["elapsed_seconds"] = time.time() - started_at
                    full_result["result_id"] = f"deep_{cache_key}"
                    status.update(label="Assessment complete", state="complete", expanded=False)
                else:
                    quick_result["elapsed_seconds"] = time.time() - started_at
                    status.update(label="Fast assessment complete", state="complete", expanded=False)

                saved = {
                    "quick_result": quick_result,
                    "full_result": full_result,
                    "source_url": cleaned_source_url,
                }
                cache[cache_key] = saved
                st.session_state["last_results"] = saved
            except requests.HTTPError as exc:
                try:
                    detail = exc.response.text[:500]
                except Exception:
                    detail = str(exc)
                st.error(f"API error: {detail}")
            except Exception as exc:
                st.error(f"Error: {exc}")

    saved = st.session_state.get("last_results")
    if saved:
        if saved.get("quick_result"):
            render_provisional_result(saved["quick_result"], saved.get("source_url", ""))
        if saved.get("full_result"):
            render_pipeline_result(saved["full_result"])


if __name__ == "__main__":
    main()
