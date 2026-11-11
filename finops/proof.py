"""Proof mode: show, on the customer's own traffic, that a cheaper model is
good enough before anyone switches.

Flow:
    1. Sample logged requests (prompt + the production response) per routable
       task type, stratified so each task gets up to ``per_task`` samples.
    2. Get the candidate model's answer, either by replaying the prompt
       against an OpenAI-compatible endpoint (OpenAI, Anthropic's and
       Google's OpenAI-compatible APIs, or the customer's LiteLLM proxy) or
       from a file of pre-computed outputs (``--candidate-outputs``), which
       lets the customer run step 2 in their own pipeline.
    3. Score candidate vs. production answer with a task-appropriate check:
         labels (classification, tagging, routing)  exact normalized match
         structured (extraction, formatting)         JSON field-level match
         free text (summarization, QA, ...)          LLM judge if configured,
                                                     else lexical F1 (flagged weak)
    4. Verdict per (task, candidate) using the Wilson 95% interval:
         pass         lower bound ≥ threshold
         fail         upper bound < threshold
         inconclusive otherwise (collect more samples)

The audit reads the saved results (``finops audit --proof proof.json``):
passed routes become *verified* savings, failed routes are blocked.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import string
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from finops.models import UsageRecord
from finops.policy import Policy
from finops.pricing import PriceBook

__all__ = ["ProofSample", "ProofGroup", "ProofReport", "select_samples", "run_proof",
           "OpenAICompatibleClient", "load_candidate_outputs", "load_proof_index",
           "score_labels", "score_json", "token_f1", "wilson_interval"]

LABEL_TASKS = {"classification", "tagging", "routing", "deduplication"}
JSON_TASKS = {"extraction", "formatting"}
F1_PASS = 0.6
JSON_PASS = 0.9


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_PUNCT = str.maketrans("", "", string.punctuation.replace("_", "").replace("-", ""))


def _norm(text: str) -> str:
    return " ".join(str(text).lower().translate(_PUNCT).split())


def score_labels(reference: str, candidate: str) -> Tuple[bool, str]:
    """Exact match after normalization; also accepts the label as the first
    line, or as a whole word in a very short answer ("Category: billing")."""
    ref, cand = _norm(reference), _norm(candidate)
    if not ref:
        return False, "empty reference"
    if ref == cand:
        return True, "exact"
    first = _norm(str(candidate).strip().splitlines()[0]) if str(candidate).strip() else ""
    if ref == first:
        return True, "first line"
    words = cand.split()
    if len(ref.split()) == 1 and len(words) <= 4 and ref in words:
        return True, "label in short answer"
    return False, f"expected '{reference.strip()[:40]}', got '{str(candidate).strip()[:40]}'"


def _extract_json(text: str) -> Optional[object]:
    decoder = json.JSONDecoder()
    text = str(text)
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def _flatten(obj: object, prefix: str = "") -> Dict[str, object]:
    out: Dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _same_value(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    return _norm(str(a)) == _norm(str(b))


def score_json(reference: str, candidate: str) -> Tuple[bool, str]:
    """Field-level accuracy against the production JSON output."""
    ref, cand = _extract_json(reference), _extract_json(candidate)
    if ref is None:
        ok, why = token_f1(reference, candidate)
        return ok, "reference not JSON; " + why
    if cand is None:
        return False, "candidate output is not valid JSON"
    rf, cf = _flatten(ref), _flatten(cand)
    if not rf:
        return rf == cf, "empty reference object"
    hits = sum(1 for k, v in rf.items() if k in cf and _same_value(v, cf[k]))
    acc = hits / len(rf)
    return acc >= JSON_PASS, f"field accuracy {acc:.0%}"


def token_f1(reference: str, candidate: str) -> Tuple[bool, str]:
    ref, cand = _norm(reference).split(), _norm(candidate).split()
    if not ref or not cand:
        return False, "empty text"
    common = defaultdict(int)
    for w in ref:
        common[w] += 1
    overlap = 0
    for w in cand:
        if common[w] > 0:
            overlap += 1
            common[w] -= 1
    if overlap == 0:
        return False, "F1 0.00"
    p, r = overlap / len(cand), overlap / len(ref)
    f1 = 2 * p * r / (p + r)
    return f1 >= F1_PASS, f"F1 {f1:.2f}"


def wilson_interval(passes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = passes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


# ---------------------------------------------------------------------------
# Candidate sources
# ---------------------------------------------------------------------------

class OpenAICompatibleClient:
    """Minimal stdlib client for any OpenAI-compatible /chat/completions API.

    Works with OpenAI, Anthropic's and Google's OpenAI-compatible endpoints,
    and LiteLLM / OpenRouter proxies. The API key is read by the caller from
    an environment variable and is never logged.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0,
                 max_retries: int = 3) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, model: str, messages: List[Dict[str, object]],
             max_tokens: Optional[int] = None) -> str:
        body: Dict[str, object] = {"model": model, "messages": messages, "temperature": 0}
        if max_tokens:
            body["max_tokens"] = max_tokens
        data = json.dumps(body).encode("utf-8")
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=data, method="POST", headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                return str(content or "")
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                detail = exc.read().decode("utf-8", "replace")[:200]
                raise RuntimeError(f"HTTP {exc.code} from {self.url}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"Cannot reach {self.url}: {exc}") from exc
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Unexpected response shape from {self.url}") from exc
        raise RuntimeError("unreachable")


def prompt_key(rec: UsageRecord) -> str:
    """Stable id for a record: request_id, else a hash of the prompt."""
    if rec.request_id:
        return rec.request_id
    return "sha1:" + hashlib.sha1((rec.prompt or "").encode("utf-8")).hexdigest()


def load_candidate_outputs(path: str) -> Dict[str, str]:
    """JSONL/JSON of {"request_id" | "prompt_sha1": ..., "output": ...}."""
    from finops.ingest.tabular import read_rows
    out: Dict[str, str] = {}
    for row in read_rows(path):
        if not isinstance(row, dict) or "output" not in row:
            continue
        key = row.get("request_id") or (f"sha1:{row['prompt_sha1']}" if row.get("prompt_sha1") else None)
        if key:
            out[str(key)] = str(row["output"])
    return out


# ---------------------------------------------------------------------------
# Sampling and running
# ---------------------------------------------------------------------------

@dataclass
class ProofSample:
    key: str
    task_type: str
    production_model: str
    candidate: str
    passed: Optional[bool] = None
    reason: str = ""
    error: str = ""
    prompt_preview: str = ""
    reference_preview: str = ""
    candidate_preview: str = ""


@dataclass
class ProofGroup:
    task_type: str
    candidate: str
    n: int = 0
    passed: int = 0
    errors: int = 0
    scorer: str = ""
    pass_rate: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 1.0
    verdict: str = "inconclusive"
    failures: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ProofReport:
    threshold: float
    groups: List[ProofGroup] = field(default_factory=list)
    samples_considered: int = 0
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, object]:
        return {"version": 1, "threshold": self.threshold,
                "samples_considered": self.samples_considered, "notes": self.notes,
                "results": [asdict(g) for g in self.groups]}


def _targets_for(rec: UsageRecord, policy: Policy, book: PriceBook) -> List[str]:
    price = book.resolve(rec.model)
    if price is None:
        return []
    from finops.llm_audit import route_candidates
    return route_candidates(price, rec.task_type, policy)


def select_samples(records: Sequence[UsageRecord], policy: Policy, book: PriceBook,
                   per_task: int = 50, seed: int = 7,
                   candidate_override: Optional[str] = None
                   ) -> Dict[Tuple[str, str], List[UsageRecord]]:
    """Stratified random sample of replayable records per (task, candidate)."""
    pools: Dict[Tuple[str, str], List[UsageRecord]] = defaultdict(list)
    for rec in records:
        if not (rec.prompt and rec.response):
            continue
        targets = [candidate_override] if candidate_override else _targets_for(rec, policy, book)
        for target in targets:
            pools[(rec.task_type, target)].append(rec)
    rng = random.Random(seed)
    return {k: (rng.sample(v, per_task) if len(v) > per_task else list(v))
            for k, v in pools.items()}


def _scorer_for(task: str, judge: Optional[Callable[[UsageRecord, str], Tuple[bool, str]]]):
    if task in LABEL_TASKS:
        return "label match", lambda rec, out: score_labels(rec.response or "", out)
    if task in JSON_TASKS:
        return "json field match", lambda rec, out: score_json(rec.response or "", out)
    if judge:
        return "llm judge", judge
    return "lexical F1 (weak)", lambda rec, out: token_f1(rec.response or "", out)


def make_judge(client: OpenAICompatibleClient, judge_model: str
               ) -> Callable[[UsageRecord, str], Tuple[bool, str]]:
    def judge(rec: UsageRecord, candidate: str) -> Tuple[bool, str]:
        prompt = (
            "You are grading whether a cheaper model's answer is an acceptable substitute "
            "for a production answer.\n\n"
            f"TASK INPUT:\n{(rec.prompt or '')[:6000]}\n\n"
            f"PRODUCTION ANSWER (reference):\n{(rec.response or '')[:4000]}\n\n"
            f"CANDIDATE ANSWER:\n{candidate[:4000]}\n\n"
            "Is the candidate at least as correct, complete and useful for this task as the "
            "reference? Reply with JSON only: {\"verdict\": \"pass\" or \"fail\", "
            "\"reason\": \"<10 words\"}")
        text = client.chat(judge_model, [{"role": "user", "content": prompt}], max_tokens=100)
        obj = _extract_json(text)
        if isinstance(obj, dict) and str(obj.get("verdict", "")).lower() in ("pass", "fail"):
            return obj["verdict"].lower() == "pass", "judge: " + str(obj.get("reason", ""))[:80]
        return False, "judge returned unparseable output"
    return judge


def _messages(rec: UsageRecord) -> List[Dict[str, object]]:
    msgs = rec.extra.get("messages")
    if isinstance(msgs, list) and msgs:
        return msgs
    return [{"role": "user", "content": rec.prompt or ""}]


def estimate_replay_cost(samples: Dict[Tuple[str, str], List[UsageRecord]],
                         book: PriceBook) -> float:
    total = 0.0
    for (_, candidate), recs in samples.items():
        price = book.get(candidate)
        if price:
            total += sum(price.cost(r.input_tokens / max(1, r.requests),
                                    r.output_tokens / max(1, r.requests)) for r in recs)
    return total


def run_proof(records: Sequence[UsageRecord], policy: Policy, book: PriceBook, *,
              per_task: int = 50, threshold: Optional[float] = None,
              client: Optional[OpenAICompatibleClient] = None,
              candidate_outputs: Optional[Dict[str, str]] = None,
              model_ids: Optional[Dict[str, str]] = None,
              judge: Optional[Callable[[UsageRecord, str], Tuple[bool, str]]] = None,
              candidate_override: Optional[str] = None,
              concurrency: int = 4, seed: int = 7,
              progress: Optional[Callable[[str], None]] = None) -> ProofReport:
    """Run the quality proof. Exactly one of *client* / *candidate_outputs*
    must be given. *model_ids* maps canonical model names to the id the
    endpoint expects (e.g. ``{"claude-haiku-5-5": "anthropic/claude-haiku-5-5"}``
    for a LiteLLM proxy)."""
    if (client is None) == (candidate_outputs is None):
        raise ValueError("Provide either a client or candidate_outputs.")
    threshold = policy.proof_pass_threshold if threshold is None else threshold
    model_ids = model_ids or {}
    samples = select_samples(records, policy, book, per_task, seed, candidate_override)
    report = ProofReport(threshold=threshold,
                         samples_considered=sum(len(v) for v in samples.values()))
    replayable = sum(1 for r in records if r.prompt and r.response)
    if not samples:
        report.notes.append(
            f"No replayable samples: {replayable} records have both prompt and response, "
            "and none belong to a routable task. Log prompts/responses (e.g. LiteLLM "
            "store_prompts_in_spend_logs) and tag task types.")
        return report

    for (task, candidate), recs in sorted(samples.items()):
        scorer_name, scorer = _scorer_for(task, judge)
        group = ProofGroup(task_type=task, candidate=candidate, scorer=scorer_name)

        def evaluate(rec: UsageRecord) -> ProofSample:
            sample = ProofSample(key=prompt_key(rec), task_type=task,
                                 production_model=rec.model, candidate=candidate,
                                 prompt_preview=(rec.prompt or "")[:160],
                                 reference_preview=(rec.response or "")[:160])
            try:
                if candidate_outputs is not None:
                    out = candidate_outputs.get(sample.key)
                    if out is None:
                        sample.error = "no candidate output for this request"
                        return sample
                else:
                    out = client.chat(model_ids.get(candidate, candidate), _messages(rec))
                sample.candidate_preview = out[:160]
                sample.passed, sample.reason = scorer(rec, out)
            except Exception as exc:  # noqa: BLE001 - one bad sample must not end the run
                sample.error = str(exc)[:200]
            return sample

        if progress:
            progress(f"{task} → {candidate}: scoring {len(recs)} samples ({scorer_name})")
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            results = list(pool.map(evaluate, recs))
        for s in results:
            if s.error:
                group.errors += 1
                continue
            group.n += 1
            if s.passed:
                group.passed += 1
            elif len(group.failures) < 5:
                group.failures.append({"key": s.key, "reason": s.reason,
                                       "prompt": s.prompt_preview,
                                       "production": s.reference_preview,
                                       "candidate": s.candidate_preview})
        group.pass_rate = group.passed / group.n if group.n else 0.0
        group.ci_low, group.ci_high = wilson_interval(group.passed, group.n)
        if group.n and group.ci_high < threshold:
            group.verdict = "fail"
        elif group.n >= policy.proof_min_samples and group.ci_low >= threshold:
            group.verdict = "pass"
        else:
            group.verdict = "inconclusive"
        if scorer_name.startswith("lexical") and group.verdict == "pass":
            group.verdict = "inconclusive"
            report.notes.append(f"{task}: lexical overlap can't prove free-text quality; "
                                "rerun with --judge-model to get a pass verdict.")
        if group.errors:
            report.notes.append(f"{task} → {candidate}: {group.errors} sample(s) errored "
                                "and were excluded.")
        report.groups.append(group)
    return report


def load_proof_index(path: str) -> Dict[Tuple[str, str], str]:
    """Read a saved proof JSON into {(task_type, candidate): verdict}."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    index: Dict[Tuple[str, str], str] = {}
    for g in data.get("results", []):
        index[(str(g["task_type"]).lower(), str(g["candidate"]).lower())] = str(g["verdict"])
    return index
