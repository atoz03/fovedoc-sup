#!/usr/bin/env python3
"""DMR de-risk 校准用的轻量 LLM 正确性判分（pred vs reference）。

不是正式 grounded judge（那条需要 grounded-json 记录 + atomic facts）；这里只回答一个
校准问题：V-MQAR 在 oracle pages+memory 下的 0.17 是真天花板，还是 exact-match 地板。
走与正式 judge 相同的链路：gpt-5.5 / OpenAI-compatible endpoint / responses / 顶层 instructions。
verdict = correct | partial | incorrect → score 1.0 / 0.5 / 0.0，partial 用来暴露
"跨页只对了一半"（关联半成立）的情况。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Endpoint and model come from the environment so no credential or private proxy host is ever
# written to this file. OPENAI_BASE_URL is the base (".../v1").
# The historical default is kept only so old invocations still resolve — it is the proxy whose
# outage silently wrote 192 `verdict: "error"` rows into the B2 run, so prefer setting the env.
_BASE_URL = str(os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
JUDGE_MODEL = str(os.environ.get("DMR_JUDGE_MODEL") or "gpt-5.5")

# Two wire dialects. The original proxy served /responses; the 2026-09 proxy serves only
# /chat/completions and answers /responses with HTTP 500 "not implemented". A wrong guess here
# fails EVERY row identically, and the file-level resume check then treats a file of
# `verdict: "error"` as "already judged" — the B2 n=981 trap. So the dialect is probed once,
# up front, and printed; it is never inferred per-row mid-run.
_API_STYLE = str(os.environ.get("DMR_JUDGE_API") or "auto").strip().lower()

# 8000, not 2000: a reasoning judge spends the budget on hidden reasoning tokens and then
# returns EMPTY content, which lands as `unparsed:` and burns all four retries. Applied to BOTH
# dialects and pinned in one place -- for a reasoning model the budget can change the answer, so
# every file in a comparison must be graded under the same value or the difference between two
# files stops being attributable to the thing under test.
_MAX_OUT = int(os.environ.get("DMR_JUDGE_MAX_OUTPUT") or 8000)


def _endpoint(style: str) -> str:
    return f"{_BASE_URL}/responses" if style == "responses" else f"{_BASE_URL}/chat/completions"


# urllib advertises `Python-urllib/3.x`, which a Cloudflare-fronted proxy rejects with HTTP 403
# and a body of `error code: 1010` — before the request ever reaches the model. The 2026-09-04
# proxy does exactly this, and the symptom is indistinguishable from a dead endpoint: the dialect
# probe fails on BOTH styles and the run aborts, or (if the probe is skipped) every row lands as
# `verdict: "error"`. Same host answers instantly under any other UA. Overridable in case a
# future proxy filters differently.
_USER_AGENT = str(os.environ.get("DMR_JUDGE_USER_AGENT") or "curl/8.5.0")


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "User-Agent": _USER_AGENT}


# Reasoning effort for the GPT-5 series. THREE things about this are load-bearing:
#
#  1. Only the NESTED form works. `{"reasoning": {"effort": "high"}}` is honoured and echoed back;
#     `{"reasoning_effort": "high"}` at the top level returns HTTP 200, is silently dropped, and
#     the response echoes `effort: medium`. The flat spelling therefore looks like it worked while
#     changing nothing — verified against this proxy on 2026-09-05.
#  2. The provider default is `medium`. Every file graded before this parameter existed was graded
#     at `medium`, so `medium` is the value that keeps new verdicts poolable with the recall sweep
#     and the foreign-memory control. Do not change it for one arm of a comparison.
#  3. Effort changes the grader. It is provenance in exactly the way `judge_model` is, so it is
#     recorded in every summary and the repair guard refuses to splice across a mismatch.
#
# Set to the empty string to omit the field entirely (needed for non-reasoning models).
_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_REASONING_EFFORT = str(os.environ.get("DMR_JUDGE_REASONING_EFFORT", "medium")).strip().lower()
if _REASONING_EFFORT and _REASONING_EFFORT not in _EFFORTS:
    # Fail here, not per row: a rejected value 400s on every request, and a full file of
    # `verdict: "error"` is then skipped forever by the file-level resume check.
    raise SystemExit(
        f"DMR_JUDGE_REASONING_EFFORT={_REASONING_EFFORT!r} is not one of {_EFFORTS}")


def _effort_applies() -> bool:
    """Only the GPT series takes this, and only on /responses."""
    return bool(_REASONING_EFFORT) and JUDGE_MODEL.lower().startswith("gpt-")


def _build_body(style: str, user: str) -> bytes:
    if style == "responses":
        payload = {"model": JUDGE_MODEL, "instructions": INSTRUCTIONS, "input": user,
                   "max_output_tokens": _MAX_OUT}
        if _effort_applies():
            payload["reasoning"] = {"effort": _REASONING_EFFORT}
    else:
        payload = {"model": JUDGE_MODEL, "max_tokens": _MAX_OUT, "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": user}]}
    return json.dumps(payload).encode()


def _extract_text(style: str, j: dict[str, Any]) -> str:
    if style == "responses":
        txt = ""
        for item in j.get("output", []):
            for c in (item.get("content") or []):
                if c.get("type") == "output_text":
                    txt += c.get("text", "")
        return txt
    choices = j.get("choices") or [{}]
    return str((choices[0].get("message") or {}).get("content") or "")
INSTRUCTIONS = (
    "You are a strict grader for visual-document question answering. "
    "You are given a question, the reference (gold) answer, and a model's answer. "
    "Decide whether the model's answer is correct. Be lenient about wording, formatting, "
    "and extra explanation, but strict about the actual facts. If the question has multiple "
    "parts, ALL parts must be factually correct for 'correct'; if only some parts are correct, "
    "use 'partial'. Output STRICT JSON only, no prose, with keys: "
    '{"verdict": "correct"|"partial"|"incorrect", "reason": "<short>"}.'
)
_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"(correct|partial|incorrect)"', re.I)
_REASON_RE = re.compile(r'"reason"\s*:\s*"(.*?)"\s*[,}]', re.S)


def _parse_verdict(txt: str) -> dict[str, Any] | None:
    """Recover the verdict from the model's JSON, tolerating invalid escapes.

    The grader is asked for strict JSON but is grading answers that quote LaTeX, so it emits
    things like `the function $g^*(\\cdot)$` inside the `reason` string. `\\c` is not a legal
    JSON escape, `json.loads` raises, and the row is retried four times against a model that
    deterministically produces the same text — then written off as `verdict: "error"`. Those
    rows are not missing at random: they concentrate in maths-heavy documents (arXiv), so
    silently dropping them biases the very families the fidelity report says differ most.
    """
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        for candidate in (m.group(0), re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", m.group(0))):
            try:
                parsed = json.loads(candidate)
            except Exception:  # noqa: BLE001
                continue
            verdict = str(parsed.get("verdict", "")).strip().lower()
            if verdict in _SCORE:
                return {"verdict": verdict, "score": _SCORE[verdict],
                        "reason": str(parsed.get("reason", ""))[:300]}
    # Last resort: the verdict token itself is plain ASCII even when the reason is not.
    v = _VERDICT_RE.search(txt or "")
    if v:
        verdict = v.group(1).lower()
        r = _REASON_RE.search(txt or "")
        return {"verdict": verdict, "score": _SCORE[verdict],
                "reason": (r.group(1) if r else "")[:300]}
    return None


def _judge_one(key: str, question: str, answers: list[str] | None, pred: str,
               timeout: int, retries: int) -> dict[str, Any]:
    ref = " | ".join(str(a) for a in (answers or []) if str(a).strip()) or "(none)"
    user = (f"Question:\n{question}\n\nReference answer:\n{ref}\n\n"
            f"Model answer:\n{pred}\n\nGrade now. Output strict JSON only.")
    body = _build_body(_API_STYLE, user)
    url = _endpoint(_API_STYLE)
    last = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=_headers(key))
            r = urllib.request.urlopen(req, timeout=timeout)
            j = json.loads(r.read())
            txt = _extract_text(_API_STYLE, j)
            got = _parse_verdict(txt)
            if got is not None:
                return got
            last = f"unparsed: {txt[:120]}"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            import time; time.sleep(min(5 * attempt, 20))
            continue
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:80]}"
            import time; time.sleep(min(2 * attempt, 8))
            continue
    return {"verdict": "error", "score": None, "reason": last}


def _detect_style(key: str, timeout: int, attempts: int = 4) -> str:
    """Probe both dialects with a trivial grade and keep the first that answers.

    Retries, because the probe is a single request standing in front of thousands: one transient
    TLS handshake timeout here aborts a whole file. That is not hypothetical -- it cost the
    vmqar_drop1 memory arm on the 2026-09-03 run while all nine other files judged with
    n_error = 0, and a partly-judged sweep cannot be analysed at all, since the levels are
    compared against each other.
    """
    import time
    errs: list[str] = []
    for attempt in range(1, attempts + 1):
        for style in ("responses", "chat"):
            try:
                req = urllib.request.Request(
                    _endpoint(style),
                    data=_build_body(style, 'Question:\nx\n\nReference answer:\nx\n\n'
                                            'Model answer:\nx\n\nGrade now. '
                                            'Output strict JSON only.'),
                    headers=_headers(key))
                j = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
                if _extract_text(style, j).strip():
                    return style
                errs.append(f"try{attempt} {style}: empty response")
            except Exception as e:  # noqa: BLE001
                errs.append(f"try{attempt} {style}: {type(e).__name__} {str(e)[:60]}")
        if attempt < attempts:
            time.sleep(min(5 * attempt, 20))
    raise SystemExit("judge endpoint unusable — " + "; ".join(errs[-6:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise SystemExit("缺少 OPENAI_API_KEY")

    global _API_STYLE
    if _API_STYLE not in ("responses", "chat"):
        _API_STYLE = _detect_style(key, timeout=min(args.timeout, 60))
    eff = _REASONING_EFFORT if _effort_applies() else "(not sent)"
    print(f"[judge] model={JUDGE_MODEL} api={_API_STYLE} effort={eff} "
          f"url={_endpoint(_API_STYLE)}", flush=True)

    records = [json.loads(l) for l in open(args.predictions, encoding="utf-8") if l.strip()]

    def work(rec: dict[str, Any]) -> dict[str, Any]:
        jr = _judge_one(key, rec.get("question") or "", rec.get("answers"),
                        str(rec.get("pred") or ""), args.timeout, args.retries)
        return {**rec, "judge": jr}

    out: list[dict[str, Any]] = [None] * len(records)  # type: ignore
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(work, r): i for i, r in enumerate(records)}
        for fut in futs:
            pass
        for fut, i in list(futs.items()):
            out[i] = fut.result()
            done += 1
            if done % 10 == 0:
                print(f"  judged {done}/{len(records)}", flush=True)

    scores = [r["judge"]["score"] for r in out if r["judge"]["score"] is not None]
    n_err = sum(1 for r in out if r["judge"]["score"] is None)
    vc = {}
    for r in out:
        vc[r["judge"]["verdict"]] = vc.get(r["judge"]["verdict"], 0) + 1
    summary = {
        "predictions": args.predictions, "n": len(records),
        # Provenance: verdicts from different judges must never be pooled, and this campaign has
        # now used three (gpt-5.5, gpt-5.4, and the Gemini line). Record which one wrote the file.
        # Reasoning effort is part of the grader's identity, not a performance knob: the same
        # model at a different effort is a different judge. Recorded so a later comparison can
        # check it, and so the repair guard can refuse a cross-effort splice.
        "judge_model": JUDGE_MODEL, "judge_api": _API_STYLE,
        "judge_reasoning_effort": (_REASONING_EFFORT if _effort_applies() else None),
        "n_judged": len(scores), "n_error": n_err,
        "judge_score_mean": (sum(scores) / len(scores)) if scores else None,
        "verdict_counts": vc,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.output + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
