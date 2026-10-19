# Sentinel: Build Plan

> Conformal guardrails for generative models and AI agents.
> A service that wraps any LLM endpoint and returns a statistically valid confidence guarantee on each output, abstaining or escalating when the guarantee cannot be met.

**Source paper:** Conf-Gen: Conformal Uncertainty Quantification for Generative Models (arXiv:2605.28920)
**Owner:** Kaleb
**Status:** Not started

---

## 1. North Star

Sentinel turns "the model is probably right" into "this output satisfies a chosen risk level, certified." It is a middleware layer, not a model. You pick a risk level (for example, error rate among answered questions stays under 10 percent), and Sentinel uses conformal risk control to either return an answer that meets that bar or abstain and escalate.

The reason this works as a resume project for infra, platform, and DevOps roles: it is a real deployed service with observability, caching, a circuit breaker, and a statistical guarantee that reads like an SLO for machine learning. The reason it survives an ML interview: the core is a genuine, modern uncertainty-quantification technique, not a softmax threshold.

### Definition of done

You are done when all of the following are true:

1. `docker compose up` brings the whole system online with one command (API, Redis, Prometheus, Grafana).
2. A `/v1/guarded-generate` endpoint returns an answer, an abstention, or an escalation, plus the guarantee metadata.
3. You can show empirically that realized risk stays at or below the target level on a held-out replay stream.
4. A Grafana dashboard shows live coverage versus target, abstention rate, and latency percentiles.
5. A short writeup with real measured numbers exists, and your resume bullets are filled in with those numbers.

---

## 2. The technical core (read this twice)

This is the part that has to be correct before anything else matters. Build it offline first, prove it works, then wrap a service around it.

### 2.1 The concrete task (MVP)

Selective question answering with abstention. The model answers a factual question, or it abstains. Sentinel guarantees that among the questions it chooses to answer, the error rate stays below your target `alpha`.

This is the cleanest instantiation of the paper's idea, it has readily available datasets, and the guarantee is easy to explain in an interview.

### 2.2 Nonconformity score

The score measures how "unusual" or "risky" an output is. Lower score means safer. You will implement two:

- **Baseline: token-level negative log-likelihood.** Average negative log-prob of the generated answer. Cheap, weak.
- **Headline: semantic entropy.** Sample k generations for the same question, cluster them by semantic equivalence (use a natural-language-inference model or sentence embeddings to decide when two answers mean the same thing), then compute the entropy over the semantic clusters. High semantic entropy means the model is internally inconsistent, which correlates with being wrong. This is the modern, impressive choice and it is your differentiator.

> Reference for semantic entropy: "Detecting Hallucinations in Large Language Models Using Semantic Entropy" (Farquhar, Kossen, Kuhn, Gal, Nature 2024) and the earlier "Semantic Uncertainty" (Kuhn, Gal, Farquhar, ICLR 2023). Look them up rather than trusting a remembered arXiv ID.

### 2.3 The conformal risk control procedure

Plain language version of what you are implementing:

1. Hold out a labeled **calibration set** of questions where you know the correct answer.
2. For each calibration example, run the model, compute the nonconformity score, and record whether the answer was correct.
3. Define a loss, for example `loss = 1 if (answered and wrong) else 0`.
4. Sweep a threshold `lambda` on the score. For a given `lambda`, you answer when score is below `lambda` and abstain otherwise. Compute the empirical risk at that `lambda`.
5. Conformal risk control picks the threshold using a finite-sample bound so that the expected risk on future data is at most `alpha`. The bound adds a small correction term that depends on calibration set size, which is what makes the guarantee distribution-free and valid in finite samples rather than just asymptotically.
6. At serving time you use the chosen `lambda`. Answer if score below it, otherwise abstain and escalate.

> References for the method: "Conformal Risk Control" (Angelopoulos, Bates, Fisch, Lei, Schuster) and the very readable "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" (Angelopoulos and Bates). Read the gentle intro first.

### 2.4 The honest engineering caveat (mention this in interviews, it scores points)

In real production you do not have ground-truth labels at inference time, so you cannot measure realized risk live on every request. Sentinel handles this with a **shadow evaluation stream**: a separate labeled stream (or a delayed-feedback simulation) that lets you estimate realized risk and detect calibration drift. Knowing this limitation and engineering around it is exactly the kind of maturity that separates a strong project from a toy.

---

## 3. Architecture

```
                         +-------------------+
   request --------->    |   FastAPI gateway |
                         |  (circuit breaker)|
                         +---------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                    |
        +-----v-----+        +-----v------+       +-----v------+
        |  model    |        |  scoring   |       | conformal  |
        |  adapter  |------->|  (logprob, |------>| engine     |
        | (Ollama)  |        |  sem. ent.)|       | (CRC)      |
        +-----------+        +------------+       +-----+------+
                                                        |
                          answer / abstain / escalate <-+
                                   |
              +--------------------+--------------------+
              |                    |                    |
        +-----v-----+        +-----v------+       +-----v------+
        |  Redis    |        | Prometheus |       | escalation |
        | (cached   |        |  /metrics  |       |   queue    |
        | quantiles)|        +-----+------+       +------------+
        +-----------+              |
                            +------v------+
                            |   Grafana   |
                            | dashboards  |
                            +-------------+
```

---

## 4. Repo structure

```
sentinel/
├── README.md
├── pyproject.toml              # use uv for dependency management
├── .env.example
├── Makefile                    # make dev, make test, make up, make seed
├── docker-compose.yml          # api + redis + prometheus + grafana
├── Dockerfile
├── docs/
│   ├── BUILD_PLAN.md           # this file
│   ├── architecture.md
│   └── conformal_theory.md     # your own notes as you learn it
├── sentinel/
│   ├── __init__.py
│   ├── config.py               # pydantic-settings, reads .env
│   ├── models/                 # inference adapters
│   │   ├── base.py             # abstract: generate(), generate_with_logprobs()
│   │   ├── ollama_adapter.py
│   │   └── api_adapter.py      # optional cloud fallback
│   ├── scoring/
│   │   ├── base.py             # abstract Score interface
│   │   ├── logprob.py
│   │   └── semantic_entropy.py
│   ├── conformal/              # the core, built and tested first
│   │   ├── split_conformal.py
│   │   ├── crc.py              # conformal risk control
│   │   └── calibration.py      # fit/load/store thresholds
│   ├── gateway/
│   │   ├── app.py              # FastAPI app factory
│   │   ├── routes.py           # /v1/guarded-generate, /healthz
│   │   ├── schemas.py          # pydantic request/response models
│   │   └── middleware.py       # circuit breaker
│   ├── cache/
│   │   └── redis_client.py
│   ├── monitoring/
│   │   ├── metrics.py          # prometheus_client counters/histograms/gauges
│   │   └── shadow_eval.py      # realized-risk estimation on labeled stream
│   └── escalation/
│       └── queue.py            # simple Redis-backed queue to start
├── eval/
│   ├── datasets/               # downloaded QA data, gitignored
│   ├── run_calibration.py      # build calibration set, fit threshold
│   ├── run_coverage_test.py    # prove realized risk <= alpha end to end
│   └── benchmark_latency.py
├── notebooks/
│   └── 01_conformal_sanity_check.ipynb
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboards/sentinel.json
├── scripts/
│   ├── seed_calibration.py
│   └── replay_stream.py        # feeds the shadow eval / demo
└── tests/
    ├── test_crc.py             # the most important test file
    ├── test_scoring.py
    └── test_gateway.py
```

---

## 5. Tech stack (2026)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.12 | |
| Dependency mgmt | uv | fast, modern, looks current |
| Model serving (dev) | Ollama | run Llama 3.1 8B or Qwen2.5 7B locally |
| Model serving (scale) | vLLM on Modal | serverless GPU, optional stretch |
| API | FastAPI + Uvicorn | |
| Validation | Pydantic v2 | typed request/response contracts |
| Cache / queue | Redis | cached quantiles, escalation queue |
| Metrics | prometheus-client | `/metrics` endpoint |
| Dashboards | Grafana | the visual centerpiece |
| Scoring math | NumPy, scikit-learn | CKA-free here, just entropy and quantiles |
| NLI / embeddings | a small NLI model or sentence-transformers | for semantic clustering |
| Containerization | Docker, docker-compose | one-command bring-up |
| Tests | pytest | |
| CI (stretch) | GitHub Actions | lint + test on push |

---

## 6. Milestones

| ID | Milestone | Exit criteria |
|---|---|---|
| M0 | Foundations and spike | Repo skeleton, env, model adapter returns text and logprobs |
| M1 | Conformal core, offline | CRC implemented, coverage validated on synthetic then real calibration data |
| M2 | Scoring layer | logprob and semantic entropy implemented and compared |
| M3 | Gateway | `/v1/guarded-generate` works, Redis caching, circuit breaker, dockerized |
| M4 | Monitoring | Prometheus metrics, Grafana dashboard, shadow eval, escalation queue |
| M5 | Eval and benchmarks | end-to-end coverage proof, ablations, latency report |
| M6 | Polish and ship | one-command up, README, demo, writeup, resume bullets filled |

Build order rule: M1 before everything. If the conformal core is wrong, a beautiful dashboard is just a beautiful lie.

---

## 7. Week-by-week scope

Calibrated to roughly 10 to 15 hours per week so it fits around classes. If you switch to full-time over summer you can compress two of these weeks into one. Each week has one concrete deliverable so a bad week never leaves you with nothing.

### Week 1 — Foundations (M0)
- Set up repo, uv, pre-commit, basic `Makefile`.
- Install Ollama, pull a small model, confirm you can generate text and pull token logprobs.
- Implement `models/base.py` and `ollama_adapter.py`.
- Read the gentle intro to conformal prediction. Start `docs/conformal_theory.md` in your own words.
- **Deliverable:** a script that takes a question and prints an answer plus its average negative log-likelihood.

### Week 2 — Conformal core on synthetic data (M1)
- Implement `split_conformal.py`. Validate it on a synthetic setup where you control the data and know the right answer, confirming empirical coverage lands at `1 - alpha`.
- Write `tests/test_crc.py` first, then make it pass.
- **Deliverable:** a notebook plot showing realized coverage hitting the target across several alpha values.

### Week 3 — CRC on real calibration data (M1)
- Download a QA dataset (TriviaQA or Natural Questions open variant).
- Build the calibration pipeline: run the model over questions, grade correctness (exact match or a cheap LLM judge), record scores.
- Implement `crc.py` to choose `lambda` controlling error-among-answered at `alpha`.
- **Deliverable:** `run_calibration.py` outputs a fitted threshold and a report showing offline risk at or below alpha.

### Week 4 — Scoring layer (M2)
- Implement `semantic_entropy.py`: sample k generations, cluster by semantic equivalence, compute entropy.
- Compare semantic entropy versus logprob on the risk-coverage tradeoff (lower abstention at the same risk is better).
- **Deliverable:** a plot comparing both scores, with a one-paragraph conclusion on which wins and why.

### Week 5 — Gateway, part 1 (M3)
- Build the FastAPI app: `/v1/guarded-generate` runs model, scoring, conformal decision, returns `{decision, answer, score, threshold, alpha}`.
- Pydantic schemas for request and response.
- **Deliverable:** working local endpoint you can curl.

### Week 6 — Gateway, part 2 (M3)
- Add Redis: cache the fitted quantile or threshold, optional response cache.
- Implement the circuit breaker middleware: trip when abstention rate over a rolling window exceeds a configured ceiling, returning a clear degraded-mode response.
- Write `Dockerfile` and a first `docker-compose.yml` with api plus redis.
- **Deliverable:** `docker compose up` brings up api and redis, endpoint still works.

### Week 7 — Monitoring, part 1 (M4)
- Add `prometheus-client` metrics: request counter, latency histogram, abstention-rate gauge, realized-risk gauge.
- Implement `shadow_eval.py` plus `replay_stream.py` so a labeled stream produces realized-risk estimates.
- Add Prometheus to compose, scraping `/metrics`.
- **Deliverable:** Prometheus is collecting Sentinel metrics.

### Week 8 — Monitoring, part 2 (M4)
- Add Grafana to compose, provision `sentinel.json` dashboard: coverage versus target line, abstention rate, latency p50 p95 p99, escalation queue depth.
- Finish the escalation queue.
- **Deliverable:** a dashboard worth screenshotting for your README and resume.

### Week 9 — Eval and benchmarks (M5)
- `run_coverage_test.py`: end-to-end proof that realized risk stays at or below alpha under replay.
- Ablations: alpha sweep, semantic entropy versus logprob, calibration set size sensitivity.
- `benchmark_latency.py`: measure added latency from scoring and the conformal decision.
- **Deliverable:** an eval report with tables and plots and real numbers.

### Week 10 — Polish and ship (M6)
- One-command bring-up verified on a clean checkout.
- README with architecture diagram, quickstart, and a demo gif.
- Short technical writeup or blog post on the conformal core and the shadow-eval design.
- Fill in resume bullets with measured numbers.
- **Deliverable:** a repo you would be happy to send a recruiter, and a writeup link.

---

## 8. If time gets tight (scope cuts in priority order)

Cut from the bottom up. The goal is always a working, honest, deployed core.

1. Drop vLLM and Modal. Local Ollama is enough to demonstrate the engineering.
2. Drop semantic entropy, ship with the logprob score. Note semantic entropy as future work.
3. Replace Grafana with a single FastAPI-served HTML metrics page if Grafana provisioning eats too much time.
4. Keep M1 through M4 sacred. A correct, deployed, monitored guardrail with logprob scoring is still a strong project. A half-built version with every bell is not.

---

## 9. Metrics to capture for your resume

Log these as you go so the bullets write themselves:

- Realized risk versus target alpha on the replay stream (proves the guarantee holds).
- Abstention rate at the chosen alpha (lower is better at fixed risk).
- Reduction in abstention from semantic entropy versus logprob, in percentage points.
- Added latency from scoring plus conformal decision (target the calibration lookup under a few tens of milliseconds with Redis caching).
- p95 and p99 end-to-end latency.
- Throughput under load if you run a Locust test.

### Draft resume bullets (fill the brackets with your numbers)

- Built a distribution-free guardrail service implementing conformal risk control, certifying selective QA outputs at a configurable risk level and holding realized error at or below [alpha] percent on a held-out replay stream.
- Engineered a FastAPI and Redis gateway with Prometheus and Grafana observability, surfacing live coverage versus target and auto-tripping a circuit breaker when abstention exceeded [X] percent over a rolling window.
- Reduced abstention by [X] percentage points at fixed risk by replacing a log-probability score with a sampling-based semantic-entropy nonconformity score, validated through ablations.

---

## 10. How to talk about it in interviews

- One sentence: "Sentinel is a middleware layer that gives any LLM a statistically valid guarantee on its outputs, abstaining when it cannot meet the bar, with the guarantee monitored live like an SLO."
- For infra and platform interviewers, lead with the service: gateway, caching, circuit breaker, observability, one-command deploy.
- For ML interviewers, lead with the core: conformal risk control, semantic entropy as a nonconformity score, and the shadow-eval design that handles the no-labels-in-production problem.
- Always volunteer the honest limitation (no live labels, hence shadow eval). It signals you understand the difference between a demo and a system.

---

## 11. Learning resources (look them up, do not trust remembered IDs)

- "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" by Angelopoulos and Bates. Start here.
- "Conformal Risk Control" by Angelopoulos, Bates, Fisch, Lei, Schuster.
- "Semantic Uncertainty" by Kuhn, Gal, Farquhar (ICLR 2023).
- "Detecting Hallucinations in Large Language Models Using Semantic Entropy" by Farquhar, Kossen, Kuhn, Gal (Nature 2024).
- The Conf-Gen paper itself (arXiv:2605.28920) for the generative adaptation framing.
- FastAPI docs, prometheus-client docs, Grafana provisioning docs.