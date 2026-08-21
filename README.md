# DevDex Eval — a benchmark for developer search

Three things developers search for: the repo behind a described capability, the docs page that
answers a how-to, and the issue or PR where a bug was fixed. Scored behind a real agent,
deterministically — no model judge.

| Track | items | Question | Gold |
|---|---|---|---|
| repo | 393 | "find the repo that does X" | one GitHub repository |
| issue/PR | 400 | "where was this bug fixed" | the issue **or** the PR |
| docs | 386 | "how do I do X in library Y" | one documentation page |

**recall@10** — fraction of the item's golds cited. **MRR@10** — 1 / rank of the first gold
cited. Both computed on the agent's first 10 citations.

This repo ships a 594-item public sample (~50% of each track); the rest is held back to keep the
memorisation gate working.

## Results

Claude Opus (`claude-opus-4-8`), one search tool per arm. `[ctl]` rows are controls, not products;
`No tools` is the memory floor every row is read against.

**Combined** — equal-weight mean of the three track means, 95% CI from a stratified bootstrap:

| system | recall@10 | 95% CI | MRR@10 | 95% CI |
|---|---|---|---|---|
| Firecrawl Developer Index | **0.631** | [0.605, 0.657] | **0.596** | [0.571, 0.622] |
| Parallel | 0.577 | [0.553, 0.602] | 0.561 | [0.538, 0.585] |
| Mintlify | 0.546 | [0.520, 0.572] | 0.538 | [0.512, 0.565] |
| Exa | 0.537 | [0.511, 0.563] | 0.538 | [0.512, 0.564] |
| Native web search `[ctl]` | 0.454 | [0.430, 0.477] | 0.458 | [0.435, 0.483] |
| Context7 `†` | 0.168 | [0.151, 0.185] | 0.145 | [0.130, 0.161] |
| No tools `[ctl]` | 0.013 | [0.007, 0.020] | 0.009 | [0.005, 0.015] |

**Per track:**

| system | repo recall | repo MRR | issue/PR recall | issue/PR MRR | docs recall | docs MRR |
|---|---|---|---|---|---|---|
| Firecrawl Developer Index | 0.761 | 0.755 | **0.660** | **0.633** | **0.472** | **0.401** |
| Parallel | **0.819** | **0.798** | 0.629 | 0.624 | 0.282 | 0.261 |
| Mintlify | 0.743 | 0.733 | 0.560 | 0.581 | 0.334 | 0.301 |
| Exa | 0.733 | 0.723 | 0.585 | 0.609 | 0.293 | 0.282 |
| Context7 | 0.008 `†` | 0.008 `†` | 0.029 `†` | 0.031 `†` | 0.466 | 0.395 |
| Native web search `[ctl]` | 0.807 | 0.794 | 0.275 | 0.329 | 0.280 | 0.251 |
| GitHub CLI `[ctl]` | 0.234 | 0.232 | 0.469 | 0.560 | — | — |
| No tools `[ctl]` | 0.005 | 0.001 | 0.000 | 0.000 | 0.034 | 0.027 |

`—` = not measured, never zero. GitHub CLI has no docs cell, so it has no combined score.

`†` Context7 indexes library documentation; repo and issue/PR are outside that domain and both
cells fail the 10% dead-run bar (71% and 52% dead).
Read its combined as coverage — one of three tracks answered — not as retrieval quality. On docs
it is second overall.

`Firecrawl Developer Index` is `firecrawl_developer_search`, a curated index of issues, PRs,
READMEs and docs — one search tool, 10 results per call, the same depth every other engine here
gets. Firecrawl's general web search is a different surface and is not listed; run it with
`--arm fc-web`, and see `benchmark/results/results.json` for its scores.

On any single track the top two or three arms are statistically indistinguishable under a paired
bootstrap. The leader separates only on the combined score.

## Install

```bash
uv sync --extra dev        # creates .venv and installs everything, including pytest
cp .env.example .env       # ANTHROPIC_API_KEY required; add only the vendor keys you need
uv run pytest devdex/tests -q
```

With pip instead: `pip install -e ".[dev]"`. Both install the `devdex` and `devdex-report`
commands; examples below use `uv run`, which pip users can drop.

Runs cost real money — roughly $0.28/item, about $165 for one arm across all three tracks.
Smoke-test with `--limit` first.

## Run an arm

```bash
uv run devdex --track repo --arm fc-mcp --driver opus --n-runs 1 --limit 10
uv run devdex-report --pass p1
```

- `--arm` — `fc-mcp`, `fc-web`, `exa-mcp`, `parallel-mcp`, `mintlify`, `context7`,
  `gh-hybrid`, `websearch`, `no-tool`. Needs that vendor's key from `.env.example`.
- `--track` — `repo`, `fix` (the issue/PR track) or `docs`.
- `--limit` — drop it for a full track.

## Run your own engine

```bash
uv run python benchmark/run_benchmark.py --name yourco \
    --mcp-url https://mcp.yourco.com/mcp --search-tool your_search --limit 10 --track repo
```

Add `--fetch-tool your_fetch` if you ship a reader, `--auth "Bearer $KEY"` if your server needs
one. Drop `--limit` and use `--track all` for a full run. To re-score records already on disk, pass
`--report-only` with the same `--name` you ran under — it locates your run directories — plus
`--out`, without which nothing is written:

```bash
uv run python benchmark/run_benchmark.py --name yourco --report-only --out results.json
```

Your server is mounted exactly like every arm above: same agent, same tool gate, same scorer.

**What your server needs**

| | |
|---|---|
| transport | streamable **HTTP** MCP (stdio is not supported by this route) |
| query parameter | one of `query`, `q`, `search_queries`, `objective`, `search`, `prompt` |
| result limit *(optional)* | `k`, `limit`, `numResults`, `num_results`, `max_results`, `maxResults`, `topK`, `count` |
| domain filter *(optional)* | `includeDomains`, `include_domains`, `domains`, `site`, `allowed_domains` |
| results | JSON, markdown or prose; each hit needs a URL and some text |

Only parameters your schema declares are ever set, so you are never handed a capability you don't
ship — nor denied one you do. Arms without a limit are reported without depth-sensitive columns;
arms without a domain filter get the scope as a query hint instead.

**The query parameter name matters.** If it isn't one of the names above the run still completes
but degrades quietly: GitHub scoping is skipped on repo and issue/PR, and the logged query becomes
the raw argument object. Run `--limit 10` first and check the `search_log` entries look right.

See [CONTRIBUTING.md](CONTRIBUTING.md) to submit a score for the table.

## Method

**repo** — capability descriptions of author-declared repositories, stratified by star count. A
name-leak gate rejects any query containing the repo's name.

**docs** — a real documentation passage with its most distinctive words banned from the query;
queries overlapping the passage too heavily are rejected. Gold accepts the GitHub source file
**or** the rendered page, since engines return different forms of it.

**issue/PR** — real issue–PR pairs where a PR closed an issue; citing either counts.

**Memorisation gate** — candidates were run with search disabled and scored the same way; anything
the model answered anyway was dropped from the set. The floors it leaves are the `no tools`
control's own scores (repo 0.005, issue/PR 0.000, docs 0.034), so the gate is re-checkable from any
run.

**Scoring** — deterministic reference and canonical-URL matching against fixed golds. A failed run
counts as a miss rather than an exclusion. No model judge decides any published number.

## Dataset

| | full (held back) | public sample (shipped) |
|---|---|---|
| repo | 393 | 198 |
| issue/PR | 400 | 195 |
| docs | 386 | 201 |

`benchmark/` the public runner · `devdex/gt/` the public sample · `devdex/harness/` the agent loop and tool gate ·
`devdex/scorer/` produced every number above · `devdex/tests/` 132 tests

## License

MIT
