# Contributing

## Submitting your engine's score

Your engine is driven by the same agent loop, tool gate, depth cap and scorer as every arm in the
README. There is one scorer; no submission gets a separate path.

An MCP server is required.

1. **Smoke-test first**, to confirm your server is reachable and its tool resolves:

   ```bash
   export ANTHROPIC_API_KEY=...
   uv run python benchmark/run_benchmark.py --name yourco --mcp-url https://mcp.yourco.com/mcp \
       --search-tool your_search --track repo --limit 20
   ```

2. **Run the full public sample** — drop `--limit`, use `--track all`.

3. **Open a PR** with:
   - the exact command you ran
   - the run output under `devdex/runs/<timestamp>-ext-yourco-*/`, or at minimum a `results.json`
     written by:

     ```bash
     uv run python benchmark/run_benchmark.py --name yourco --report-only --out results.json
     ```

     Pass the same `--name` you ran under — it is what locates your run directories — and `--out`,
     without which nothing is written to disk.
   - anything about your MCP surface that isn't evident from its schema
     
To submit a provider, open a PR against the repo with your results on the public half, and email rafael@sideguide.dev with valid API keys so we can rerun on the held-out half.
