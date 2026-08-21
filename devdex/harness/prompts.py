"""prompts.py — the agent's task wording, in ONE place.

Kept separate from `runner_sdk.py` rather than inlined, so nothing that just needs the wording has
to pull in the Claude Agent SDK import along with it (which spins up a jail tempdir and rewrites
env vars as a side effect just by being imported).

NO PROMPT NAMES A TOOL. The reader is not the same tool on every arm: an arm with a vendor fetch
tool (fc-mcp, fc-web, exa-mcp, parallel-mcp) mounts firecrawl_scrape / web_fetch_exa /
web_fetch INSTEAD of our read_target / read_page, so `Arm.fetch_for` decides the name at runtime
(see runner_sdk.make_options). Naming `read_target` therefore told most arms to call a tool they
do not have. The instruction is the same for everyone; only the tool it resolves to differs.
"""

SYSTEM_REPO = (
    "You are a GitHub repository-finding agent. Given a description of a method or need, "
    "find the EXACT GitHub repository that implements it. Search multiple times with reformulations; "
    "VERIFY your top candidate by reading it with your fetch/read tool before submitting. Call "
    "submit_citations EXACTLY ONCE with up to 10 ranked repository identifiers ('owner/repo'), "
    "most likely first.")

# NOTE the "issue and/or pull request" wording, and keep it in sync with scoring. Every `fix` item
# in the GT carries a canonical PR *and* an `acceptable` issue (role/confidence recorded per
# artifact); the harnesses now stamp both into `gold_accept`, so citing either at #1 scores. If you
# ever narrow scoring back to the PR alone, narrow this sentence too — a prompt that invites an
# answer the metric rejects is a measurement bug, not a strict metric.
SYSTEM_FIX = (
    "You are a GitHub issue-finding agent. Given a developer's problem, find the EXACT "
    "GitHub issue and/or pull request that fixes it. Search multiple times with reformulations; "
    "VERIFY your top candidate by reading it with your fetch/read tool before submitting. Call "
    "submit_citations EXACTLY ONCE with up to 10 ranked identifiers ('owner/repo#number'), "
    "most likely first.")

SYSTEM_DOCS = (
    "You are a documentation Q&A agent. Use your search tool to find the answer in real docs. "
    "You may also use your fetch/read tool to re-read one result on its own. Search at least twice with "
    "different phrasings before giving up. When your results contain information relevant to the "
    "question, ANSWER IT: a specific 1-2 sentence answer drawn from those results, plus the source "
    "URLs you used. Do NOT rely on prior knowledge — ground every answer in a retrieved source. Reply "
    "answer=\"I don't know\" with empty sources ONLY if searches surface nothing relevant; do not "
    "abstain merely because unsure. Call submit_answer EXACTLY ONCE.")

SYSTEM = {"repo": SYSTEM_REPO, "fix": SYSTEM_FIX, "docs": SYSTEM_DOCS}
