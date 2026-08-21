"""preflight.py — connect to every arm's MCP server, list its tools, freeze the contract.

RUN THIS BEFORE ANY RUN. It writes `tools_manifest.json`: for each arm, the tool names the
vendor actually exposes today and their input schemas.

Why it is not optional:

  * Tool names drift. Exa's own docs and release notes disagree about which tools are on by
    default. If `arms.py` names a tool the server no longer exposes, every call is denied and
    the arm scores 0.000 — a plumbing bug that reads exactly like a product result.
  * Parameter normalisation needs the schemas. The harness injects k and a github filter ONLY
    into keys a tool actually declares (provenance.normalize_input). No schema, no injection,
    and the arms stop being comparable.
  * The manifest IS the reproducibility contract for this harness, the way a python adapter is for
    an adapter harness: it records exactly which vendor surface produced a given set of numbers.
    Commit it alongside the results.

Usage:
  python3 preflight.py               # all MCP arms
  python3 preflight.py exa-mcp       # one arm
"""
import json, os, subprocess, sys, urllib.request

# Same dual import as control.py: this file is both `devdex.harness.preflight` and a documented
# standalone script (`python3 preflight.py exa-mcp`), so it must resolve either way.
#
# The k / domain key lists ARE the fairness rule. They were duplicated here as literals, so the
# preflight report could disagree with what the controller actually injects. One source.
try:
    from devdex.harness import arms as A
    from devdex.harness.provenance import K_KEYS, DOMAIN_KEYS
except ImportError:                      # this directory is on sys.path (script mode)
    import arms as A
    from provenance import K_KEYS, DOMAIN_KEYS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tools_manifest.json")
PROTO = "2024-11-05"
# Both remote MCPs sit behind Cloudflare, which 403s a request with urllib's default
# User-Agent (error 1010, "blocked based on your browser's signature"). That failure looks
# exactly like a bad API key, so it is worth naming: send a real UA or preflight lies to you.
UA = "devdex-mcp-preflight/1.0"
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": PROTO, "capabilities": {},
                   "clientInfo": {"name": "devdex-mcp-preflight", "version": "1"}}}
LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
READY = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}


def _sse_or_json(body, ctype):
    """Streamable-HTTP servers may answer a POST with SSE. Take the last data: frame."""
    if "event-stream" in (ctype or ""):
        frames = [l[5:].strip() for l in body.splitlines() if l.startswith("data:")]
        return json.loads(frames[-1]) if frames else {}
    return json.loads(body or "{}")


def list_http(url, headers):
    hdr = {"Content-Type": "application/json",
           "Accept": "application/json, text/event-stream",
           "User-Agent": UA, "MCP-Protocol-Version": PROTO, **(headers or {})}
    req = urllib.request.Request(url, data=json.dumps(INIT).encode(), headers=hdr)
    with urllib.request.urlopen(req, timeout=60) as r:
        init = _sse_or_json(r.read().decode(errors="replace"), r.headers.get("content-type"))
        sid = r.headers.get("mcp-session-id") or r.headers.get("Mcp-Session-Id")
    if sid:
        hdr["Mcp-Session-Id"] = sid
    try:                                        # best-effort; some servers do not require it
        urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(READY).encode(), headers=hdr), timeout=30)
    except Exception:
        pass
    req = urllib.request.Request(url, data=json.dumps(LIST).encode(), headers=hdr)
    with urllib.request.urlopen(req, timeout=60) as r:
        out = _sse_or_json(r.read().decode(errors="replace"), r.headers.get("content-type"))
    return (out.get("result") or {}).get("tools", []), (init.get("result") or {}).get("serverInfo", {})


def list_stdio(cmd, args, env):
    p = subprocess.Popen([cmd, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, env={**os.environ, **(env or {})})
    try:
        for msg in (INIT, READY, LIST):
            p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        tools, info = [], {}
        for _ in range(400):                    # npx servers print banners before the handshake
            line = p.stdout.readline()
            if not line:
                break
            try:
                m = json.loads(line)
            except Exception:
                continue
            if m.get("id") == 1:
                info = (m.get("result") or {}).get("serverInfo", {})
            if m.get("id") == 2:
                tools = (m.get("result") or {}).get("tools", [])
                break
        return tools, info
    finally:
        p.kill()


def main():
    want = sys.argv[1:] or [n for n, a in A.ARMS.items() if a.kind in ("mcp", "local")]
    # DO NOT carry the old manifest forward. Merging meant a tool the vendor had RENAMED kept its
    # stale entry, `full in manifest` still said OK, and the hard gate passed — the failure then
    # surfaced at runtime as every call denied, i.e. the 0.000-that-looks-like-a-product-result
    # this file exists to prevent. Entries for servers we did not just contact are dropped.
    manifest = {}
    if os.path.exists(OUT):
        prev = json.load(open(OUT))
        touched = {ns for n in want if A.ARMS[n].kind == "mcp" for ns in A.ARMS[n].servers}
        manifest = {k: v for k, v in prev.items() if (v.get("server") or k.split("__")[1]) not in touched}
    ok = True
    for name in want:
        arm = A.ARMS[name]
        if arm.kind == "local":
            # No server to interrogate — the control arm's tool is ours (control.gh_search), so
            # the thing to verify is that the binary it shells out to is present and authenticated.
            import shutil
            ok_gh = bool(shutil.which("gh")) and subprocess.run(
                ["gh", "auth", "status"], capture_output=True).returncode == 0
            print(f"  [{name}] local tool control.gh_search — "
                  f"gh CLI {'OK (authenticated)' if ok_gh else 'MISSING or NOT AUTHENTICATED'}")
            print(f"    {'OK  ' if ok_gh else 'MISS'} tracks={sorted(arm.tracks)} "
                  f"tool={arm.tools_for('repo')}")
            ok = ok and ok_gh
            continue
        if arm.kind != "mcp":
            continue
        for ns, cfg in arm.servers.items():
            try:
                if cfg.get("type") == "stdio":
                    tools, info = list_stdio(cfg["command"], cfg.get("args", []), cfg.get("env"))
                else:
                    tools, info = list_http(cfg["url"], cfg.get("headers"))
            except Exception as e:
                print(f"  [{name}/{ns}] CONNECT FAILED: {type(e).__name__}: {str(e)[:160]}")
                ok = False
                continue
            names = sorted(t.get("name", "") for t in tools)
            print(f"  [{name}/{ns}] {info.get('name', ns)} {info.get('version', '')} — "
                  f"{len(tools)} tools: {', '.join(names) or '(none)'}")
            for t in tools:
                # FULL description, not a 400-char cut. The manifest is the vendor-surface
                # contract; trimming it would silently hide a vendor's own description change
                # behind an unrelated-looking diff. (persisted records are
                # complete.)
                manifest[f"mcp__{ns}__{t['name']}"] = {
                    "server": ns, "tool": t["name"],
                    "description": t.get("description") or "",
                    "input_schema": t.get("inputSchema") or t.get("input_schema") or {}}
        # The assertion that matters: does the tool this arm will actually CALL still exist?
        # Resolution (not the raw candidate list) is what the harness runs, so preflight has to
        # report the same thing the harness will do — including which candidate went live.
        for track in A.TRACKS:
            if not arm.runs(track):
                print(f"    --   {track:5} not run by this arm (corpus does not cover it) — "
                      f"empty cell by design")
                continue
            live, fb = A.live_tools(arm, track, manifest)
            if fb:
                # A fallback is a RESULT, not a recovery: web_search_exa cannot express
                # includeDomains, so a substituted exa arm is scoped differently from the one the
                # method section describes. Loud here, stamped on every record, flagged by the
                # analyzer. Never let this scroll past unnoticed.
                print(f"    SUBST {track:5} {fb['from']} is GONE -> running {fb['to']}"
                      f"\n          ^ tool_fallback stamped on every record; the arm is readable, "
                      f"NOT quotable until arms.py is updated")
            for full in list(live) + sorted(arm.free_tools):
                mark = "OK  " if full in manifest else "MISS"
                if full not in manifest:
                    ok = False
                schema = (manifest.get(full) or {}).get("input_schema") or {}
                props = list((schema.get("properties") or {}).keys())
                kk = next((k for k in K_KEYS if k in props), None)
                dk = next((k for k in DOMAIN_KEYS if k in props), None)
                role = "FREE  " if full in arm.free_tools else ""
                print(f"    {mark} {track:5} {role}{full}"
                      f"   k-param={kk or '—'}  domain-param={dk or '—'}  params={props}")
    json.dump(manifest, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"\nwrote {OUT} ({len(manifest)} tools)")
    # An arm whose tool exposes no k and no domain filter is NOT silently equalised. It runs
    # with the vendor's own default depth (and query-level scoping if arms.py sets a hint),
    # and every call records what could not be normalised. Read that column before comparing.
    print("NOTE: a '—' above means the vendor cannot express that control. The harness records\n"
          "      it per call (search_log[].meta.normalized) instead of pretending it is equal.")
    if not ok:
        print("\nPREFLIGHT FAILED — fix arms.py tool names / credentials before running.")
        sys.exit(1)


if __name__ == "__main__":
    main()
