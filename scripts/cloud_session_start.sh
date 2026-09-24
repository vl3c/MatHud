#!/bin/bash
# SessionStart hook for Claude Code cloud sessions (CLAUDE_CODE_REMOTE=true); does nothing elsewhere.
#
# 1. Link ./venv to /opt/mathud-venv, where the cloud environment's setup script installs the
#    packages. The environment cache keeps /opt across sessions, but each session gets a fresh
#    clone, and the CLI starts the app with ./venv/bin/python.
# 2. Trust the sandbox proxy's CA in Chrome's certificate store (Chrome ignores the system store),
#    so headless Chrome can load the app's CDN scripts during client tests. The proxy starts after
#    the setup script and its certificate can change between sessions, so this runs every session.
#    Needs certutil, which the setup script installs with libnss3-tools.

[ "${CLAUDE_CODE_REMOTE:-}" = true ] || exit 0

project_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}"
if [ -x /opt/mathud-venv/bin/python ] && [ ! -e "$project_dir/venv" ]; then
    ln -s /opt/mathud-venv "$project_dir/venv"
fi

proxy_ca=/root/.ccr/agent-proxy-ca.crt
nss_db="$HOME/.pki/nssdb"
if [ -r "$proxy_ca" ] && command -v certutil >/dev/null 2>&1; then
    mkdir -p "$nss_db"
    [ -f "$nss_db/cert9.db" ] || certutil -d "sql:$nss_db" -N --empty-password
    certutil -d "sql:$nss_db" -D -n mathud-agent-proxy >/dev/null 2>&1
    certutil -d "sql:$nss_db" -A -t "C,," -n mathud-agent-proxy -i "$proxy_ca" ||
        echo "cloud_session_start: could not add the proxy CA to $nss_db" >&2
else
    echo "cloud_session_start: skipped the proxy CA (need $proxy_ca and certutil)" >&2
fi
exit 0
