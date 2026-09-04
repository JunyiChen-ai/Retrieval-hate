#!/bin/bash
# Layout check across machines (CLAUDE.md "文件落位"): home dir must hold only the allowed
# entries, every machine must be on the same commit with a clean tree.
# Usage: bash scripts/check_layout.sh            (checks local + uoa-lab1 + uoa-lab3)
ALLOW='^(data|miniconda3|Retrieval-hate|snap|venvs|Hate-follow-up|AgentDebugX|Auto-claude-code-research-in-sleep|miniconda\.sh|Miniconda3-latest-Linux-x86_64\.sh|CLAUDE\.md)$'
CHECK='cd ~/Retrieval-hate 2>/dev/null || { echo "  NO REPO"; exit 0; }
echo "  commit: $(git rev-parse --short HEAD)  dirty: $(git status --short | grep -v "^??" | wc -l)  untracked: $(git status --short | grep "^??" | wc -l)"
stray=$(ls ~ | grep -vE "'"$ALLOW"'")
[ -n "$stray" ] && echo "  STRAY in ~: $(echo $stray | tr "\n" " ")" || echo "  ~ clean"'
for h in local uoa-lab1 uoa-lab3; do
  echo "== $h"
  if [ "$h" = local ]; then bash -c "$CHECK"; else timeout 60 ssh "$h" "$CHECK" || echo "  ssh failed"; fi
done
