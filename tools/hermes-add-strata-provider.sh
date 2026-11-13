#!/usr/bin/env bash
# Register the local Strata server as a named Hermes provider ("strata"), without changing the active model.
#
#   ./tools/hermes-add-strata-provider.sh
#
# Afterwards pick it per session with `/model` (or `hermes model`); the default model stays as it was.
# Backs up ~/.hermes/config.yaml first. Safe to run again (the same keys are set to the same values).
set -euo pipefail

url=${STRATA_URL:-http://127.0.0.1:8793/v1}
model=${STRATA_MODEL:-qwen3.8-flash-next-iq3s-strata}
ctx=${STRATA_CONTEXT:-262144}

conf=${HERMES_HOME:-$HOME/.hermes}/config.yaml
cp -a "$conf" "$conf.bak-strata-$(date +%Y%m%d-%H%M%S)"

before=$(hermes config get model.default 2>/dev/null || grep -m1 -E '^\s+default:' "$conf")

hermes config set providers.strata.name "Strata (local P100)"
hermes config set providers.strata.api "$url"
hermes config set providers.strata.transport chat_completions
hermes config set providers.strata.default_model "$model"
hermes config set providers.strata.context_length "$ctx"
# `/model strata` (session only; `/model strata --global` would make it the default). --force: `config set`
# does not know model_aliases and warns, but Hermes reads it (checked: `hermes chat -m strata` reaches Strata)
hermes config set --force model_aliases.strata.model "$model"
hermes config set --force model_aliases.strata.provider custom
hermes config set --force model_aliases.strata.base_url "$url"

after=$(hermes config get model.default 2>/dev/null || grep -m1 -E '^\s+default:' "$conf")
[[ "$before" == "$after" ]] || echo "WARNING: model.default changed: $before -> $after" >&2

echo
echo "providers.strata -> $url ($model, context $ctx); active model unchanged: $after"
echo "use it:  /model strata   (in a session)   or   hermes chat --provider strata -m $model"
curl -fsS -m 5 "$url/models" >/dev/null && echo "server is up" || echo "server not reachable right now (start it first)"
