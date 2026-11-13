#!/usr/bin/env bash
# Add NVIDIA's CUDA binary cache (cache.nixos-cuda.org) to the system Nix config and restart the daemon.
# CUDA is unfree, so cache.nixos.org has no CUDA packages; without this cache Nix prepares them locally.
#
#   sudo ./tools/setup-nix-cuda-cache.sh
#
# Safe to run twice: it changes nothing when the cache is already configured. Keeps a backup of nix.conf.
set -euo pipefail

conf=/etc/nix/nix.conf
url=https://cache.nixos-cuda.org
key=cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M=

[[ $EUID -eq 0 ]] || { echo "run with sudo: sudo $0" >&2; exit 1; }
[[ -f $conf ]] || { echo "$conf not found" >&2; exit 1; }

if grep -qF "$key" "$conf"; then
    echo "already configured in $conf"
else
    backup=$conf.bak-$(date +%Y%m%d-%H%M%S)
    cp -a "$conf" "$backup"
    echo "backup: $backup"
    cat >>"$conf" <<EOF

# CUDA binary cache (CUDA is unfree, so cache.nixos.org has none); added by p100-strata/tools/setup-nix-cuda-cache.sh
extra-substituters = $url
extra-trusted-public-keys = $key
EOF
    echo "added $url to $conf"
fi

systemctl restart nix-daemon
systemctl is-active --quiet nix-daemon || { echo "nix-daemon did not come back up: systemctl status nix-daemon" >&2; exit 1; }
echo "nix-daemon restarted"

echo
echo "substituters now:"
# root does not read the user's ~/.config/nix/nix.conf (where flakes/nix-command are enabled), so ask for it here
nix --extra-experimental-features nix-command config show substituters
