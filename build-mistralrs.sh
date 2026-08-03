#!/usr/bin/env bash
# Cross-compile the fork's `mistralrs` CPU binary for BOTH linux arches from one
# host via cargo-zigbuild (zig cc supplies the cross toolchain + glibc — no QEMU,
# no per-arch Docker). Outputs dist/amd64/mistralrs and dist/arm64/mistralrs.
#
# Deps (checked below): rustup, zig (`brew install zig` or `pip install ziglang`),
# cargo-zigbuild (auto-installed if missing).
#
# Usage:  scripts/build-mistralrs.sh [amd64|arm64|both]   (default: both)
set -euo pipefail

want="${1:-both}"
repo="${MISTRALRS_REPO:-https://github.com/sandyz1000/mistral.rs.git}"
ref="${MISTRALRS_REF:-33853e3f427b83c8b1dc297832137107a7674134}"
# Pin a conservative glibc so the binary runs on the debian:trixie runner (and older).
glibc="${GLIBC:-2.31}"

root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/dist/src"   # under dist/ (git-ignored, .dockerignore'd)

amd64_triple="x86_64-unknown-linux-gnu"
arm64_triple="aarch64-unknown-linux-gnu"

command -v rustup >/dev/null || { echo "need rustup" >&2; exit 1; }
command -v cargo-zigbuild >/dev/null || cargo install --locked cargo-zigbuild

# Pick a zig cargo-zigbuild understands. A pip-pinned `ziglang` (e.g. 0.13.0)
# beats a too-new PATH zig (brew's 0.16 breaks `zig cc -v` parsing).
if [ -n "${ZIG_COMMAND:-}" ]; then
  :
elif python3 -m ziglang version >/dev/null 2>&1; then
  export ZIG_COMMAND="python3 -m ziglang"
elif ! command -v zig >/dev/null; then
  echo "need zig: pip install 'ziglang==0.13.0'  (or brew install zig)" >&2; exit 1
fi

targets=()
case "$want" in
  amd64) targets=("$amd64_triple") ;;
  arm64) targets=("$arm64_triple") ;;
  both)  targets=("$amd64_triple" "$arm64_triple") ;;
  *) echo "arg must be amd64|arm64|both" >&2; exit 1 ;;
esac
for t in "${targets[@]}"; do rustup target add "$t" >/dev/null; done

if [ ! -d "$src/.git" ]; then
  git init "$src"
  git -C "$src" remote add origin "$repo"
fi
git -C "$src" fetch --depth 1 origin "$ref"
git -C "$src" checkout -q FETCH_HEAD

# The fork's .cargo/config.toml pins `target-cpu=native` (= the host, e.g.
# apple-m2), which is fatal when cross-compiling to another arch. Drop it; keep
# the wasm block. RUSTFLAGS below also overrides it, but this removes any doubt.
printf '[target.wasm32-unknown-unknown]\nrustflags = ["-C", "target-feature=+simd128"]\n' \
  > "$src/.cargo/config.toml"

# zigbuild takes multiple --target in one invocation; glibc suffix pins the ABI.
zig_targets=()
for t in "${targets[@]}"; do zig_targets+=(--target "${t}.${glibc}"); done

# Empty RUSTFLAGS (still overrides the fork's `target-cpu=native`) — don't set an
# explicit target-cpu: build scripts forward it to `zig cc` as `-mcpu`, which
# clang rejects (e.g. `unknown target CPU 'generic'`). Default baseline is fine.
export RUSTFLAGS="${RUSTFLAGS:-}"
( cd "$src" && cargo zigbuild --release --locked --no-default-features \
    -p mistralrs-cli "${zig_targets[@]}" )

copy() {  # <triple> <arch-dir>
  mkdir -p "$root/dist/$2"
  cp "$src/target/$1/release/mistralrs" "$root/dist/$2/mistralrs"
  echo "-> dist/$2/mistralrs"
}
for t in "${targets[@]}"; do
  [ "$t" = "$amd64_triple" ] && copy "$t" amd64
  [ "$t" = "$arm64_triple" ] && copy "$t" arm64
done
