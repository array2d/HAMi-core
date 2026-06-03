#!/bin/bash
# cu_hook_lookup benchmark: old strcmp chain vs new Trie dispatch
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> compiling benchmark..."
gcc -O2 -Wall -o cu_hook_bench cu_hook_bench.c

echo "==> running (this takes ~10-20 seconds)..."
echo
./cu_hook_bench

echo
echo "==> done."
