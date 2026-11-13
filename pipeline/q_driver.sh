#!/bin/bash
# usage: q_driver.sh BUDGET CONDS enc:var [enc:var ...]   (sequentially runs the combos inside one budget)
cd "${PCOS_BENCH_ROOT:-.}"; B=$1; C=$2; shift 2; T0=$(date +%s)
for ev in "$@"; do e=${ev%%:*}; v=${ev##*:}; rem=$(( B - ($(date +%s) - T0) )); [ $rem -lt 20 ] && break
  python3 pipeline/11_quantum.py run $rem --enc $e --var $v --conds $C; done
