#!/usr/bin/env bash
set -Eeuo pipefail

# Optional integration check for the real bcftools/plugin chain. The Python
# test suite covers the reducer separately because CI may not have bcftools.
BCFTOOLS="${BCFTOOLS:-bcftools}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

actual="$({
  "$BCFTOOLS" view --threads 1 -Ou \
    -S "$ROOT/tests/data/toy.samples.txt" \
    -T "$ROOT/tests/data/toy.callable.bed" \
    "$ROOT/tests/data/toy.vcf"
} | {
  "$BCFTOOLS" +fill-tags -Ou -- \
    -S "$ROOT/tests/data/toy.groups.tsv" -t AC,AN
} | {
  "$BCFTOOLS" query \
    -f $'%POS0\t%REF\t%ALT\t%FILTER\t%AC_AFR{0}\t%AN_AFR\t%AC_EUR{0}\t%AN_EUR\n'
})"

expected=$'0\tA\tG\tPASS\t1\t4\t0\t4\n9998\tC\tT\tPASS\t4\t4\t1\t4\n10000\tA\tC\tPASS\t1\t4\t1\t4\n10000\tA\tG\tPASS\t1\t4\t1\t4\n14999\tA\tAT\tPASS\t1\t4\t1\t4\n19999\tA\tC,G\tPASS\t1\t4\t0\t4\n20999\tG\tA\t.\t1\t4\t1\t4\n21999\tG\tA\t.\t1\t4\t1\t4\n21999\tG\tAT\tq10\t1\t4\t1\t4'

if [[ "$actual" != "$expected" ]]; then
  printf 'unexpected grouped AC/AN output\n--- expected ---\n%s\n--- actual ---\n%s\n' \
    "$expected" "$actual" >&2
  exit 1
fi

printf 'bcftools fixture pipeline: ok\n'
