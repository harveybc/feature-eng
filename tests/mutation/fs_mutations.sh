#!/bin/bash
# Mutation check: each mutation must turn its FS family red.
W=$(cd "$(dirname "$0")/../.." && pwd)
S=${1:?usage: fs_mutations.sh <scratch dir outside /tmp>}
run() { # name test python-replacement-script
  d=$S/$1; rm -rf $d; mkdir -p $d/tools $d/tests; cp $W/tools/progressive_selection.py $W/tools/profile_train_wide.py $d/tools/; cp $W/tests/__init__.py $W/tests/_ps_fixtures.py $W/tests/test_fs*.py $W/tests/test_ps_edge_cases.py $d/tests/
  python3 -c "$3" $d/tools/progressive_selection.py || { echo "$1 MUTATION_NOT_APPLIED"; return; }
  out=$(cd $d && env OPENBLAS_NUM_THREADS=1 python -m unittest $2 2>&1 | tail -1)
  echo "$1 -> $2 : $out"
}
R='import sys;p=sys.argv[1];s=open(p).read();a,b=OLD,NEW;assert a in s;open(p,"w").write(s.replace(a,b,1))'
m() { echo "${R/OLD,NEW/$1}"; }
run fs01_full_rows tests.test_fs01_future_perturbation "$(m '"    a, b = fold[\"train\"]\n    Xf = X[a:b]","    a, b = fold[\"train\"]\n    b = len(X)\n    Xf = X[a:b]"')"
run fs02_self_forecast tests.test_fs02_probe_targets "$(m '"    if source_column != asset_column:","    if False:"')"
run fs15_failure_zero tests.test_fs15_metric_coverage "$(m '"cell(m, \"FAILED\", reason=f\"{type(exc).__name__}: {exc}\"[:200])","cell(m, \"MEASURED\", value=0.0)"')"
run fs16_no_synergy tests.test_fs16_reversible_priority "$(m '"            if gain > floor:","            if False:"')"
run fs19_no_identity tests.test_fs19_profile_reuse "$(m '"return _sha({\"identity\": identity,","return _sha({\"identity\": None,"')"
run fs02_unit_bug tests.test_fs02_real_timestamps "$(m '"    return ((dt - pd.Timestamp(\"1970-01-01\")) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)","    return dt.astype(\"int64\").to_numpy() // 10**9"')"
