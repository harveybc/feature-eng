#!/bin/bash
set -e
R=$HOME/.local/state/crispdm-data-foundation/m03_feature_inventory_20260930
E=$R/eurusd_lake_5m_to_1h; cd $E
PY=$HOME/anaconda3/envs/tensorflow/bin/python
ENVS="env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="
F=eurusd_1h_from_lake_5m.csv
N=$(($(wc -l < $F) - 1))
P=$(awk -F, 'NR>1 && $1 >= "2024-01-01" {print NR-2; exit}' $F)   # rows before the first 2024 row
NT=$((N*70/100)); NV=$((N*85/100)); S2T=$((P*80/100))
TT=$(awk -F, -v r=$((NT+1)) 'NR==r{print $1}' $F)
echo "N=$N P=$P NT=$NT NV=$NV S2T=$S2T last_S1_train_ts=$TT"
SHA=$(sha256sum $F | cut -c1-64)
cat > manifest.v2.json <<J
{"schema":"feature_train_manifest.v2","split":"TRAIN","dataset_id":"lake_derived.eurusd_1h_from_histdata_5m.train","governance":"LOCAL_FILE",
 "resource_sha256":"$SHA","path":"$F","path_note":"lane B derivative of lake features/trading_asset_data/eurusd/5m.parquet (c746f344), PROVENANCE.json e66903ca",
 "registered_rows":$N,"columns_total":6,"split_rule":"S1 70/15/15 chronological rows: TRAIN [0,$NT); this TRAIN ends before 2024 so it is also inside S2 TRAIN+validation",
 "boundaries":{"train":[0,$NT]},"timestamp_column":"DATE_TIME","timestamp_format":"%Y-%m-%d %H:%M:%S","step_seconds":3600,
 "default_role":"feature","target_channels":[],"declared_periods_rows":{"day_rows_nominal":24,"trading_week_rows_nominal":120},
 "primary_period":"day_rows_nominal","max_missing_fraction":0.2,
 "columns":{"N_5M_BARS":{"role":"excluded","reason":"DATA_COMPLETENESS_COUNT: number of 5m bars aggregated into the hour; a quality field, not a market feature"}}}
J
rm -rf profile declaration.json
CUDA_VISIBLE_DEVICES="" $HOME/.local/bin/crispdm-run -q -W 1800 -m 1G -t 20m -n laneB-lake-eurusd-profile -- $ENVS $PY $R/codeH/tools/profile_train_wide.py --manifest manifest.v2.json --source-root $E --output $E/profile
CUDA_VISIBLE_DEVICES="" $HOME/.local/bin/crispdm-run -q -W 1800 -m 256M -t 5m -n laneB-lake-eurusd-declare -- $ENVS $PY $R/codeH/tools/declare_admissible_inputs.py --profile profile/profile.json --output declaration.json
echo "$N $P $NT $NV $S2T" > split_rows.txt
