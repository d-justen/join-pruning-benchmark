# Join-Pruning Benchmark Generator

This repository contains a small Python generator for a DuckDB benchmark that targets single-join semi-join reduction / sideways information passing experiments.

## What It Generates

- A DuckDB database with:
  - `dim(d_key, dummy_s*, mask_c*)`
  - `fact(fact_id, fk_ordered, fk_uniform, measure)`
- A query directory with:
  - one `ordered` query per cluster count
  - one `uniform` query per cluster count
  - `metadata.json` describing the generated cluster layouts

## CLI

```bash
python3 gen_benchmark.py \
  --db out/sip.duckdb \
  --queries-dir out/queries \
  --build-size 100000 \
  --domain-multiplier 8 \
  --fact-rows 10000000 \
  --cluster-counts 2,4,8,16 \
  --dummy-string-cols 2 \
  --dummy-string-len 24 \
  --seed 7
```

## Parameter Semantics

- `build_size`: number of post-predicate dimension rows
- `domain_multiplier`: total dimension rows are `build_size * domain_multiplier`
- `fact_rows`: number of fact rows
- `cluster_counts`: number of key clusters retained by each mask
- `dummy_string_cols`: number of dummy dimension payload columns selected with `ANY_VALUE`
- `dummy_string_len`: width of each dummy payload column

## Cluster Layouts

For a requested cluster count `C`, the generator creates:

- `C` retained key ranges
- `C - 1` interior gaps
- first retained range starts at the smallest key
- last retained range ends at the largest key
- range and gap sizes are balanced as evenly as possible

The balanced interval layouts are precomputed into boolean mask columns on `dim`, so the query predicate is just `WHERE d.mask_cXXXXX`.

## Cross-Product Workload Generator

The repository also contains a second generator under [cross_product_benchmark](/Users/d-justen/Development/join-pruning-benchmark/cross_product_benchmark) for a workload that is the cross-product of:

- build size
- selectivity
- dimension predicate type
- probe clustering

It materializes:

- `dim_100k`
- `dim_100m`
- `fact(fk_100k_ordered, fk_100k_uniform, fk_100m_ordered, fk_100m_uniform, ...)`

Each dimension table gets one boolean mask per `(selectivity, predicate_type)` combination:

- `range`
- `shoulder`
- `seasonal`
- `scattered`

Current fixed workload:

- build sizes: `100000`, `100000000`
- selectivities: `0.5%`, `5%`
- predicate types: `range`, `shoulder`, `seasonal`, `scattered`
- probe clusterings: `ordered`, `uniform`
- fact rows: `100000000`

Run it with:

```bash
python3 -m cross_product_benchmark.cli \
  --db out/cross-product.duckdb \
  --queries-dir out/cross-product-queries \
  --seed 7
```

This emits 32 query files, for example:

- `b100k_s0005_range_ordered.sql`
- `b100m_s0050_scattered_uniform.sql`
