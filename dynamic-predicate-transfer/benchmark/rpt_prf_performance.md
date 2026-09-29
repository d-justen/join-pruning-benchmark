# PRF performance work

## First implementation round

1. Keep native scan equality/min/max filters when PRF is selected, alongside PRF statistics pruning.
2. Add a selection-producing runtime-filter interface. Keep Cascade's existing lookup as the default implementation; let PRF write selections directly.
3. Specialize PRF lookup once per vector by representation, validity and direct-range count. Use safe branchless bitmap addressing and unrolled direct-range checks.
4. Test the actual PRF kernels for flat, dictionary, constant and nullable vectors, all integer widths, integer extremes, empty/unfinalized filters and all representations. Run the SQL integration tests.
5. Run the full JOB suite before and after, with eight threads, the same complete database cache, sequential execution and reversed mode order between rounds. Retain raw output and verify every query against the benchmark's expected results.

Compression, sizing, construction and full scan-level PRF tuple filtering are deferred so this round measures changes to early filtering and probing.

All five steps are complete. The engine changes are in:

- `src/execution/operator/persistent/physical_create_bf.cpp`: retain scan predicates for PRF edges.
- `src/include/duckdb/optimizer/predicate_transfer/bloom_filter/runtime_filter.hpp`: selection API with a default adapter for Cascade.
- `src/execution/operator/filter/physical_use_bf.cpp`: consume filter selections while preserving compaction and adaptive disablement.
- `src/optimizer/predicate_transfer/bloom_filter/prefix_range_filter.cpp`: fused PRF selection, flat/all-valid and constant fast paths, dictionary/null handling, safe bitmap addressing and specialized direct-range counts.

## Validation

The release build succeeded. The three direct PRF test cases passed 802,731 assertions, and the PRF SQL integration and existing join-filter pushdown tests passed 32 assertions. The direct tests compare selections with both exact membership (where the representation is exact) and the existing scalar lookup, so passing all rows cannot satisfy them. They exercise all eight integer types, one through four singleton and multi-value ranges, bitmaps, approximate full-width signed/unsigned domains, repeated/reordered dictionary positions, nulls, constants, zero-length chunks and unavailable/empty filters.

```sh
build/release/test/unittest '[rpt_prf]'
build/release/test/unittest 'test/sql/optimizer/test_rpt_prefix_range_filter.test,test/optimizer/pushdown/join_filter_pushdown.test'
```

Every timed JOB execution also matched the benchmark's expected answer. No sanitizer/debug build was run in this round.

## JOB comparison

Measured on the local arm64 machine, with the release build and eight DuckDB threads. The database cache was already fully loaded. Each stage ran two sequential sweeps in these orders: Cascade, PRF, none; then none, PRF, Cascade. No build, test or other benchmark ran concurrently with these sweeps. Each query had one untimed warmup and five timed repetitions. There are 1,130 timed samples per mode per stage, and 6,780 verified timed executions overall. Query profiling was disabled for these measurements.

Times below are the average total over a sweep of 113 queries times five timed repetitions, not the runtime of one execution of each query.

| Mode | Before, seconds | After, seconds | Change |
|---|---:|---:|---:|
| RPT+ Cascade | 24.2815 | 24.0525 | -0.9% |
| RPT+ PRF | 28.2550 | 23.1349 | **-18.1%** |
| No RPT+ | 31.0673 | 31.0454 | -0.1% |

After these changes, PRF is **3.8% faster than Cascade** and **25.5% faster than no RPT+** by this aggregate. It was 3.2% and 4.5% faster than Cascade in the individual sweeps. These are local measurements from two sweeps, not a cross-machine performance guarantee.

| Mode | Before sweep 1 | Before sweep 2 | After sweep 1 | After sweep 2 |
|---|---:|---:|---:|---:|
| Cascade | 24.221564 | 24.341505 | 23.986727 | 24.118311 |
| PRF | 28.134448 | 28.375458 | 23.230150 | 23.039692 |
| None | 31.174474 | 30.960081 | 31.082292 | 31.008503 |

Using each query's median across its ten timed samples, the sum of medians changed from 5.640885 to 4.618603 seconds for PRF (also an 18.1% improvement). The new Cascade sum is 4.801759 seconds. PRF improved over its previous implementation on 110 of 113 queries and beat the new Cascade implementation on 74 of 113 queries.

Examples of PRF improvements, using per-query medians:

| Query | Before, ms | After, ms | Reduction |
|---|---:|---:|---:|
| 31b | 100.404 | 51.538 | 48.7% |
| 26b | 80.271 | 51.540 | 35.8% |
| 26c | 90.112 | 61.598 | 31.6% |
| 16a | 52.543 | 31.232 | 40.6% |

The three regressions against old PRF were 10b (33.919 to 35.254 ms), 06f (108.409 to 109.449 ms), and 32a (3.412 to 3.438 ms). No universal per-query speedup is claimed.

Raw samples and complete logs are retained locally under `build/rpt-prf-performance/`: `before.csv`, `after.csv`, and one `.log` per stage/sweep/mode. The local `run.py` helper rejects incomplete suites or ERROR/INCORRECT/TIMEOUT output; `summarize.py` computes the aggregates above. To repeat a mode directly:

```sh
build/release/benchmark/benchmark_runner 'benchmark/imdb/.*.benchmark' --threads=8 --rpt_filter prf --disable-timeout
```

Run Cascade and none with the same arguments and their respective `--rpt_filter` values. Do not run these settings concurrently against the shared benchmark cache.

## Operator-level confirmation

After the full benchmark, six alternating profiled repetitions per mode were run at one thread on 22c, 17b and 14a. These profiling results are separate from the full-JOB timing table. Timings are median USE_BF operator times, including selection/compaction, not isolated lookup timings.

| Query/operator | Cascade input rows | PRF input rows | Cascade time, ms | PRF time, ms |
|---|---:|---:|---:|---:|
| 22c: small-key movie_keyword filter | 754,782 | 754,782 | 2.999 | 1.523 |
| 17b: singleton movie_keyword filter | 41,838 | 41,838 | 0.436 | 0.434 |
| 17b: large cast_info bitmap filter | 36,237,723 | 36,237,723 | 97.474 | 40.507 |
| 14a: small-key movie_keyword filter | 633,612 | 633,612 | 2.520 | 1.280 |

The previous integration sent 4,523,913 movie_keyword rows to PRF in both 22c and 17b. Restoring scan predicates removes that extra work. The large bitmap case now shows PRF's expected probe advantage at identical input cardinality, while the direct-range cases are faster or avoid redundant filtering. These changes were implemented together, so the full-JOB improvement is attributed to the combined changes rather than partitioned into additive percentages.

Compression, filter construction, materialization, scans and joins remain part of total query cost. Efficient compression and measured construction optimizations are the next candidates; full PRF tuple filtering inside scans remains a separate integration step.

## Second implementation round: compression, construction and range index

The implementation plan for this round was:

1. Replace per-set-bit dyadic reduction with the baseline's word-level pair packing. Fuse destination population/run analysis with reduction, retain run endpoints for direct-range conversion, stop run analysis after discovering more than four runs, and reuse two bitmap allocations across accepted levels.
2. Remove atomic updates from exclusively owned construction bitmaps. Use the global bitmap for a single task; for parallel builds, create private bitmaps and merge them after all producers finish. Bound task count by workers, available chunk groups and a 64 MiB private-storage budget per build event.
3. Replace the list of every nonzero word with an index of long homogeneous empty/full word runs. Scan short ranges directly, classify long ranges through the index, and support both always-false and always-true results. Preserve conservative handling of NULLs and probes extending beyond build bounds.
4. Expose separate allocation/initialization, insertion, merge, compression and index timings in CREATE_BF profile metadata. Add scalar-reference compression and parallel/range tests, including sanitizer validation. Rerun JOB in all three settings against a fresh pre-change measurement.

All three algorithmic changes are implemented. Point-probe kernels, scan filter placement, integer mapping, bitmap sizing, the 0.001 compression-FPR budget and the 16 KiB cache target are unchanged. The reference for the port is the locally cached source from the supplied `prf-v2` branch, not a newly verified branch-head commit.

### Implementation details and remaining differences

- Compression now reduces two 64-bit source words to one destination word using the baseline shift/mask sequence. Destination statistics and direct-range endpoints are collected in that pass; conversion no longer walks individual buckets. Rejected candidates leave the source untouched. Accepted levels swap reusable buffers, and the final retained bitmap is right-sized. One initial source-metrics pass is still separate from the first reduction; the upstream implementation can fuse those two passes as well.
- Parallel private storage is capped **per build event**, not per query or process. This is not yet integrated with the temporary-memory manager's reservation policy. If the budget permits only one task, construction is serial and needs no private bitmap. Cascade-only task scheduling remains unchanged. Mixed PRF/Cascade creators share the bounded task schedule; Cascade keeps its existing insertion path.
- Private allocation is currently performed when scheduling tasks, and merges are serial after the producers complete. These are measurable remaining differences from a fully optimized construction pipeline. The existing public atomic Insert path is retained for concurrent callers outside this task-private path.
- The range index stores only homogeneous runs longer than 2,048 complete words. Partial final words are handled by boundary masks. ALWAYS_TRUE means the PRF accepts every possible non-NULL value in the statistics interval, not that every value exists in the join build set. Nullable statistics downgrade that result to unknown.
- `PRF Build Timings` in CREATE_BF `extra_info` reports `init_ms`, `insert_task_ms`, `merge_ms`, `compress_ms` and `index_ms`, separately for each unique filter. Initialization includes global and private allocations/clears. Insertion is **summed task elapsed time**, not critical-path wall time; it excludes materialized-chunk fetches and min/max aggregation. The phase totals therefore cannot be added to infer total query latency. Instrumentation runs once per insertion chunk and once per phase, never per tuple.
- Forced-parallel sanitizer testing exposed an existing shared RPT+ memory-estimation bug: zero/unavailable progress could produce infinity and then an undefined integer conversion. The estimate now waits for meaningful progress and compares floating-point byte estimates without overflowing an integer. This small correctness safeguard applies to both Cascade and PRF and is included in the after build.

The next candidates remain source/first-reduction fusion, flat/all-valid construction specialization, memory-reservation-aware parallel allocation/merge, and scan-level tuple filtering. These should be prioritized using the new phase timings rather than assuming their costs dominate.

### Second-round validation

The release build and the assertion-enabled debug build with AddressSanitizer and UndefinedBehaviorSanitizer both passed **2,475,938 assertions across seven test cases**. The suite includes all previous vector/type tests, exhaustive point-membership comparison with an independent scalar dyadic compressor over 24 domain/pattern combinations, actual four-thread private-bitmap construction, long empty/full and mixed ranges, partial-word boundaries, NULL statistics, outside-bound intervals, and a forced-parallel RPT+ SQL join with duplicate and NULL keys. The existing join-filter pushdown test also passes. These sanitizer runs are not ThreadSanitizer race checks.

```sh
build/release/test/unittest '[rpt_prf],test/sql/optimizer/test_rpt_prefix_range_filter.test,test/optimizer/pushdown/join_filter_pushdown.test'
build/debug/test/unittest '[rpt_prf],test/sql/optimizer/test_rpt_prefix_range_filter.test,test/optimizer/pushdown/join_filter_pushdown.test'
```

### Second-round JOB results

The fresh `core-before` and `core-after` stages each ran two full sweeps with the same eight-thread configuration and complete database cache. Mode order was Cascade/PRF/none, then none/PRF/Cascade. Each sweep contained 113 queries with five timed repetitions each; profiling was disabled and no builds or tests ran concurrently. All **6,780 timed executions** across both stages verified against the benchmark's expected answers.

As above, each aggregate is the average total of 565 timed samples, not one execution of the suite.

| Mode | Before, seconds | After, seconds | Change |
|---|---:|---:|---:|
| RPT+ Cascade | 24.469645 | 24.850301 | +1.6% |
| RPT+ PRF | 23.256090 | 22.638925 | **-2.7%** |
| No RPT+ | 31.298548 | 32.198960 | +2.9% |

| Mode | Before sweep 1 | Before sweep 2 | After sweep 1 | After sweep 2 |
|---|---:|---:|---:|---:|
| Cascade | 24.544496 | 24.394794 | 24.808479 | 24.892123 |
| PRF | 23.195314 | 23.316866 | 22.608256 | 22.669594 |
| None | 31.608170 | 30.988926 | 31.780158 | 32.617761 |

PRF is **8.9% faster than Cascade** and **29.7% faster than no RPT+** in the after stage. Using per-query medians over ten samples, PRF improved on **100/113 queries** and beat Cascade on **106/113**. The sum of PRF query medians fell from 4.641274 to 4.519575 seconds (2.6%). The largest median improvement was 10c, from 149.958 to 113.582 ms (24.3%); the largest absolute regression was 03a, from 35.339 to 36.135 ms.

The control modes slowed by 1.6–2.9%, so the widened PRF/Cascade gap is not solely PRF's own measured improvement. There are only two sweeps, and adaptive RPT+ decisions can be timing-sensitive. These results support a modest aggregate improvement from this round, not an isolated causal percentage for each component or a general performance guarantee. No component-by-component ablation was run.

Raw CSVs and logs are under `build/rpt-prf-performance/core-before*` and `core-after*`. The local comparison command is:

```sh
python3 build/rpt-prf-performance/summarize.py core-before core-after
```

### Phase measurements and next priorities

After JOB finished, six profiled repetitions were collected for selected queries at one and eight threads. The following are medians of the summed PRF phase times over all unique filters in each eight-thread query, in milliseconds. They are separate profiling runs, not samples used in the JOB table.

| Query | Initialize | Insert (task sum) | Merge | Compress | Range index |
|---|---:|---:|---:|---:|---:|
| 22c | 0.0405 | 1.1838 | 0 | 0.7233 | 0.0031 |
| 17b | 0.0131 | 2.8420 | 0 | 0.4770 | 0.0013 |
| 14a | 0.0102 | 0.9020 | 0 | 0.2969 | 0.0005 |
| 31b | 0.0191 | 0.9049 | 0 | 0.6158 | 0.0007 |
| 10c | 0.0222 | 7.1770 | 0.1271 | 0.1591 | 0.0003 |

The first four sampled queries did not use private-bitmap merging: their creators selected the existing single-task path even with eight query threads. The non-atomic single-writer path is therefore important independently of parallel construction. Query 10c does exercise merging, with 0.1271 ms of merge time; its 7.1770 ms insertion value is summed across tasks, not wall time. In these examples, initialization/index construction are already tiny; insertion is the largest measured PRF phase and compression remains second. Materialization, min/max aggregation, probes, scans and joins are outside these phase totals.

The next practical order is **flat/all-valid insertion specialization**, then **source-metrics/first-reduction fusion**. Tune parallel allocation/merging using queries such as 10c that actually exercise that path; without an ablation, the sampled phase timings do not establish the isolated parallel-build speedup. Further range-index construction tuning is low priority for these queries. Scan-level tuple filtering remains a separate integration change with potentially larger effects than these remaining kernel refinements.

The phase profiles are retained in `build/rpt-prf-performance/core-profiles/`; `core_profile.py` generated them without overwriting earlier profiles. There are no corresponding pre-round per-phase counters, so these numbers describe remaining costs, not a before/after phase-level speedup.

## Third round: PRF scan pushdown

Implemented stages 1–4:

1. **Scan selection kernel.** Refine an incoming selection in original vector coordinates, handling flat, dictionary, constant and nullable inputs. Reuse the specialized bitmap/direct-range kernels without an intermediate Boolean result array.
2. **Storage dispatch.** Execute the PRF table filter in `ColumnSegment::FilterSelection`; retain statistics pruning. Thread-local, growable selection buffers support conjunctions and codec calls larger than `STANDARD_VECTOR_SIZE`. Constant-validity filtering rejects NULL keys.
3. **Placement and publication.** Find eligible native scans through streaming filters, existing `USE_BF` operators and reference-only projections. Stop at materialization, joins, computed projections, sampled scans, virtual/nested keys and non-native sources. Publish application-specific ownership only from the actual sequential scan's global initialization, after dynamic/static filter merging. Never claim optional or OR children, or index sources. A dropped filter cannot disable the operator fallback.
4. **Single point-filter location.** A scan-owned application replaces its generated min/max predicates and bypasses downstream PRF probes. Keep `USE_BF` for build dependencies and chunk compaction. Unsupported applications retain the existing operator path, including min/max and optional statistics filters where available. `Tuple Placement` in profiling reports actual ownership. No new adaptive scan-disable policy is included.

`SET rpt_prf_scan_pushdown = false` restores the operator placement for a same-binary comparison. It defaults to true and affects PRF only. Cascade's filtering algorithm and placement are unchanged.

### Lifecycle corrections discovered by validation

The existing RPT+ creator cached a pipeline on the physical plan and reused it across prepared executions. Both Cascade and PRF crashed on the second execution. Pipeline dependency lookup now belongs to the current execution's `PipelineBuildState`. Builds reset success, dynamic predicates and PRF validity so stale filters cannot survive an abandoned build; downstream use also checks build success. These lifecycle safeguards apply to both modes where relevant, but do not change Cascade's filtering algorithm.

A separate pre-existing issue remains: a prepared plan can keep stale static statistics bounds after inserts expand a key domain. This reproduces with `rpt_filter='none'`, independently of PRFs. The mutation regression stays inside the original domain; replan queries after domain expansion until that broader issue is fixed.

### Correctness validation

Release and ASan/UBSan debug builds each passed **2,659,518 assertions in 9 cases**:

```sh
build/release/test/unittest '[rpt_prf],test/sql/optimizer/test_rpt_prefix_range_filter.test,test/optimizer/pushdown/join_filter_pushdown.test'
build/debug/test/unittest '[rpt_prf],test/sql/optimizer/test_rpt_prefix_range_filter.test,test/optimizer/pushdown/join_filter_pushdown.test'
```

Tests compare sparse/reversed scan selections against scalar membership across all eight supported integer types and vector encodings, exercise reusable storage selections beyond a standard vector, and verify real scan ownership against operator placement. Parallel prepared executions are repeated before/after checkpointing and with transaction-local inserts, updates, deletes and NULLs. Existing compression, extreme-domain, empty/unavailable-filter and join-filter regressions remain included. An additional forced-RLE checkpointed join returned 285,715 rows with both PRF and no RPT+.

### Reproduce the scan comparison

With the complete IMDb cache already loaded:

```sh
python3 benchmark/run_rpt_prf_comparison.py scan-after --rounds 3 --scan-comparison
```

The script refuses to overwrite existing logs. Choose a new stage name for another run. It runs Cascade, operator PRF, scan PRF and no RPT+ sequentially, reverses their order on alternating rounds, and checks all 113 expected answers and all five timed samples per query. Raw logs and CSVs are kept under `build/rpt-prf-performance/`. Profiling, builds and tests are not run concurrently with timing.

### Third-round JOB results

All **6,780 timed executions** in the same-binary comparison matched the expected answers. Eight threads, three complete sweeps per mode, five timed executions per query. Values below are totals over 565 samples, averaged across the three sweeps—not the time for one execution of each query.

| Mode | Sweep 1 (s) | Sweep 2 (s) | Sweep 3 (s) | Mean (s) |
|---|---:|---:|---:|---:|
| Cascade | 24.057386 | 24.416929 | 24.724776 | 24.399697 |
| PRF, operator | 22.869418 | 22.739364 | 22.780323 | 22.796368 |
| PRF, scan | 18.522828 | 18.585817 | 18.662228 | **18.590291** |
| No RPT+ | 31.271863 | 31.256494 | 31.289738 | 31.272698 |

Scan PRF reduced aggregate time by **18.5% versus operator PRF**, **23.8% versus Cascade**, and **40.6% versus no RPT+**. Using per-query medians over all 15 samples, scan PRF improved **102/113 queries** versus operator PRF. The sum of query medians fell from 4.539723 to 3.706408 seconds (18.4%). This is a consistent improvement in this workload/configuration, not a universal speedup guarantee or an isolated attribution to each implementation component.

Largest absolute median gains included 30b (59.743 → 28.442 ms), 18a (61.129 → 30.066 ms), and 25b (66.389 → 36.484 ms). Eleven queries regressed: the largest absolute regression was 14b (15.114 → 16.188 ms, 7.1%); 06f and 31b increased by 0.751 and 0.695 ms respectively. Scan-level adaptive disabling remains deferred and may help some low-benefit applications; these results alone do not establish each regression's cause.

A separate fresh pre-change baseline (`scan-before`, also three sweeps) averaged 24.364115 s Cascade, 22.802185 s PRF, and 31.254221 s no RPT+. The new operator-PRF control is essentially unchanged (-0.03%), with Cascade +0.15% and no RPT+ +0.06%. The primary pushdown comparison remains the same-binary table above.

### Placement evidence and remaining limits

Separate profiles taken after benchmarking show that applications really move into scans, and that other applications retain operator ownership:

| Query | Scan-owned / total PRF applications | Scan output rows, operator PRF | Scan output rows, scan PRF |
|---|---:|---:|---:|
| 10c | 3 / 8 | 9,155,837 | 7,663,987 |
| 17b | 5 / 9 | 41,711,015 | 1,363,193 |
| 22c | 8 / 19 | 4,061,570 | 842,115 |
| 31b | 9 / 20 | 1,705,835 | 1,067,819 |

Row counts sum emitted cardinalities across all table scans in the query; they are not physical rows read or a measurement of avoided decoding. Codec-specific selection can save additional work, but this fork does not guarantee late materialization for every column/codec. Query 31b also illustrates that fewer emitted rows do not guarantee lower end-to-end latency. Raw placement profiles are in `build/rpt-prf-performance/scan-profiles/`.

This is native-scan pushdown, not full feature parity with every upstream PRF scan consumer. Non-native scans, materialization boundaries, computed projections and index execution retain operator fallback. If filter unification drops a PRF, the surviving scan tree does not claim ownership. For a planned scan application that ultimately uses an index, its generated min/max has already been omitted, but the complete operator PRF still checks bounds and membership. Broader placement coverage, adaptive disabling and deeper codec integration are separate follow-ups.
