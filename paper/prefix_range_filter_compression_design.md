# Prefix Range Filter compression design

This design describes the compression implemented in this repository's `duckdb/` checkout as of 22 October 2026, at commit `0b737d677c9730fb8e48b8e89ed303f782f6bc73`. It covers compression-enabled filters, including cases where the compression policy retains the original bitmap. The optional baseline with compression disabled is outside its scope.

A Prefix Range Filter (PRF) maps a bounded key domain to occupied buckets. Compression repeatedly doubles the bucket width by OR-ing adjacent bitmap bits, subject to limits on false positives and coverage expansion. Construction can perform these reductions on temporary intervals or bucket IDs instead of repeatedly rewriting a bitmap. The installed filter has **two storage modes**: a bitmap or a multi-range representation containing at most four direct ranges. A large final bitmap can additionally have an **auxiliary range index** for accelerating statistics lookups.

The central implementation is [`PrefixRangeBitmap<U>`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp), with temporary interval operations in [`PrefixRangeIntervals`](../duckdb/src/include/duckdb/planner/filter/prefix_range_intervals.hpp). These structures describe in-memory runtime filters; there is no compressed byte-stream format involved.

## Domain and initial bitmap

Let `convert(key)` produce the comparable integer used by the filter, and define:

| Symbol | Meaning |
| --- | --- |
| `m` | Minimum comparable value |
| `span` | Maximum minus minimum |
| `D = span + 1` | Number of values in the inclusive domain |
| `s` | Number of low-order offset bits discarded |
| `w = 2^s` | Bucket width |
| `L = (span >> s) + 1` | Number of logical buckets |
| `W = ceil(L / 64)` | Number of physical bitmap words |
| `A` | Number of occupied buckets, or set bits |
| `R` | Number of maximal consecutive runs of set bits |

For an in-domain key, insertion computes:

```text
offset = convert(key) - m
bucket = offset >> s
bitmap[bucket >> 6] |= 1ULL << (bucket & 63)
```

Bucket `b` represents inclusive offsets `[b*w, min((b+1)*w - 1, span)]` from `m`. Buckets are aligned relative to `m`. A set bit says that at least one build key falls in that bucket. Duplicate keys set the same bit; NULL keys are skipped. Worker-local bitmaps are merged using wordwise OR before compression starts. A builder with exclusive access can populate the global bitmap directly.

Numeric filters use the corresponding unsigned physical type as `U`; subtraction from the converted minimum, stored in `U`, forms the domain offset. Signed values retain their bit pattern when converted, so domain offsets and stored endpoints can involve unsigned wraparound. Supported physical integer widths are 8, 16, 32, and 64 bits. Strings use a 32-bit, byte-order-comparable encoding of their first **four bytes**, with zero padding for short strings. These are bytes, not Unicode characters. Different full strings can therefore share a prefix even when `s = 0`. Compression operates on this prefix domain and does not recover string suffixes. The type conversion is implemented by `NumericConverter`, `StringPrefixConverter`, and [`string_t::GetPrefixIntegerComparable`](../duckdb/src/include/duckdb/common/types/string_type.hpp#L97).

The initial shift comes from the caller:

- **SQL hash joins:** [`PlanPrefixRangeFilter`](../duckdb/src/execution/operator/join/physical_hash_join.cpp#L1559) selects `s = 0` for at most `2^23` buckets. For larger domains, it chooses the smallest shift giving at most `2^26` buckets, so the initial bitmap payload is at most 8 MiB. Domains between these thresholds can still start at shift zero. The runtime compression FPR threshold is `0.001` (0.1%).
- **Standalone filter benchmarks:** `filter_benchmark`, `range_benchmark`, and `join_pruning_benchmark` call `TryComputeSizing` with their configured FPR limit. This helper starts at zero and considers shifts 1 through 63 until `N * (2^s - 1) / (span - N)` exceeds the limit; if `span <= N`, it immediately chooses zero. Here `N` is the supplied build-row count. This initial sizing formula is separate from the post-build policy below.

## Reduction and acceptance policy

One *dyadic reduction* merges each aligned pair of buckets:

```text
next[b] = current[2*b] OR current[2*b + 1]  // nonexistent tail bit is zero
next_shift = s + 1
next_logical_buckets = ceil(L / 2)
next_words = ceil(next_logical_buckets / 64)
```

Every accepted reduction preserves all previously accepted keys and can admit additional keys. It removes one more low-order bit from the domain offset. The logical bitmap approximately halves at each step, although word rounding can leave the physical word count unchanged for tiny bitmaps.

### False positive estimate

The implementation computes a coverage quantity:

```text
C(A, s) = min(A * 2^s, D)
```

This counts a full bucket width for every occupied bucket and caps the total at the domain size. It is a coverage proxy: a partially filled final domain bucket can make it larger than the exact union of bucket intervals. Coverage arithmetic uses `uhugeint_t` intermediates.

Before any reduction, it fixes a positive-count estimate `P`. For numeric keys, let `K` be the supplied build-row count, divided by estimated multiplicity when that multiplicity is greater than one. The division uses `double` and conversion to `idx_t` truncates the result. For strings, `K = 0`: full-string multiplicity does not estimate the number of distinct four-byte prefixes. With initial active count `A0` and coverage `C0`:

```text
P = max(A0, min(K, C0))

FPR(A, s) = 0                         if D <= P or C(A, s) <= P
            (C(A, s) - P) / (D - P)  otherwise
```

**`P` stays fixed throughout compression**, including when buckets merge. Recomputing it from the reduced active count would incorrectly forget positive values already accounted for. For strings, this means `P = A0` throughout the reductions.

This FPR is a model over absent values in the bounded comparable domain. It is not a measured error rate on probe rows, nor an error bound on full-string equality. Its numeric-key accuracy depends on the distinct-count estimate; treating duplicate build rows as distinct can make it optimistic. The exact source formulas are in [`CoveredValues`, `PositiveCountEstimate`, and `DomainFalsePositiveRate`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L683).

### Decision order

The initial bitmap is analyzed before any change is installed. If its FPR already exceeds threshold `T`, compression ends with that bitmap and its analysis. It does not first convert a small run count to direct ranges. In SQL execution, [`CompletePrefixRangeFilterAnalysis`](../duckdb/src/execution/join_hashtable.cpp#L920) then disables tuple filtering by this PRF; a deferred Bloom filter is built if one is available. Statistics pruning still has access to the PRF.

Otherwise, the compressor immediately converts a bitmap with **one through four complete runs** to direct ranges. For a bitmap with more runs, each candidate level is accepted in this exact order:

```text
accept(candidate):
    if candidate.FPR > T:
        return false
    if candidate has 1..4 runs with all endpoints known:
        return true
    if candidate.coverage <= current.coverage:
        return true
    return current_words * 8 > 16 KiB
       and candidate.coverage <= C0 + floor(C0 / 16)
```

Thus the FPR limit always applies. A candidate that can become direct ranges bypasses both the cache-size test and the coverage-expansion test. A candidate with unchanged coverage also bypasses those two tests. Other candidates require the **current** bitmap payload to exceed 16 KiB and may increase the coverage proxy by at most 6.25% relative to the **initial** coverage, with integer rounding down. This is a cumulative allowance, not an allowance renewed at every level. It limits relative expansion that could otherwise look inexpensive under a domain-wide FPR. The 16 KiB value is a fixed constant, not a detected CPU cache size.

On acceptance, the compressor installs the candidate's logical state and tries the next level, unless it can already finish as direct ranges. On the first rejection, it stops and keeps the previous level. It does not search beyond a rejected level for a later four-range opportunity. Empty bitmaps stay in bitmap mode. Repeated compression calls after finalization return cached analysis without compressing again. These rules are centralized in [`ShouldAcceptDyadicLevel`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L944).

## How compression is executed

All construction paths implement the same pairwise OR transformation and acceptance policy. The temporary representation is chosen to reduce construction work; it does not introduce another installed storage mode.

### Sparse builds

[`UseSparseCompression`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L1056) selects a serial path when `0 < build_rows <= 4096` and the initial bitmap has at least 4096 words, or 32 KiB of payload.

It scans the bitmap once into sorted occupied bucket IDs, skipping zero blocks of 64 words. Each reduction shifts every ID right by one and removes adjacent duplicates. Active count is the list length; consecutive IDs determine runs. Two bounded lists replace repeated bitmap reductions. If the shift changes and the result remains a bitmap, the final bitmap is allocated and populated once. If direct ranges are possible, the lists supply their endpoints. Large empty word runs for the auxiliary index are derived directly from gaps between occupied words.

### Preflight rejection

For other builds, [`TrySkipCompression`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L1060) can prove that compression should stop before allocating reduction buffers:

1. Inspect at most the first 128 complete source words. Count runs in their pairwise-OR result. Finding at least five reduced runs rules out direct ranges in both the source and the first candidate.
2. Only if that proof succeeds, count the source and first-candidate occupied buckets across the entire bitmap, masking the final partial word. Pair occupancy can be counted without packing or writing a destination bitmap.
3. Apply the initial FPR check and the normal candidate acceptance rule. If either rejects further work, retain the source bitmap and cache its analysis. Build its auxiliary index only when the initial FPR is acceptable.

This shortcut uses exact occupancy counts for its decision. Examining the prefix establishes a lower bound on the run count; it does not estimate the full bitmap's FPR. If the shortcut cannot reject compression, execution continues below.

### Bounded interval scratch representation

The general path first tries to collect occupied runs as sorted **inclusive bucket intervals** `[lower, upper]`. Adjacent runs across words or worker partitions are merged. Collection is limited to **65,536 intervals**. Each interval has two `idx_t` endpoints, so one full buffer contains 1 MiB of endpoint payload on this 64-bit build. Serial reduction uses at most two such buffers, or 2 MiB, separate from bitmap storage and container overhead.

Collection has two exact implementations. The blocked version checks blocks of 128 words: an all-zero block is skipped; an all-one block consisting of complete logical words becomes one interval. Mixed blocks and partial tails are inspected word by word, using bit scans to extract consecutive set-bit runs. The other version skips the block classification and performs the word scans directly.

A deterministic sample chooses between them. With fewer than 16 complete 128-word blocks, blocked collection is used. Otherwise, up to 16 blocks are checked at indices `(j * 2654435761 + 1013904223) % block_count`. Encountering four mixed sampled blocks selects direct word scanning. **The sample only chooses a collection algorithm.** Both collect exact intervals and use exact occupied-bucket counts.

Reduction then operates on interval endpoints:

```text
reduce_intervals(source):
    result = []
    for [lo, hi] in source:
        reduced = [lo >> 1, hi >> 1]
        append reduced, merging with the previous interval
            if they overlap or are adjacent
    active = sum(hi - lo + 1 for [lo, hi] in result)
    return result, active
```

For example, `[0,1], [3,4], [6,7]` maps to `[0,0], [1,2], [3,3]`, which merges into `[0,3]`. This is exactly the occupied-run representation of the pairwise-OR bitmap.

[`PrepareIntervalCompression`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L1259) performs successive reductions and policy decisions on these intervals. If the result is direct ranges, no replacement bitmap is written. If no level was accepted, the source bitmap is retained. Otherwise, the final bitmap is materialized once: zero the destination words, set masked boundary words, and fill fully covered interior words with all ones.

### Bitmap fallback

If interval collection exceeds its budget, it leaves the source untouched and falls back to [`AnalyzeAndReduceBitmap`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L861).

For each source word, the implementation first computes `(word | (word >> 1)) & 0x5555555555555555`, retaining one OR-result per pair. A sequence of shifts and masks packs these results into the low 32 bits. Two packed source words form one destination word. Invalid source and destination tail bits are masked to zero.

The first pass simultaneously measures the source and writes and measures the first candidate. Later passes need only measure the candidate because the current level's metrics are cached. Metrics include popcount and, while a four-range result remains possible, run starts and ends. Neighboring bits are included when classifying word boundaries. Once more than four runs have been established, full run bookkeeping can stop while occupancy counting continues. An exact run count can be recomputed later when requested by `GetCompressionInfo()`.

The reduction handles blocks of 64 destination words, corresponding to 128 source words. Homogeneous source blocks can be filled and counted as a unit. Mixed blocks are subdivided into groups of eight destination words before ordinary pair packing.

The source allocation and one initially half-sized scratch allocation are reused by swapping their roles after accepted levels. Rejection leaves the current accepted level intact. At completion, unused scratch is released and the final bitmap is copied into an allocation of exactly the required word capacity if the reused buffer is larger. A direct-range result releases bitmap storage instead.

### Parallel execution

Parallel compression is considered only for an initialized, unfinished bitmap with more than one logical bucket, a nonnegative FPR threshold, more than one available task, at least **4 MiB of initial bitmap payload**, and no sparse-build specialization. Preflight can still finish compression before tasks are created.

[`ParallelIntervalCompressionState`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L1438) partitions source words into stripes. Stripe size is at least 32,768 words and otherwise the ideal per-task size rounded up to a multiple of 128 words. Each collector receives `floor(65536 / task_count)` intervals of budget. The coordinator merges completed intervals in stripe order, coalescing adjacent boundary intervals, then performs the reduction decisions serially on the combined list. If a replacement bitmap is needed, materialization is partitioned again into disjoint word stripes. During collection/merging and reduction, at most two global interval budgets of endpoint payload coexist.

Any worker exhausting its interval budget causes the whole collection to fall back to parallel bitmap reduction. An uneven distribution can trigger this even if a serial collection would fit the global budget. No source bitmap or compression metadata has been changed at that point.

[`ParallelBitmapCompressionState`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L1543) partitions each candidate into destination stripes of at least 32,768 words, aligned to multiples of 64 words. Tasks read an immutable current bitmap and write disjoint destination words. They derive preceding/following source and candidate bits from the immutable source, so runs crossing task boundaries are counted once without reading concurrent writes. After the pass, the coordinator combines metrics and applies the same acceptance rule as serial execution.

The join's compression event waits for every task before `FinishPass()`. Once only one task is needed, subsequent work runs directly in the coordinator instead of scheduling a new event for each pass. Parallelism changes construction work and scratch distribution, not the final accepted bucket semantics.

## Compression result and lookup

| Component | Stored content | Role |
| --- | --- | --- |
| Bitmap mode | `m`, `span`, final `s`, logical bucket count, packed 64-bit words | Point membership and range classification |
| Multi-range mode | One through four `DirectRange { lower, width }` entries | Replaces the bitmap for point and range lookup |
| Auxiliary range index | Sorted `HomogeneousWordRun { begin_word, end_word, state }` entries | Accelerates long range lookups while the bitmap remains present |

### Bitmap

The final bitmap retains the original domain bounds and the final accepted shift. Bucket zero occupies bit zero of word zero. The allocation requests `64 + 8*W` bytes to provide a 64-byte-aligned word array; only `8*W` bytes hold bitmap words. Bits beyond `L` in the final word are zero. This is an ordinary packed bitmap at coarser resolution; it has no per-word compressed encoding.

Conceptually, point lookup rejects NULL and out-of-domain values, then tests the corresponding bit:

```text
y = convert(key) - m
if y > span: return false
b = y >> s
return ((bitmap[b >> 6] >> (b & 63)) & 1) != 0
```

The actual bitmap probe avoids the out-of-range branch by masking its word index to zero and AND-ing the result with the range-validity bit. Vector lookup emits the indices of surviving non-NULL rows. The auxiliary index is not involved in point probes.

### Multi-range representation

The implementation calls this mode `DIRECT_RANGES`. It uses a fixed embedded array with capacity for four entries plus `range_count`. For each accepted run of set buckets `[a,b]`, [`SetDirectRanges`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L973) computes:

```text
lower = U(m + a * 2^s)
upper = U(m + min((b + 1) * 2^s - 1, span))
width = U(upper - lower)
store { lower, width }
```

Endpoints are in the comparable value domain, and the last range is clipped at the domain maximum when necessary. Entries follow bucket order. The array reserves `8*sizeof(U)` bytes of endpoint payload: 32 bytes for 32-bit comparables, including strings, or 64 bytes for 64-bit comparables. This array is embedded in the filter object even in bitmap mode; these figures are not total object sizes. Direct-range mode releases the bitmap allocation and has no active auxiliary index.

The point-membership expression in the implementation is `(value - range.lower) <= range.width`. Scalar lookup checks the stored ranges; vector lookup dispatches to specializations for one, two, three, or four ranges and ORs their membership results.

Converting a bitmap run to a direct range preserves that run's bucket coverage exactly. The result can nevertheless include false positives introduced by an earlier shift or by string-prefix collisions. The internal name `HasExactRanges()` means that all run endpoints are known exactly; it does **not** mean exact membership in the original build-key set.

For a range query, the direct-range implementation checks intersection and continuous coverage: no overlap returns `FILTER_ALWAYS_FALSE`; complete coverage returns `FILTER_ALWAYS_TRUE`; partial overlap or a gap returns `NO_PRUNING_POSSIBLE`.

### Auxiliary range index

For a bitmap finalized with acceptable FPR, [`BuildHomogeneousWordRunIndex`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L1015) builds the index when `floor(L/64) > 2048`. It records only maximal runs of **more than 2048 complete words** that are all zero (`EMPTY`) or all one (`FULL`). Each entry is a half-open word interval `[begin_word, end_word)` and a state. Mixed words, short homogeneous runs, and the partial final word are excluded.

Construction visits anchors at word indices `0, 2048, 4096, ...`. A mixed anchor is skipped. A homogeneous anchor is expanded left and right to the entire same-state run, using 64-word comparisons followed by individual-word checks. Runs exceeding the threshold are recorded, and anchors inside the processed run are skipped. Every eligible long run contains an anchor. The index is therefore a compact directory of long empty or full spans; it does not replace bitmap words.

For the sparse-build path, gaps between occupied words directly produce long `EMPTY` entries. With at most 4096 occupied buckets, a full-word run cannot be long enough to qualify for this index.

Bitmap range lookup uses the following procedure after the type wrapper has handled domain bounds:

1. Map the inclusive lower and upper bounds to bucket positions. If both are in one word, classify just the bits under their combined mask.
2. Otherwise, classify the masked first and last words. Classification is `EMPTY` if all selected bits are zero, `FULL` if all are one, and `MIXED` otherwise. Combining unequal states produces `MIXED`; matching states remain unchanged. A mixed result can return immediately.
3. For at most 2048 complete interior words, scan them directly. For a longer interior with an index, binary-search for the last indexed run whose beginning is at or before the interior's beginning. Its state applies only if its end also covers the entire interior; otherwise return `MIXED`.
4. Combine the interior with the boundary states and map `EMPTY` to `FILTER_ALWAYS_FALSE`, `FULL` to `FILTER_ALWAYS_TRUE`, and `MIXED` to `NO_PRUNING_POSSIBLE`.

With the index built, long lookups use two boundary-word checks plus `O(log H)` index search, where `H` is the number of index entries. Short scans inspect at most 2048 interior words. An absent containing entry does not require scanning a long interval. A bitmap retained after failing the initial FPR check has no index, so its range lookups scan the interior words directly.

For example, an index entry `{100, 5000, EMPTY}` can classify an interior word interval `[200, 4000)` as empty. It cannot classify `[200, 5100)` as homogeneous; that lookup remains mixed. Both queries still classify their masked boundary words separately.

The index's reported memory is `vector.capacity() * sizeof(HomogeneousWordRun)`, including struct padding, rather than a packed endpoint-bit encoding. Final bitmap bytes and auxiliary-index bytes are reported separately by `GetCompressionInfo()`.

`FILTER_ALWAYS_TRUE` reports full PRF coverage of the comparable interval that was tested. It does not prove a join match. Numeric and string range wrappers check domain overlap, and string statistics bounds use conservative prefix padding when maxima are truncated. These wrappers and ordinary join evaluation remain necessary around the compressed representation.

## Worked examples

### Lossless bucket reduction that retains a bitmap

Start at `s = 0`, domain `[0,17]`, with keys `{0,1,4,5,8,9,12,13,16,17}`. Bit sequences below are written in increasing bucket order, with bucket zero on the left:

```text
s = 0: 11 00 11 00 11 00 11 00 11   A = 10, R = 5, C = 10
s = 1:  1  0  1  0  1  0  1  0  1   A =  5, R = 5, C = 10
s = 2:     1     1     1     1     1   A =  5, R = 1, C = 18
```

With ten distinct positives, `P = 10`. The first reduction has FPR zero and is accepted even though the original bitmap is already tiny. The next candidate has FPR `(18-10)/(18-10) = 1`; at threshold `0.1`, it is rejected despite its single run. Final storage is the shift-one bitmap. Logical buckets shrink from 18 to 9, although both fit in one physical word. This is covered by the test “Prefix range filter dyadic analysis preserves original positive lower bound.”

### Direct ranges without bucket widening

For keys `[0,9] ∪ [100,109] ∪ [300,309] ∪ [700,709]` at shift zero, there are four occupied runs. Compression immediately stores:

```text
{lower:   0, width: 9}
{lower: 100, width: 9}
{lower: 300, width: 9}
{lower: 700, width: 9}
```

No bucket reduction is needed. Bitmap allocation becomes zero; a probe such as 105 passes and 250 fails. This is covered by “Prefix range filter direct compression preserves four sparse value ranges.”

### Cumulative expansion stops a larger bitmap

Consider six clusters of 33 consecutive keys each, beginning at `c*524288` for `c = 0..5`, with domain `[0, 2^22]`, initial shift zero, and threshold `0.001`. Initially `P = C0 = 198`. The expansion ceiling is `198 + floor(198/16) = 210`.

| Candidate shift | Occupied buckets | Coverage proxy | Policy outcome |
| --- | ---: | ---: | --- |
| 0 | 198 | 198 | Initial state |
| 1 | 102 | 204 | Accept: large current bitmap and coverage at most 210 |
| 2 | 54 | 216 | Reject: cumulative coverage exceeds 210 |

Both candidates have six runs and FPR below `0.001`. The second is rejected by the cumulative coverage limit. Final buckets have width two, so each cluster's adjacent value at offset 33 also passes, while offset 34 fails. This example is the test “Prefix range sparse and dense compression bound cumulative expansion.”

### A later direct-range result can be approximate

For `{0,1,3,4,6,7,9,10,12,13}` in `[0,13]`, the shift-zero bitmap has five runs. At shift one, every bucket is occupied, giving one direct range `[0,13]`. The FPR is `1.0`, so this conversion is accepted only with a threshold allowing that value; the regression test explicitly uses `T = 1.0`. The ordinary SQL threshold `0.001` rejects it. This illustrates why exact run endpoints and exact key membership are different properties.

## Source and regression references

The design's main source locations are:

| Subject | Implementation |
| --- | --- |
| Public analysis, compression, and result metadata | [`PrefixRangeFilter`](../duckdb/src/include/duckdb/planner/filter/table_filter_functions.hpp#L181) |
| Policy and storage installation | [`ShouldAcceptDyadicLevel` through `BuildHomogeneousWordRunIndex`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L944) |
| Sparse, preflight, and interval paths | [`UseSparseCompression` and following methods](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L1056) |
| Interval collection, reduction, and materialization | [`PrefixRangeIntervals`](../duckdb/src/include/duckdb/planner/filter/prefix_range_intervals.hpp) |
| Parallel coordination | [`ParallelIntervalCompressionState` and `ParallelBitmapCompressionState`](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L1438) |
| Point and range lookup | [`PrefixRangeBitmap` lookup methods](../duckdb/src/planner/filter/table_filter_prefix_range_function.cpp#L153) |
| SQL sizing and post-build analysis | [`physical_hash_join.cpp`](../duckdb/src/execution/operator/join/physical_hash_join.cpp#L1559), [`join_hashtable.cpp`](../duckdb/src/execution/join_hashtable.cpp#L874) |

[`test/optimizer/prefix_range_filter_compression.cpp`](../duckdb/test/optimizer/prefix_range_filter_compression.cpp) covers the worked examples, multiplicity estimates, policy ordering, final allocation size, long homogeneous runs, sparse/dense parity, parallel boundaries, interval-budget overflow, tail masking, and interval materialization against independent dyadic reduction. [`test_prefix_range_filter_parallel_compression.test`](../duckdb/test/sql/join/inner/test_prefix_range_filter_parallel_compression.test) exercises compression through SQL with one and eight threads.
