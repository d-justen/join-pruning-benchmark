// Compile against each vendored engine. Explicit multiplicity estimates isolate
// compression policy from version-dependent source statistics and RPT sampling.
#include "duckdb.hpp"
#ifdef DUCKDB_RPT
#include "duckdb/optimizer/predicate_transfer/bloom_filter/prefix_range_filter.hpp"
#else
#include "duckdb/planner/filter/table_filter_functions.hpp"
#endif
#include <iomanip>
#include <iostream>

using namespace duckdb;

static void Check(ClientContext &context, const string &name,
                  const vector<int64_t> &keys, int64_t maximum,
                  double multiplicity, idx_t threads) {
#ifdef DUCKDB_RPT
  PrefixRangeFilter filter(LogicalType::BIGINT, multiplicity);
  filter.Initialize(context, keys.size(), Value::BIGINT(0),
                    Value::BIGINT(maximum));
  filter.SetSingleWriter();
#else
  auto owned = PrefixRangeFilter::CreatePrefixRangeFilter(LogicalType::BIGINT);
  auto &filter = *owned;
  PrefixRangeFilter::Sizing sizing;
  if (!PrefixRangeFilter::TryComputeFixedSizeSizing(
          Value::BIGINT(0), Value::BIGINT(maximum), idx_t(1) << 26, sizing)) {
    throw std::runtime_error("Cannot size fixture");
  }
  filter.Initialize(context, keys.size(), Value::BIGINT(0),
                    Value::BIGINT(maximum), sizing, multiplicity);
  auto state = filter.InitializeBuildState(context);
#endif
  DataChunk chunk;
  chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT});
  auto insert = [&]() {
    for (idx_t begin = 0; begin < keys.size(); begin += STANDARD_VECTOR_SIZE) {
      const auto count =
          MinValue<idx_t>(STANDARD_VECTOR_SIZE, keys.size() - begin);
      for (idx_t i = 0; i < count; i++)
        chunk.data[0].SetValue(i, Value::BIGINT(keys[begin + i]));
      chunk.SetCardinality(count);
#ifdef DUCKDB_RPT
      filter.Insert(chunk, {0});
#else
			FlatVector::SetSize(chunk.data[0], count_t(count));
			filter.InsertKeys(chunk.data[0], count, *state);
#endif
    }
  };
  insert();
#ifdef DUCKDB_RPT
  filter.Finalize(context);
  auto info = filter.GetDiagnostics();
  const auto mode = info["final_mode"] == "direct_ranges" ? 1 : 0;
  const auto shift = std::stoull(info["final_shift"]);
  const auto ranges = std::stoull(info["range_count"]);
  const auto active = std::stoull(info["active_buckets"]);
  const auto fpr = std::stod(info["false_positive_rate"]);
  const auto fallback = filter.NeedsBloomFallback();
  if (fallback) {
    insert();
    filter.CompleteBloomFallback();
  }
#else
  filter.MergeBuildState(*state);
  auto parallel =
      threads > 1
          ? filter.InitializeParallelCompression(context, 0.001, threads)
          : nullptr;
  if (parallel) {
    do {
      for (idx_t task = 0; task < parallel->TaskCount(); task++)
        parallel->ExecuteTask(task);
    } while (parallel->FinishPass());
  } else {
    filter.Compress(context, 0.001);
  }
  const auto info = filter.GetCompressionInfo();
  const auto mode = info.mode == CompressionMode::DIRECT_RANGES ? 1 : 0;
  const auto shift = info.shift, ranges = info.range_count,
             active = info.active_buckets;
  const auto fpr = info.false_positive_rate;
  const auto fallback = fpr > 0.001;
#endif
  auto contains = [&](int64_t key) {
    chunk.data[0].SetValue(0, Value::BIGINT(key));
    chunk.SetCardinality(1);
#ifdef DUCKDB_RPT
    vector<uint32_t> result(STANDARD_VECTOR_SIZE);
    filter.Lookup(chunk, result, {0});
    return result[0] != 0;
#else
		FlatVector::SetSize(chunk.data[0], count_t(1));
		SelectionVector result(1);
    return filter.LookupKeys(chunk.data[0], result, 1) == 1;
#endif
  };
  for (auto key : keys) {
    if (!contains(key))
      throw std::runtime_error("False negative: " + name);
  }
  uint64_t digest = 0;
  // Bloom hashing can differ across engine versions. Compare PRF membership.
  if (!fallback) {
    for (int64_t key = -1; key <= maximum + 1;
         key += MaxValue<int64_t>(1, maximum / 10000)) {
      digest = digest * 31 + contains(key);
    }
  }
  std::cout << name << '\t' << threads << '\t' << mode << '\t' << shift << '\t'
            << ranges << '\t' << active << '\t' << std::setprecision(17) << fpr
            << '\t' << fallback << '\t' << digest << '\n';
}

int main() {
  DuckDB db(nullptr);
  Connection con(db);
  for (idx_t threads : {idx_t(1), idx_t(4)}) {
    if (con.Query("SET threads=" + std::to_string(threads))->HasError())
      return 1;
    for (idx_t rows : {idx_t(10000), idx_t(100000), idx_t(300000)}) {
      for (idx_t multiplicity : {idx_t(1), idx_t(8)}) {
        vector<int64_t> keys;
        const auto distinct = rows / multiplicity;
        for (idx_t r = 0; r < rows; r++) {
          const auto rank = r / multiplicity;
          keys.push_back(rank < distinct / 2 ? rank
                                             : 100000000 - distinct + rank);
        }
        Check(*con.context,
              "shoulder_" + std::to_string(rows) + "_" +
                  std::to_string(multiplicity),
              keys, 99999999, multiplicity, threads);
      }
    }
    vector<int64_t> sparse;
    for (idx_t r = 0; r < 150000; r++)
      sparse.push_back((r % 100000) * 700);
    Check(*con.context, "fractional_multiplicity", sparse, 99999 * 700, 1.5,
          threads);
    Check(*con.context, "missing_statistics", sparse, 99999 * 700, 1.0,
          threads);
    const int64_t end = (1 << 26) - 59;
    Check(*con.context, "parallel_boundaries",
          {0, 1, 2, 16777214, 16777215, 16777216, 16777217, 33554431, end}, end,
          1.0, threads);
    Check(*con.context, "parallel_ranges",
          {0, 16777214, 16777215, 16777216, 16777217, end}, end, 1.0, threads);
    vector<int64_t> bitmap;
    for (int64_t key = 0; key < 262147; key += 4096)
      bitmap.push_back(key);
    bitmap.push_back(262146);
    Check(*con.context, "cache_bitmap", bitmap, 262146, 1.0, threads);
  }
}
