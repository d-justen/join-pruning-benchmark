//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/predicate_transfer/bloom_filter/prefix_range_filter.hpp
//
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/common/allocator.hpp"
#include "duckdb/optimizer/predicate_transfer/bloom_filter/runtime_filter.hpp"

#include <array>

namespace duckdb {

//! Integer Prefix Range Filter used by RPT+.
//!
//! Values are mapped to an order-preserving unsigned domain and then to an
//! equi-width bitmap. Finalize can coarsen adjacent buckets without exceeding
//! the configured false-positive-rate budget, or replace up to four bitmap
//! runs with direct ranges.
class PrefixRangeFilter : public RuntimeFilter {
public:
	static constexpr idx_t MAX_BUCKET_COUNT = idx_t(1) << 26;
	static constexpr idx_t CACHE_TARGET_BYTES = 16 * 1024;
	static constexpr double MAX_COMPRESSION_FALSE_POSITIVE_RATE = 0.001;
	static constexpr idx_t MAX_DIRECT_RANGES = 4;

	explicit PrefixRangeFilter(LogicalType key_type_p);

	static bool SupportedType(const LogicalType &type);

	void Initialize(ClientContext &context, idx_t row_count, const Value &min, const Value &max) override;
	void Insert(DataChunk &chunk, const vector<idx_t> &bound_cols_built) override;
	int Lookup(DataChunk &chunk, vector<uint32_t> &results, const vector<idx_t> &bound_cols_applied) const override;
	idx_t Select(DataChunk &chunk, SelectionVector &sel, vector<uint32_t> &scratch,
	             const vector<idx_t> &bound_cols_applied) const override;
	void Finalize(ClientContext &context) override;
	//! Refine a scan selection, preserving positions in the original vector.
	idx_t SelectScan(Vector &keys, UnifiedVectorFormat &data, const SelectionVector &input,
	                 idx_t count, SelectionVector &result) const;
	//! Construction tasks either exclusively own the global bitmap or a private bitmap.
	void SetSingleWriter() {
		single_writer = true;
	}
	unique_ptr<PrefixRangeFilter> CreateLocal(ClientContext &context) const;
	void MergeLocal(const PrefixRangeFilter &local);
	string GetBuildTimings() const;

	bool IsValid() const override {
		return finalized;
	}
	bool RequiresBounds() const override {
		return true;
	}
	RuntimeFilterType GetType() const override {
		return RuntimeFilterType::PREFIX_RANGE;
	}
	string GetName() const override {
		return "Prefix Range";
	}
	string GetInfo() const override;
	idx_t GetMemoryUsage() const override;
	FilterPropagateResult CheckStatistics(BaseStatistics &stats) const override;

private:
	enum class Mode { BITMAP, DIRECT_RANGES };
	struct DirectRange {
		uint64_t lower;
		uint64_t upper;
	};

	void AllocateBitmap(ClientContext &context, idx_t buckets);
	void Compress(ClientContext &context);
	void BuildLookupIndex();
	struct BitmapMetrics {
		idx_t active = 0;
		idx_t runs = 0;
		idx_t ended = 0;
		uint64_t previous = 0;
		std::array<idx_t, MAX_DIRECT_RANGES> starts;
		std::array<idx_t, MAX_DIRECT_RANGES> ends;
		void Add(uint64_t word, idx_t base, idx_t valid_bits);
		void Finish(idx_t buckets);
	};
	void ConvertToDirectRanges(const BitmapMetrics &metrics);
	double FalsePositiveRate(idx_t active_buckets, idx_t candidate_shift) const;
	bool LookupComparable(uint64_t comparable) const;
	FilterPropagateResult LookupRange(uint64_t lower, uint64_t upper) const;

	template <class T>
	static uint64_t ToComparable(T value);
	static bool ValueToComparable(const Value &value, uint64_t &result);
	template <class T, bool SINGLE_WRITER>
	void InsertVector(Vector &keys, idx_t count);
	template <bool SINGLE_WRITER>
	void InsertInternal(DataChunk &chunk, const vector<idx_t> &cols);
	template <class T>
	void LookupVector(Vector &keys, idx_t count, vector<uint32_t> &results) const;
	template <class T>
	idx_t SelectVector(Vector &keys, idx_t count, SelectionVector &sel) const;
	template <class T, idx_t RANGE_COUNT>
	idx_t SelectVectorMode(UnifiedVectorFormat &data, bool flat, idx_t count, SelectionVector &sel) const;
	template <class T, idx_t RANGE_COUNT, bool FLAT, bool ALL_VALID>
	idx_t SelectLoop(UnifiedVectorFormat &data, idx_t count, SelectionVector &sel,
	                 const SelectionVector *input = nullptr) const;
	template <class T>
	idx_t SelectScanType(Vector &keys, UnifiedVectorFormat &data, const SelectionVector &input,
	                    idx_t count, SelectionVector &result) const;

private:
	LogicalType key_type;
	Mode mode = Mode::BITMAP;
	bool finalized = false;
	bool empty = false;
	uint64_t minimum = 0;
	uint64_t maximum = 0;
	uint64_t span = 0;
	idx_t shift = 0;
	idx_t logical_bucket_count = 0;
	idx_t word_count = 0;
	idx_t base_active_bucket_count = 0;
	idx_t active_bucket_count = 0;
	double false_positive_rate = 0;
	AllocatedData bitmap_data;
	uint64_t *bitmap = nullptr;
	std::array<DirectRange, MAX_DIRECT_RANGES> direct_ranges;
	idx_t direct_range_count = 0;
	struct HomogeneousRun {
		idx_t begin;
		idx_t end;
		bool full;
	};
	vector<HomogeneousRun> homogeneous_runs;
	bool single_writer = false;
	double initialize_ms = 0, insert_ms = 0, merge_ms = 0, compress_ms = 0, index_ms = 0;
};

} // namespace duckdb
