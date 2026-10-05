#include "duckdb/optimizer/predicate_transfer/bloom_filter/prefix_range_filter.hpp"

#include "duckdb/common/bit_utils.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <type_traits>

namespace duckdb {

namespace {

static uint64_t ValidBitsMask(idx_t valid_bits) {
	if (valid_bits >= 64) {
		return ~uint64_t(0);
	}
	if (valid_bits == 0) {
		return 0;
	}
	return (uint64_t(1) << valid_bits) - 1;
}

using BuildClock = std::chrono::steady_clock;
static double ElapsedMS(BuildClock::time_point start) {
	return std::chrono::duration<double, std::milli>(BuildClock::now() - start).count();
}

static idx_t PopCount(uint64_t word) {
#if defined(__GNUC__) || defined(__clang__)
	return __builtin_popcountll(word);
#else
	word -= (word >> 1) & 0x5555555555555555ULL;
	word = (word & 0x3333333333333333ULL) + ((word >> 2) & 0x3333333333333333ULL);
	word = (word + (word >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
	return (word * 0x0101010101010101ULL) >> 56;
#endif
}

static uint64_t PackMergedPairs(uint64_t word) {
	word = (word | (word >> 1)) & 0x5555555555555555ULL;
	word = (word | (word >> 1)) & 0x3333333333333333ULL;
	word = (word | (word >> 2)) & 0x0f0f0f0f0f0f0f0fULL;
	word = (word | (word >> 4)) & 0x00ff00ff00ff00ffULL;
	word = (word | (word >> 8)) & 0x0000ffff0000ffffULL;
	return (word | (word >> 16)) & 0xffffffffULL;
}

} // namespace

PrefixRangeFilter::PrefixRangeFilter(LogicalType key_type_p, double estimated_multiplicity_p)
    : key_type(std::move(key_type_p)), estimated_multiplicity(estimated_multiplicity_p) {
}

bool PrefixRangeFilter::SupportedType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
		return true;
	default:
		return false;
	}
}

template <class T>
uint64_t PrefixRangeFilter::ToComparable(T value) {
	using U = typename std::make_unsigned<T>::type;
	auto comparable = static_cast<U>(value);
	if (std::is_signed<T>::value) {
		comparable ^= U(1) << (sizeof(T) * 8 - 1);
	}
	return static_cast<uint64_t>(comparable);
}

bool PrefixRangeFilter::ValueToComparable(const Value &value, uint64_t &result) {
	if (value.IsNull()) {
		return false;
	}
	switch (value.type().id()) {
	case LogicalTypeId::TINYINT:
		result = ToComparable<int8_t>(value.GetValueUnsafe<int8_t>());
		return true;
	case LogicalTypeId::SMALLINT:
		result = ToComparable<int16_t>(value.GetValueUnsafe<int16_t>());
		return true;
	case LogicalTypeId::INTEGER:
		result = ToComparable<int32_t>(value.GetValueUnsafe<int32_t>());
		return true;
	case LogicalTypeId::BIGINT:
		result = ToComparable<int64_t>(value.GetValueUnsafe<int64_t>());
		return true;
	case LogicalTypeId::UTINYINT:
		result = ToComparable<uint8_t>(value.GetValueUnsafe<uint8_t>());
		return true;
	case LogicalTypeId::USMALLINT:
		result = ToComparable<uint16_t>(value.GetValueUnsafe<uint16_t>());
		return true;
	case LogicalTypeId::UINTEGER:
		result = ToComparable<uint32_t>(value.GetValueUnsafe<uint32_t>());
		return true;
	case LogicalTypeId::UBIGINT:
		result = ToComparable<uint64_t>(value.GetValueUnsafe<uint64_t>());
		return true;
	default:
		return false;
	}
}

void PrefixRangeFilter::AllocateBitmap(ClientContext &context, idx_t buckets) {
	logical_bucket_count = buckets;
	word_count = MaxValue<idx_t>((buckets + 63) / 64, 1);
	auto &allocator = BufferManager::GetBufferManager(context).GetBufferAllocator();
	bitmap_data = allocator.Allocate(64 + word_count * sizeof(uint64_t));
	bitmap = reinterpret_cast<uint64_t *>((reinterpret_cast<uintptr_t>(bitmap_data.get()) + 63) & ~uintptr_t(63));
	std::fill_n(bitmap, word_count, 0);
}

void PrefixRangeFilter::Initialize(ClientContext &context, idx_t row_count, const Value &min, const Value &max) {
	const auto start = BuildClock::now();
	finalized = false;
	mode = Mode::BITMAP;
	bitmap_data.Reset();
	bitmap = nullptr;
	word_count = logical_bucket_count = direct_range_count = 0;
	active_bucket_count = 0;
	build_row_count = row_count;
	positive_count_estimate = 0;
	initial_shift = 0;
	shift = 0;
	use_bloom = bloom_ready = false;
	bloom_data.Reset();
	bloom = nullptr;
	false_positive_rate = 0;
	homogeneous_runs.clear();
	initialize_ms = insert_ms = merge_ms = compress_ms = index_ms = 0;
	single_writer = false;
	empty = row_count == 0 || min.IsNull() || max.IsNull();
	if (empty) {
		return;
	}
	if (min.type() != key_type || max.type() != key_type || !ValueToComparable(min, minimum) ||
	    !ValueToComparable(max, maximum) || minimum > maximum) {
		throw InternalException("Invalid bounds for RPT+ prefix range filter of type " + key_type.ToString());
	}

	span = maximum - minimum;
	shift = 0;
	// This is the fixed-size sizing rule from the PRF implementation. Small
	// domains are exact; large domains use the smallest power-of-two bucket
	// width that stays below the 2^26 bucket cap.
	while ((span >> shift) >= MAX_BUCKET_COUNT) {
		shift++;
	}
	const auto buckets = static_cast<idx_t>((span >> shift) + 1);
	initial_shift = shift;
	AllocateBitmap(context, buckets);
	initialize_ms = ElapsedMS(start);
}

template <class T, bool SINGLE_WRITER>
void PrefixRangeFilter::InsertVector(Vector &keys, idx_t count) {
	UnifiedVectorFormat data;
	keys.ToUnifiedFormat(count, data);
	auto values = UnifiedVectorFormat::GetData<T>(data);
	for (idx_t row_idx = 0; row_idx < count; row_idx++) {
		const auto source_idx = data.sel->get_index(row_idx);
		if (!data.validity.RowIsValid(source_idx)) {
			continue;
		}
		const auto comparable = ToComparable<T>(values[source_idx]);
		if (comparable < minimum || comparable > maximum) {
			continue;
		}
		if (use_bloom) {
			InsertBloomHash(Hash<T>(values[source_idx]));
			continue;
		}
		const auto bucket = static_cast<idx_t>((comparable - minimum) >> shift);
		if (SINGLE_WRITER) {
			bitmap[bucket / 64] |= uint64_t(1) << (bucket % 64);
		} else {
			auto &word = *reinterpret_cast<std::atomic<uint64_t> *>(&bitmap[bucket / 64]);
			word.fetch_or(uint64_t(1) << (bucket % 64), std::memory_order_relaxed);
		}
	}
}

void PrefixRangeFilter::Insert(DataChunk &chunk, const vector<idx_t> &bound_cols_built) {
	if (use_bloom) {
		InsertInternal<false>(chunk, bound_cols_built);
		return;
	}
	if (single_writer) {
		const auto start = BuildClock::now();
		InsertInternal<true>(chunk, bound_cols_built);
		insert_ms += ElapsedMS(start);
	} else {
		// Public concurrent insertion remains supported; do not update shared timing counters.
		InsertInternal<false>(chunk, bound_cols_built);
	}
}

template <bool SINGLE_WRITER>
void PrefixRangeFilter::InsertInternal(DataChunk &chunk, const vector<idx_t> &bound_cols_built) {
	if (empty) {
		return;
	}
	D_ASSERT(bound_cols_built.size() == 1);
	auto &keys = chunk.data[bound_cols_built[0]];
	switch (key_type.id()) {
	case LogicalTypeId::TINYINT:
		return InsertVector<int8_t, SINGLE_WRITER>(keys, chunk.size());
	case LogicalTypeId::SMALLINT:
		return InsertVector<int16_t, SINGLE_WRITER>(keys, chunk.size());
	case LogicalTypeId::INTEGER:
		return InsertVector<int32_t, SINGLE_WRITER>(keys, chunk.size());
	case LogicalTypeId::BIGINT:
		return InsertVector<int64_t, SINGLE_WRITER>(keys, chunk.size());
	case LogicalTypeId::UTINYINT:
		return InsertVector<uint8_t, SINGLE_WRITER>(keys, chunk.size());
	case LogicalTypeId::USMALLINT:
		return InsertVector<uint16_t, SINGLE_WRITER>(keys, chunk.size());
	case LogicalTypeId::UINTEGER:
		return InsertVector<uint32_t, SINGLE_WRITER>(keys, chunk.size());
	case LogicalTypeId::UBIGINT:
		return InsertVector<uint64_t, SINGLE_WRITER>(keys, chunk.size());
	default:
		throw InternalException("Unsupported RPT+ prefix range filter type " + key_type.ToString());
	}
}

unique_ptr<PrefixRangeFilter> PrefixRangeFilter::CreateLocal(ClientContext &context) const {
	auto start = BuildClock::now();
	auto local = make_uniq<PrefixRangeFilter>(key_type);
	local->empty = empty;
	local->minimum = minimum;
	local->maximum = maximum;
	local->span = span;
	local->shift = shift;
	local->single_writer = true;
	if (!empty) {
		local->AllocateBitmap(context, logical_bucket_count);
	}
	local->initialize_ms = ElapsedMS(start);
	return local;
}

void PrefixRangeFilter::MergeLocal(const PrefixRangeFilter &local) {
	const auto start = BuildClock::now();
	D_ASSERT(local.word_count == word_count && local.shift == shift && !local.finalized);
	for (idx_t i = 0; i < word_count; i++) {
		bitmap[i] |= local.bitmap[i];
	}
	initialize_ms += local.initialize_ms;
	insert_ms += local.insert_ms;
	merge_ms += ElapsedMS(start);
}

bool PrefixRangeFilter::LookupComparable(uint64_t comparable) const {
	if (empty || comparable < minimum || comparable > maximum) {
		return false;
	}
	if (mode == Mode::DIRECT_RANGES) {
		for (idx_t range_idx = 0; range_idx < direct_range_count; range_idx++) {
			if (comparable >= direct_ranges[range_idx].lower && comparable <= direct_ranges[range_idx].upper) {
				return true;
			}
		}
		return false;
	}
	const auto bucket = static_cast<idx_t>((comparable - minimum) >> shift);
	return (bitmap[bucket / 64] >> (bucket % 64)) & 1;
}

// Same 64-bit sector Bloom filter as the regular join-filter fallback. Cascade
// remains a separate RPT variant and is not substituted for this fallback.
static uint64_t PRFBloomMask(hash_t hash) {
	uint64_t mask = 0;
	for (idx_t byte = 4; byte < 8; byte++) {
		mask |= uint64_t(1) << ((hash >> (byte * 8)) & 63);
	}
	return mask;
}

void PrefixRangeFilter::InitializeBloom(ClientContext &context) {
	const auto sectors =
	    MinValue<idx_t>(NextPowerOfTwo(MaxValue<idx_t>(512, build_row_count * 12)) >> 6, idx_t(1) << 26);
	bloom_mask = sectors - 1;
	bloom_data =
	    BufferManager::GetBufferManager(context).GetBufferAllocator().Allocate(64 + sectors * sizeof(uint64_t));
	bloom = reinterpret_cast<uint64_t *>((reinterpret_cast<uintptr_t>(bloom_data.get()) + 63) & ~uintptr_t(63));
	std::fill_n(bloom, sectors, 0);
}

void PrefixRangeFilter::InsertBloomHash(hash_t hash) {
	auto &slot = *reinterpret_cast<std::atomic<uint64_t> *>(&bloom[hash & bloom_mask]);
	slot.fetch_or(PRFBloomMask(hash), std::memory_order_relaxed);
}

bool PrefixRangeFilter::LookupBloomHash(hash_t hash) const {
	const auto mask = PRFBloomMask(hash);
	return (bloom[hash & bloom_mask] & mask) == mask;
}

template <class T>
bool PrefixRangeFilter::LookupValue(T value) const {
	const auto comparable = ToComparable<T>(value);
	return use_bloom ? (!empty && comparable >= minimum && comparable <= maximum && LookupBloomHash(Hash<T>(value)))
	                 : LookupComparable(comparable);
}

template <class T>
void PrefixRangeFilter::LookupVector(Vector &keys, idx_t count, vector<uint32_t> &results) const {
	UnifiedVectorFormat data;
	keys.ToUnifiedFormat(count, data);
	auto values = UnifiedVectorFormat::GetData<T>(data);
	for (idx_t row_idx = 0; row_idx < count; row_idx++) {
		const auto source_idx = data.sel->get_index(row_idx);
		results[row_idx] = data.validity.RowIsValid(source_idx) && LookupValue<T>(values[source_idx]);
	}
}

int PrefixRangeFilter::Lookup(DataChunk &chunk, vector<uint32_t> &results,
                              const vector<idx_t> &bound_cols_applied) const {
	const auto count = chunk.size();
	if (!IsValid()) {
		std::fill_n(results.begin(), count, 1);
		return static_cast<int>(count);
	}
	if (empty) {
		std::fill_n(results.begin(), count, 0);
		return static_cast<int>(count);
	}
	D_ASSERT(bound_cols_applied.size() == 1);
	auto &keys = chunk.data[bound_cols_applied[0]];
	switch (key_type.id()) {
	case LogicalTypeId::TINYINT:
		LookupVector<int8_t>(keys, count, results);
		break;
	case LogicalTypeId::SMALLINT:
		LookupVector<int16_t>(keys, count, results);
		break;
	case LogicalTypeId::INTEGER:
		LookupVector<int32_t>(keys, count, results);
		break;
	case LogicalTypeId::BIGINT:
		LookupVector<int64_t>(keys, count, results);
		break;
	case LogicalTypeId::UTINYINT:
		LookupVector<uint8_t>(keys, count, results);
		break;
	case LogicalTypeId::USMALLINT:
		LookupVector<uint16_t>(keys, count, results);
		break;
	case LogicalTypeId::UINTEGER:
		LookupVector<uint32_t>(keys, count, results);
		break;
	case LogicalTypeId::UBIGINT:
		LookupVector<uint64_t>(keys, count, results);
		break;
	default:
		throw InternalException("Unsupported RPT+ prefix range filter type " + key_type.ToString());
	}
	return static_cast<int>(count);
}

template <class T, idx_t RANGE_COUNT, bool FLAT, bool ALL_VALID>
idx_t PrefixRangeFilter::SelectLoop(UnifiedVectorFormat &data, idx_t count, SelectionVector &sel,
                                  const SelectionVector *input) const {
	const auto values = UnifiedVectorFormat::GetData<T>(data);
	idx_t selected_count = 0;
	for (idx_t row_idx = 0; row_idx < count; row_idx++) {
		const auto position = input ? input->get_index(row_idx) : row_idx;
		const auto source_idx = FLAT ? position : data.sel->get_index(position);
		if (!ALL_VALID && !data.validity.RowIsValid(source_idx)) {
			continue;
		}
		const auto comparable = ToComparable<T>(values[source_idx]);
		uint64_t matches = 0;
		if (use_bloom) {
			matches = comparable >= minimum && comparable <= maximum && LookupBloomHash(Hash<T>(values[source_idx]));
		} else if (RANGE_COUNT == 0) {
			const auto offset = comparable - minimum;
			const uint64_t in_range = offset <= span;
			const auto bucket = offset >> shift;
			// Out-of-domain values still perform a safe load from word zero.
			const auto word_idx = (bucket >> 6) & (uint64_t(0) - in_range);
			matches = ((bitmap[word_idx] >> (bucket & 63)) & 1) & in_range;
		} else {
			// RANGE_COUNT is fixed at compile time (1..4), allowing unrolling.
			// Unsigned subtraction handles both bounds without short-circuit branches.
			for (idx_t i = 0; i < RANGE_COUNT; i++) {
				matches |= (comparable - direct_ranges[i].lower) <= (direct_ranges[i].upper - direct_ranges[i].lower);
			}
		}
		sel.set_index(selected_count, position);
		selected_count += matches;
	}
	return selected_count;
}

template <class T, idx_t RANGE_COUNT>
idx_t PrefixRangeFilter::SelectVectorMode(UnifiedVectorFormat &data, bool flat, idx_t count,
                                          SelectionVector &sel) const {
	if (flat) {
		if (data.validity.AllValid()) {
			return SelectLoop<T, RANGE_COUNT, true, true>(data, count, sel);
		}
		return SelectLoop<T, RANGE_COUNT, true, false>(data, count, sel);
	}
	if (data.validity.AllValid()) {
		return SelectLoop<T, RANGE_COUNT, false, true>(data, count, sel);
	}
	return SelectLoop<T, RANGE_COUNT, false, false>(data, count, sel);
}

template <class T>
idx_t PrefixRangeFilter::SelectVector(Vector &keys, idx_t count, SelectionVector &sel) const {
	if (keys.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(keys) || !LookupValue<T>(*ConstantVector::GetData<T>(keys))) {
			return 0;
		}
		for (idx_t i = 0; i < count; i++) {
			sel.set_index(i, i);
		}
		return count;
	}
	const bool flat = keys.GetVectorType() == VectorType::FLAT_VECTOR;
	UnifiedVectorFormat data;
	keys.ToUnifiedFormat(count, data);
	if (mode == Mode::BITMAP) {
		return SelectVectorMode<T, 0>(data, flat, count, sel);
	}
	switch (direct_range_count) {
	case 1:
		return SelectVectorMode<T, 1>(data, flat, count, sel);
	case 2:
		return SelectVectorMode<T, 2>(data, flat, count, sel);
	case 3:
		return SelectVectorMode<T, 3>(data, flat, count, sel);
	case 4:
		return SelectVectorMode<T, 4>(data, flat, count, sel);
	default:
		throw InternalException("Invalid RPT+ prefix range count");
	}
}

template <class T>
idx_t PrefixRangeFilter::SelectScanType(Vector &keys, UnifiedVectorFormat &data, const SelectionVector &input,
                                      idx_t count, SelectionVector &result) const {
	if (keys.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (!data.validity.RowIsValid(0) || !LookupValue<T>(*UnifiedVectorFormat::GetData<T>(data))) {
			return 0;
		}
		for (idx_t i = 0; i < count; i++) {
			result.set_index(i, input.get_index(i));
		}
		return count;
	}
	const bool flat = keys.GetVectorType() == VectorType::FLAT_VECTOR;
	const bool valid = data.validity.AllValid();
	switch (mode == Mode::BITMAP ? 0 : direct_range_count) {
	case 0:
		if (flat) {
			return valid ? SelectLoop<T, 0, true, true>(data, count, result, &input)
			             : SelectLoop<T, 0, true, false>(data, count, result, &input);
		}
		return valid ? SelectLoop<T, 0, false, true>(data, count, result, &input)
		             : SelectLoop<T, 0, false, false>(data, count, result, &input);
	case 1:
		if (flat) {
			return valid ? SelectLoop<T, 1, true, true>(data, count, result, &input)
			             : SelectLoop<T, 1, true, false>(data, count, result, &input);
		}
		return valid ? SelectLoop<T, 1, false, true>(data, count, result, &input)
		             : SelectLoop<T, 1, false, false>(data, count, result, &input);
	case 2:
		if (flat) {
			return valid ? SelectLoop<T, 2, true, true>(data, count, result, &input)
			             : SelectLoop<T, 2, true, false>(data, count, result, &input);
		}
		return valid ? SelectLoop<T, 2, false, true>(data, count, result, &input)
		             : SelectLoop<T, 2, false, false>(data, count, result, &input);
	case 3:
		if (flat) {
			return valid ? SelectLoop<T, 3, true, true>(data, count, result, &input)
			             : SelectLoop<T, 3, true, false>(data, count, result, &input);
		}
		return valid ? SelectLoop<T, 3, false, true>(data, count, result, &input)
		             : SelectLoop<T, 3, false, false>(data, count, result, &input);
	case 4:
		if (flat) {
			return valid ? SelectLoop<T, 4, true, true>(data, count, result, &input)
			             : SelectLoop<T, 4, true, false>(data, count, result, &input);
		}
		return valid ? SelectLoop<T, 4, false, true>(data, count, result, &input)
		             : SelectLoop<T, 4, false, false>(data, count, result, &input);
	default: throw InternalException("Invalid PRF range count");
	}
}

idx_t PrefixRangeFilter::SelectScan(Vector &keys, UnifiedVectorFormat &data, const SelectionVector &input,
                                  idx_t count, SelectionVector &result) const {
	if (!IsValid()) {
		for (idx_t i = 0; i < count; i++) { result.set_index(i, input.get_index(i)); }
		return count;
	}
	if (empty || !count) { return 0; }
	switch (key_type.id()) {
	case LogicalTypeId::TINYINT: return SelectScanType<int8_t>(keys, data, input, count, result);
	case LogicalTypeId::SMALLINT: return SelectScanType<int16_t>(keys, data, input, count, result);
	case LogicalTypeId::INTEGER: return SelectScanType<int32_t>(keys, data, input, count, result);
	case LogicalTypeId::BIGINT: return SelectScanType<int64_t>(keys, data, input, count, result);
	case LogicalTypeId::UTINYINT: return SelectScanType<uint8_t>(keys, data, input, count, result);
	case LogicalTypeId::USMALLINT: return SelectScanType<uint16_t>(keys, data, input, count, result);
	case LogicalTypeId::UINTEGER: return SelectScanType<uint32_t>(keys, data, input, count, result);
	case LogicalTypeId::UBIGINT: return SelectScanType<uint64_t>(keys, data, input, count, result);
	default: throw InternalException("Unsupported PRF scan type");
	}
}

idx_t PrefixRangeFilter::Select(DataChunk &chunk, SelectionVector &sel, vector<uint32_t> &,
                                const vector<idx_t> &bound_cols_applied) const {
	const auto count = chunk.size();
	if (!IsValid()) {
		for (idx_t i = 0; i < count; i++) {
			sel.set_index(i, i);
		}
		return count;
	}
	if (empty || count == 0) {
		return 0;
	}
	D_ASSERT(bound_cols_applied.size() == 1);
	auto &keys = chunk.data[bound_cols_applied[0]];
	switch (key_type.id()) {
	case LogicalTypeId::TINYINT:
		return SelectVector<int8_t>(keys, count, sel);
	case LogicalTypeId::SMALLINT:
		return SelectVector<int16_t>(keys, count, sel);
	case LogicalTypeId::INTEGER:
		return SelectVector<int32_t>(keys, count, sel);
	case LogicalTypeId::BIGINT:
		return SelectVector<int64_t>(keys, count, sel);
	case LogicalTypeId::UTINYINT:
		return SelectVector<uint8_t>(keys, count, sel);
	case LogicalTypeId::USMALLINT:
		return SelectVector<uint16_t>(keys, count, sel);
	case LogicalTypeId::UINTEGER:
		return SelectVector<uint32_t>(keys, count, sel);
	case LogicalTypeId::UBIGINT:
		return SelectVector<uint64_t>(keys, count, sel);
	default:
		throw InternalException("Unsupported RPT+ prefix range filter type " + key_type.ToString());
	}
}

void PrefixRangeFilter::BitmapMetrics::Add(uint64_t word, idx_t base, idx_t valid_bits) {
	word &= ValidBitsMask(valid_bits);
	active += PopCount(word);
	if (runs > MAX_DIRECT_RANGES) {
		return;
	}
	const auto prior = (word << 1) | previous;
	auto beginnings = word & ~prior;
	auto endings = prior & ~word & ValidBitsMask(valid_bits);
	const auto new_runs = PopCount(beginnings);
	if (runs + new_runs > MAX_DIRECT_RANGES) {
		runs = MAX_DIRECT_RANGES + 1;
		return;
	}
	while (beginnings) {
		starts[runs++] = base + CountZeros<uint64_t>::Trailing(beginnings);
		beginnings &= beginnings - 1;
	}
	while (endings) {
		D_ASSERT(ended < MAX_DIRECT_RANGES);
		ends[ended++] = base + CountZeros<uint64_t>::Trailing(endings) - 1;
		endings &= endings - 1;
	}
	previous = (word >> (valid_bits - 1)) & 1;
}

void PrefixRangeFilter::BitmapMetrics::Finish(idx_t buckets) {
	if (runs <= MAX_DIRECT_RANGES && previous) {
		ends[ended++] = buckets - 1;
	}
	D_ASSERT(runs > MAX_DIRECT_RANGES || runs == ended);
}

namespace {
class PRFAnalysisTask : public BaseExecutorTask {
public:
	PRFAnalysisTask(TaskExecutor &executor, std::function<void()> work_p)
	    : BaseExecutorTask(executor), work(std::move(work_p)) {
	}
	void ExecuteTask() override {
		work();
	}

private:
	std::function<void()> work;
};
} // namespace

PrefixRangeFilter::BitmapMetrics PrefixRangeFilter::AnalyzeBitmap(ClientContext &context, const uint64_t *source,
                                                                  idx_t buckets, uint64_t *destination) {
	const auto target_buckets = destination ? (buckets + 1) / 2 : buckets;
	const auto words = (target_buckets + 63) / 64;
	const auto source_words = (buckets + 63) / 64;
	const auto max_tasks = TaskScheduler::GetScheduler(context).NumberOfThreads();
	const idx_t task_count = source_words * sizeof(uint64_t) < 4 * 1024 * 1024
	                             ? 1
	                             : MaxValue<idx_t>(1, MinValue<idx_t>(max_tasks, (words + 32767) / 32768));
	const auto words_per_task = ((words + task_count * 64 - 1) / (task_count * 64)) * 64;
	vector<BitmapMetrics> parts(task_count);
	auto analyze = [&](idx_t task) {
		const auto begin = MinValue(words, task * words_per_task);
		const auto end = MinValue(words, begin + words_per_task);
		auto &metrics = parts[task];
		for (idx_t i = begin; i < end; i++) {
			uint64_t word;
			if (destination) {
				word = PackMergedPairs(source[i * 2]);
				if (i * 2 + 1 < source_words) {
					word |= PackMergedPairs(source[i * 2 + 1]) << 32;
				}
			} else {
				word = source[i];
			}
			const auto valid = MinValue<idx_t>(64, target_buckets - i * 64);
			word &= ValidBitsMask(valid);
			if (destination) {
				destination[i] = word;
			}
			metrics.Add(word, i * 64, valid);
		}
		metrics.Finish(MinValue(target_buckets, end * 64));
	};
	if (task_count == 1) {
		analyze(0);
		return parts[0];
	}
	TaskExecutor executor(context);
	for (idx_t task = 0; task < task_count; task++) {
		executor.ScheduleTask(make_uniq<PRFAnalysisTask>(executor, [&, task]() { analyze(task); }));
	}
	executor.WorkOnTasks();
	BitmapMetrics result;
	for (auto &part : parts) {
		result.active += part.active;
		if (result.runs > MAX_DIRECT_RANGES || part.runs > MAX_DIRECT_RANGES) {
			result.runs = MAX_DIRECT_RANGES + 1;
			continue;
		}
		for (idx_t i = 0; i < part.runs; i++) {
			if (result.runs && result.ends[result.runs - 1] + 1 == part.starts[i]) {
				result.ends[result.runs - 1] = part.ends[i];
			} else if (result.runs < MAX_DIRECT_RANGES) {
				result.starts[result.runs] = part.starts[i];
				result.ends[result.runs++] = part.ends[i];
			} else {
				result.runs = MAX_DIRECT_RANGES + 1;
				break;
			}
		}
	}
	return result;
}

double PrefixRangeFilter::FalsePositiveRate(idx_t active_buckets, idx_t candidate_shift) const {
	const long double domain_values = static_cast<long double>(span) + 1;
	const long double positive_lower_bound = static_cast<long double>(positive_count_estimate);
	if (domain_values <= positive_lower_bound) {
		return 0;
	}
	const long double bucket_width = std::ldexp(static_cast<long double>(1), static_cast<int>(candidate_shift));
	const long double covered =
	    MinValue<long double>(domain_values, static_cast<long double>(active_buckets) * bucket_width);
	if (covered <= positive_lower_bound) {
		return 0;
	}
	return static_cast<double>((covered - positive_lower_bound) / (domain_values - positive_lower_bound));
}

void PrefixRangeFilter::ConvertToDirectRanges(const BitmapMetrics &metrics) {
	direct_range_count = 0;
	for (idx_t i = 0; i < metrics.runs; i++) {
		const auto start = metrics.starts[i];
		const auto end = metrics.ends[i];
		const auto lower = minimum + (static_cast<uint64_t>(start) << shift);
		uint64_t upper;
		if (end + 1 >= logical_bucket_count) {
			upper = maximum;
		} else {
			upper = minimum + (static_cast<uint64_t>(end + 1) << shift) - 1;
		}
		direct_ranges[direct_range_count++] = {lower, upper};
	}
	mode = Mode::DIRECT_RANGES;
	bitmap_data.Reset();
	bitmap = nullptr;
	word_count = 0;
	homogeneous_runs.clear();
}

void PrefixRangeFilter::Compress(ClientContext &context) {
	auto metrics = AnalyzeBitmap(context, bitmap, logical_bucket_count);
	active_bucket_count = metrics.active;
	// The estimate is fixed at the initial level, exactly as in the regular PRF.
	// An occupied two-key bucket is not evidence that only one key is present.
	const auto distinct = estimated_multiplicity > 1.0
	                          ? static_cast<idx_t>(double(build_row_count) / estimated_multiplicity)
	                          : build_row_count;
	const auto covered = MinValue<long double>(static_cast<long double>(span) + 1,
	                                           std::ldexp(static_cast<long double>(active_bucket_count), shift));
	positive_count_estimate = MaxValue(
	    active_bucket_count, covered >= static_cast<long double>(distinct) ? distinct : static_cast<idx_t>(covered));
	false_positive_rate = FalsePositiveRate(active_bucket_count, shift);
	if (active_bucket_count == 0) {
		empty = true;
		bitmap_data.Reset();
		bitmap = nullptr;
		word_count = 0;
		return;
	}
	if (false_positive_rate <= MAX_COMPRESSION_FALSE_POSITIVE_RATE && metrics.runs <= MAX_DIRECT_RANGES) {
		ConvertToDirectRanges(metrics);
		return;
	}
	if (false_positive_rate > MAX_COMPRESSION_FALSE_POSITIVE_RATE) {
		return;
	}
	AllocatedData scratch_data;
	uint64_t *scratch = nullptr;
	idx_t scratch_capacity = 0;
	idx_t capacity = word_count;

	while (logical_bucket_count > 1) {
		const auto candidate_bucket_count = (logical_bucket_count + 1) / 2;
		const auto candidate_word_count = MaxValue<idx_t>((candidate_bucket_count + 63) / 64, 1);
		auto &allocator = BufferManager::GetBufferManager(context).GetBufferAllocator();
		if (scratch_capacity < candidate_word_count) {
			scratch_data = allocator.Allocate(64 + candidate_word_count * sizeof(uint64_t));
			scratch =
			    reinterpret_cast<uint64_t *>((reinterpret_cast<uintptr_t>(scratch_data.get()) + 63) & ~uintptr_t(63));
			scratch_capacity = candidate_word_count;
		}
		auto candidate_metrics = AnalyzeBitmap(context, bitmap, logical_bucket_count, scratch);
		const auto candidate_active = candidate_metrics.active;
		const auto candidate_shift = shift + 1;
		const auto candidate_fpr = FalsePositiveRate(candidate_active, candidate_shift);
		const auto candidate_runs = candidate_metrics.runs;
		const auto current_covered = static_cast<long double>(active_bucket_count) *
		                             std::ldexp(static_cast<long double>(1), static_cast<int>(shift));
		const auto candidate_covered = static_cast<long double>(candidate_active) *
		                               std::ldexp(static_cast<long double>(1), static_cast<int>(candidate_shift));
		if (candidate_fpr > MAX_COMPRESSION_FALSE_POSITIVE_RATE ||
		    (word_count * sizeof(uint64_t) <= CACHE_TARGET_BYTES && candidate_covered > current_covered &&
		     candidate_runs > MAX_DIRECT_RANGES)) {
			break;
		}

		std::swap(bitmap_data, scratch_data);
		std::swap(bitmap, scratch);
		std::swap(capacity, scratch_capacity);
		logical_bucket_count = candidate_bucket_count;
		word_count = candidate_word_count;
		shift = candidate_shift;
		active_bucket_count = candidate_active;
		false_positive_rate = candidate_fpr;
		if (candidate_runs <= MAX_DIRECT_RANGES) {
			ConvertToDirectRanges(candidate_metrics);
			return;
		}
	}
	// Retain only the final bitmap, not an oversized ping-pong allocation.
	if (capacity != word_count) {
		auto &allocator = BufferManager::GetBufferManager(context).GetBufferAllocator();
		auto compact = allocator.Allocate(64 + word_count * sizeof(uint64_t));
		auto target = reinterpret_cast<uint64_t *>((reinterpret_cast<uintptr_t>(compact.get()) + 63) & ~uintptr_t(63));
		std::copy_n(bitmap, word_count, target);
		bitmap_data = std::move(compact);
		bitmap = target;
	}
}

void PrefixRangeFilter::BuildLookupIndex() {
	if (mode != Mode::BITMAP || empty) {
		return;
	}
	homogeneous_runs.clear();
	// Any homogeneous interval longer than the scan limit contains an anchor.
	// Index only such runs; exclude the partially populated final word.
	const idx_t scan_limit = CACHE_TARGET_BYTES / sizeof(uint64_t);
	const auto complete_words = logical_bucket_count / 64;
	if (complete_words <= scan_limit) {
		return;
	}
	for (idx_t anchor = 0; anchor < complete_words;) {
		const auto word = bitmap[anchor];
		if (word != 0 && word != ~uint64_t(0)) {
			anchor += scan_limit;
			continue;
		}
		auto begin = anchor;
		auto end = anchor + 1;
		while (begin > 0 && bitmap[begin - 1] == word) {
			begin--;
		}
		while (end < complete_words && bitmap[end] == word) {
			end++;
		}
		if (end - begin > scan_limit) {
			homogeneous_runs.push_back({begin, end, word != 0});
		}
		do {
			anchor += scan_limit;
		} while (anchor < end);
	}
}

void PrefixRangeFilter::Finalize(ClientContext &context) {
	if (!empty) {
		auto start = BuildClock::now();
		Compress(context);
		compress_ms = ElapsedMS(start);
		start = BuildClock::now();
		BuildLookupIndex();
		index_ms = ElapsedMS(start);
		use_bloom = false_positive_rate > MAX_COMPRESSION_FALSE_POSITIVE_RATE;
		if (use_bloom) {
			InitializeBloom(context);
		}
	}
	finalized = true;
}

FilterPropagateResult PrefixRangeFilter::LookupRange(uint64_t lower, uint64_t upper) const {
	if (empty || upper < minimum || lower > maximum) {
		return FilterPropagateResult::FILTER_ALWAYS_FALSE;
	}
	const bool contained = lower >= minimum && upper <= maximum;
	lower = MaxValue<uint64_t>(lower, minimum);
	upper = MinValue<uint64_t>(upper, maximum);
	if (mode == Mode::DIRECT_RANGES) {
		for (idx_t range_idx = 0; range_idx < direct_range_count; range_idx++) {
			if (direct_ranges[range_idx].upper >= lower && direct_ranges[range_idx].lower <= upper) {
				if (contained && direct_ranges[range_idx].lower <= lower && direct_ranges[range_idx].upper >= upper) {
					return FilterPropagateResult::FILTER_ALWAYS_TRUE;
				}
				return FilterPropagateResult::NO_PRUNING_POSSIBLE;
			}
		}
		return FilterPropagateResult::FILTER_ALWAYS_FALSE;
	}

	const auto lower_bucket = static_cast<idx_t>((lower - minimum) >> shift);
	const auto upper_bucket = static_cast<idx_t>((upper - minimum) >> shift);
	const auto lower_word = static_cast<uint32_t>(lower_bucket / 64);
	const auto upper_word = static_cast<uint32_t>(upper_bucket / 64);
	// -1 denotes mixed; 0/1 denote homogeneous rejection/acceptance.
	auto classify = [](uint64_t word, uint64_t mask) -> int {
		word &= mask;
		return word == 0 ? 0 : (word == mask ? 1 : -1);
	};
	const auto lower_mask = ~uint64_t(0) << (lower_bucket % 64);
	const auto upper_mask = ValidBitsMask((upper_bucket % 64) + 1);
	int state;
	if (lower_word == upper_word) {
		state = classify(bitmap[lower_word], lower_mask & upper_mask);
	} else {
		state = classify(bitmap[lower_word], lower_mask);
		if (state < 0 || state != classify(bitmap[upper_word], upper_mask)) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
		const idx_t begin = lower_word + 1;
		const idx_t end = upper_word;
		if (end - begin <= CACHE_TARGET_BYTES / sizeof(uint64_t)) {
			for (idx_t i = begin; i < end; i++) {
				if (state != classify(bitmap[i], ~uint64_t(0))) {
					return FilterPropagateResult::NO_PRUNING_POSSIBLE;
				}
			}
		} else {
			auto entry = std::upper_bound(homogeneous_runs.begin(), homogeneous_runs.end(), begin,
			                              [](idx_t value, const HomogeneousRun &run) { return value < run.begin; });
			if (entry == homogeneous_runs.begin()) {
				return FilterPropagateResult::NO_PRUNING_POSSIBLE;
			}
			--entry;
			if (entry->end < end || int(entry->full) != state) {
				return FilterPropagateResult::NO_PRUNING_POSSIBLE;
			}
		}
	}
	return state == 0 ? FilterPropagateResult::FILTER_ALWAYS_FALSE
	                  : (state == 1 && contained ? FilterPropagateResult::FILTER_ALWAYS_TRUE
	                                             : FilterPropagateResult::NO_PRUNING_POSSIBLE);
}

FilterPropagateResult PrefixRangeFilter::CheckStatistics(BaseStatistics &stats) const {
	if (!IsValid() || stats.GetType() != key_type || stats.GetStatsType() != StatisticsType::NUMERIC_STATS ||
	    !NumericStats::HasMinMax(stats)) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	uint64_t lower;
	uint64_t upper;
	if (!ValueToComparable(NumericStats::Min(stats), lower) || !ValueToComparable(NumericStats::Max(stats), upper) ||
	    lower > upper) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	const auto result = LookupRange(lower, upper);
	return result == FilterPropagateResult::FILTER_ALWAYS_TRUE && stats.CanHaveNull()
	           ? FilterPropagateResult::NO_PRUNING_POSSIBLE
	           : result;
}

string PrefixRangeFilter::GetInfo() const {
	if (empty) {
		return "empty";
	}
	if (mode == Mode::DIRECT_RANGES) {
		return "direct ranges=" + std::to_string(direct_range_count) + " shift=" + std::to_string(shift) +
		       " fpr=" + std::to_string(false_positive_rate);
	}
	return string(use_bloom ? "bloom fallback; bitmap buckets=" : "bitmap buckets=") +
	       std::to_string(logical_bucket_count) + " active=" + std::to_string(active_bucket_count) +
	       " shift=" + std::to_string(shift) + " fpr=" + std::to_string(false_positive_rate);
}

idx_t PrefixRangeFilter::GetMemoryUsage() const {
	return bitmap_data.GetSize() + bloom_data.GetSize() + homogeneous_runs.capacity() * sizeof(HomogeneousRun) +
	       direct_range_count * sizeof(DirectRange);
}

string PrefixRangeFilter::GetBuildTimings() const {
	return "init_ms=" + std::to_string(initialize_ms) + " insert_task_ms=" + std::to_string(insert_ms) +
	       " merge_ms=" + std::to_string(merge_ms) + " compress_ms=" + std::to_string(compress_ms) +
	       " index_ms=" + std::to_string(index_ms);
}

InsertionOrderPreservingMap<string> PrefixRangeFilter::GetDiagnostics() const {
	InsertionOrderPreservingMap<string> result;
	auto precise = [](double value) {
		std::ostringstream out;
		out << std::setprecision(17) << value;
		return out.str();
	};
	result["build_rows"] = std::to_string(build_row_count);
	result["filter_ready"] = IsValid() ? "true" : "false";
	result["estimated_multiplicity"] = precise(estimated_multiplicity);
	result["estimated_distinct_keys"] = std::to_string(positive_count_estimate);
	result["initial_shift"] = std::to_string(initial_shift);
	result["final_shift"] = std::to_string(shift);
	result["final_mode"] = mode == Mode::DIRECT_RANGES ? "direct_ranges" : "bitmap";
	result["range_count"] = std::to_string(direct_range_count);
	result["active_buckets"] = std::to_string(active_bucket_count);
	result["false_positive_rate"] = precise(false_positive_rate);
	result["bloom_fallback_selected"] = use_bloom ? "true" : "false";
	result["bitmap_allocation_bytes"] = std::to_string(bitmap_data.GetSize());
	result["compression_elapsed_ms"] = std::to_string(compress_ms);
	return result;
}

} // namespace duckdb
