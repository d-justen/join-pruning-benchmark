#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb/optimizer/predicate_transfer/bloom_filter/prefix_range_filter.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include "duckdb/planner/filter/prefix_range_filter.hpp"
#include "duckdb/planner/table_filter_state.hpp"
#include "duckdb/storage/table/column_segment.hpp"
#include <thread>

namespace duckdb {

namespace {

void CheckPRFSelection(PrefixRangeFilter &filter, DataChunk &input, const vector<Value> &build, bool exact = true) {
	vector<uint32_t> lookup(STANDARD_VECTOR_SIZE);
	vector<uint32_t> scratch; // The PRF selection path must not need the intermediate result buffer.
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	const vector<idx_t> columns {1};
	filter.Lookup(input, lookup, columns);
	// Scan selection must compose with existing predicates and dictionary
	// encodings without renumbering rows or accepting NULL keys.
	UnifiedVectorFormat data;
	input.data[1].ToUnifiedFormat(input.size(), data);
	SelectionVector incoming(STANDARD_VECTOR_SIZE), output(STANDARD_VECTOR_SIZE);
	idx_t incoming_count = 0;
	for (idx_t i = input.size(); i > 0; i -= MinValue<idx_t>(i, 2)) {
		incoming.set_index(incoming_count++, i - 1);
	}
	auto scan_count = filter.SelectScan(input.data[1], data, incoming, incoming_count, output);
	idx_t expected_scan_count = 0;
	for (idx_t i = 0; i < incoming_count; i++) {
		if (lookup[incoming.get_index(i)]) {
			REQUIRE(expected_scan_count < scan_count);
			REQUIRE(output.get_index(expected_scan_count++) == incoming.get_index(i));
		}
	}
	REQUIRE(expected_scan_count == scan_count);
	const auto selected_count = filter.Select(input, sel, scratch, columns);
	idx_t expected_count = 0;
	for (idx_t i = 0; i < input.size(); i++) {
		const auto probe = input.GetValue(1, i);
		bool member = false;
		for (auto &key : build) {
			member |= !probe.IsNull() && !key.IsNull() && Value::NotDistinctFrom(probe, key);
		}
		if (exact || member || probe.IsNull()) {
			REQUIRE(lookup[i] == uint32_t(member));
		}
		if (lookup[i]) {
			REQUIRE(expected_count < selected_count);
			REQUIRE(sel.get_index(expected_count++) == i);
		}
	}
	REQUIRE(expected_count == selected_count);
}

void ExercisePRFVectors(ClientContext &context, const LogicalType &type, const vector<Value> &keys,
                        const vector<Value> &probes, bool exact = true, idx_t expected_ranges = 0) {
	PrefixRangeFilter filter(type);
	DataChunk build;
	build.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER, type});
	for (idx_t i = 0; i < keys.size(); i++) {
		build.SetValue(1, i, keys[i]);
	}
	// Duplicate and NULL build values should not change membership.
	build.SetValue(1, keys.size(), keys[0]);
	build.SetValue(1, keys.size() + 1, Value(type));
	build.SetCardinality(keys.size() + 2);
	filter.Initialize(context, build.size(), keys.front(), keys.back());
	filter.Insert(build, {1});
	filter.Finalize(context);
	if (filter.NeedsBloomFallback()) {
		filter.Insert(build, {1});
		filter.CompleteBloomFallback();
	}
	if (exact) {
		const auto ranges = expected_ranges ? expected_ranges : keys.size();
		const auto representation = ranges <= 4 ? "direct ranges=" + std::to_string(ranges) : "bitmap";
		REQUIRE(filter.GetInfo().find(representation) == 0);
	}

	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER, type});
	for (idx_t i = 0; i < probes.size(); i++) {
		input.SetValue(1, i, probes[i]);
	}
	input.SetCardinality(probes.size());
	CheckPRFSelection(filter, input, keys, exact);

	// An all-valid dictionary with a different order and repeated source positions.
	SelectionVector permutation(STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < probes.size() * 2; i++) {
		permutation.set_index(i, (i * 7 + 3) % probes.size());
	}
	DataChunk dictionary;
	dictionary.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER, type});
	dictionary.Reference(input);
	dictionary.Slice(permutation, probes.size() * 2);
	CheckPRFSelection(filter, dictionary, keys, exact);

	input.SetValue(1, 0, Value(type));
	input.SetValue(1, probes.size() / 2, Value(type));
	CheckPRFSelection(filter, input, keys, exact);
	dictionary.Reference(input);
	dictionary.Slice(permutation, probes.size() * 2);
	CheckPRFSelection(filter, dictionary, keys, exact);

	// Constant hit, miss and NULL paths (including full-size chunks).
	for (auto &value : vector<Value> {keys.front(), probes.front(), Value(type)}) {
		input.data[1].Reference(value);
		input.SetCardinality(STANDARD_VECTOR_SIZE);
		CheckPRFSelection(filter, input, keys, exact);
	}
	input.SetCardinality(0);
	CheckPRFSelection(filter, input, keys, exact);
}

} // namespace

TEST_CASE("RPT PRF selection preserves membership and vector positions", "[rpt_prf]") {
	DuckDB db(nullptr);
	Connection con(db);
	for (auto &type : vector<LogicalType> {LogicalType::TINYINT, LogicalType::SMALLINT, LogicalType::INTEGER,
	                                       LogicalType::BIGINT, LogicalType::UTINYINT, LogicalType::USMALLINT,
	                                       LogicalType::UINTEGER, LogicalType::UBIGINT}) {
		INFO(type.ToString());
		vector<Value> probes;
		for (idx_t i = 0; i < 71; i++) {
			probes.push_back(Value::BIGINT(i).DefaultCastAs(type));
		}
		probes.push_back(Value::MinimumValue(type));
		probes.push_back(Value::MaximumValue(type));
		for (idx_t range_count = 1; range_count <= 5; range_count++) {
			INFO(range_count);
			vector<Value> keys;
			for (idx_t i = 1; i <= range_count; i++) {
				keys.push_back(Value::BIGINT(i * 10).DefaultCastAs(type));
			}
			ExercisePRFVectors(*con.context, type, keys, probes);
			if (range_count <= 4) {
				keys.clear();
				for (idx_t i = 1; i <= range_count; i++) {
					for (idx_t offset = 0; offset < 3; offset++) {
						keys.push_back(Value::BIGINT(i * 10 + offset).DefaultCastAs(type));
					}
				}
				ExercisePRFVectors(*con.context, type, keys, probes, true, range_count);
			}
		}
	}
}

TEST_CASE("RPT PRF selection handles full-width integer domains", "[rpt_prf]") {
	DuckDB db(nullptr);
	Connection con(db);
	const vector<Value> signed_keys {Value::BIGINT(INT64_MIN), Value::BIGINT(INT64_MIN / 2), Value::BIGINT(-1),
	                                 Value::BIGINT(1),         Value::BIGINT(INT64_MAX / 2), Value::BIGINT(INT64_MAX)};
	const vector<Value> signed_probes {Value::BIGINT(INT64_MIN),
	                                   Value::BIGINT(INT64_MIN + 1),
	                                   Value::BIGINT(INT64_MIN / 2),
	                                   Value::BIGINT(-2),
	                                   Value::BIGINT(-1),
	                                   Value::BIGINT(0),
	                                   Value::BIGINT(1),
	                                   Value::BIGINT(INT64_MAX / 2),
	                                   Value::BIGINT(INT64_MAX - 1),
	                                   Value::BIGINT(INT64_MAX)};
	ExercisePRFVectors(*con.context, LogicalType::BIGINT, signed_keys, signed_probes, false);
	const vector<Value> unsigned_keys {Value::UBIGINT(0),
	                                   Value::UBIGINT(1),
	                                   Value::UBIGINT(UINT64_MAX / 4),
	                                   Value::UBIGINT(UINT64_MAX / 2),
	                                   Value::UBIGINT(UINT64_MAX - 1),
	                                   Value::UBIGINT(UINT64_MAX)};
	const vector<Value> unsigned_probes {Value::UBIGINT(0),
	                                     Value::UBIGINT(1),
	                                     Value::UBIGINT(2),
	                                     Value::UBIGINT(UINT64_MAX / 4),
	                                     Value::UBIGINT(UINT64_MAX / 2),
	                                     Value::UBIGINT(UINT64_MAX / 2 + 1),
	                                     Value::UBIGINT(UINT64_MAX - 1),
	                                     Value::UBIGINT(UINT64_MAX)};
	ExercisePRFVectors(*con.context, LogicalType::UBIGINT, unsigned_keys, unsigned_probes, false);
}

TEST_CASE("RPT PRF selection handles unavailable and empty filters", "[rpt_prf]") {
	DuckDB db(nullptr);
	Connection con(db);
	PrefixRangeFilter filter(LogicalType::INTEGER);
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	input.SetValue(0, 0, Value::INTEGER(10));
	input.SetValue(0, 1, Value(LogicalType::INTEGER));
	input.SetCardinality(2);
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	vector<uint32_t> scratch;
	REQUIRE(filter.Select(input, sel, scratch, {0}) == 2);
	REQUIRE(sel.get_index(0) == 0);
	REQUIRE(sel.get_index(1) == 1);
	filter.Initialize(*con.context, 0, Value(LogicalType::INTEGER), Value(LogicalType::INTEGER));
	filter.Finalize(*con.context);
	REQUIRE(filter.Select(input, sel, scratch, {0}) == 0);
}

TEST_CASE("RPT PRF compression matches a scalar dyadic reference", "[rpt_prf]") {
	DuckDB db(nullptr);
	Connection con(db);
	// Odd final words, runs crossing word boundaries, dense/sparse bitmaps,
	// repeated accepted reductions and rejected reductions.
	for (idx_t domain : {idx_t(65), idx_t(257), idx_t(16385), idx_t(262147)}) {
		for (idx_t pattern = 0; pattern < 6; pattern++) {
			vector<uint8_t> bits(domain, 0);
			for (idx_t i = 0; i < domain; i++) {
				bits[i] = pattern == 0   ? i % 2 == 0
				          : pattern == 1 ? i % 127 < 65
				          : pattern == 2 ? i % 4096 < 128
				          : pattern == 3 ? i % 8192 == 0
				          : pattern == 4 ? i < domain / 2
				                         : (i * 1103515245ULL + 12345) % 1009 < 100;
			}
			bits.front() = bits.back() = 1;
			PrefixRangeFilter filter(LogicalType::UINTEGER);
			idx_t build_rows = 0;
			for (auto bit : bits) {
				build_rows += bit;
			}
			filter.Initialize(*con.context, build_rows, Value::UINTEGER(0), Value::UINTEGER(domain - 1));
			filter.SetSingleWriter();
			DataChunk chunk;
			chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::UINTEGER});
			idx_t count = 0, base_active = 0;
			for (idx_t i = 0; i < domain; i++) {
				if (!bits[i])
					continue;
				base_active++;
				chunk.SetValue(0, count++, Value::UINTEGER(i));
				if (count == STANDARD_VECTOR_SIZE) {
					chunk.SetCardinality(count);
					filter.Insert(chunk, {0});
					count = 0;
				}
			}
			chunk.SetCardinality(count);
			filter.Insert(chunk, {0});
			filter.Finalize(*con.context);
			auto runs = [](const vector<uint8_t> &values) {
				idx_t n = 0;
				for (idx_t i = 0; i < values.size(); i++)
					n += values[i] && (!i || !values[i - 1]);
				return n;
			};
			idx_t shift = 0, active = base_active;
			while (bits.size() > 1 && runs(bits) > 4) {
				vector<uint8_t> candidate((bits.size() + 1) / 2, 0);
				for (idx_t i = 0; i < bits.size(); i++)
					candidate[i / 2] |= bits[i];
				idx_t next_active = 0;
				for (auto bit : candidate)
					next_active += bit;
				const auto covered = next_active * (idx_t(1) << (shift + 1));
				const auto fpr = domain == base_active
				                     ? 0.0L
				                     : (static_cast<long double>(MinValue<idx_t>(domain, covered)) - base_active) /
				                           (domain - base_active);
				if (fpr > 0.001 || (((bits.size() + 63) / 64) * 8 <= 16384 && covered > active * (idx_t(1) << shift) &&
				                    runs(candidate) > 4))
					break;
				bits = std::move(candidate);
				active = next_active;
				shift++;
			}
			REQUIRE(filter.GetInfo().find("shift=" + std::to_string(shift) + " ") != string::npos);
			vector<uint32_t> lookup(STANDARD_VECTOR_SIZE);
			for (idx_t begin = 0; begin < domain; begin += STANDARD_VECTOR_SIZE) {
				count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, domain - begin);
				for (idx_t i = 0; i < count; i++)
					chunk.SetValue(0, i, Value::UINTEGER(begin + i));
				chunk.SetCardinality(count);
				filter.Lookup(chunk, lookup, {0});
				for (idx_t i = 0; i < count; i++)
					REQUIRE(lookup[i] == bits[(begin + i) >> shift]);
			}
		}
	}
}

TEST_CASE("RPT PRF shoulder compression survives selectivity and multiplicity changes", "[rpt_prf]") {
	DuckDB db(nullptr);
	Connection con(db);
	const idx_t domain = 100000000;
	for (idx_t threads : {idx_t(1), idx_t(4)}) {
		REQUIRE(!con.Query("SET threads=" + std::to_string(threads))->HasError());
		for (idx_t rows : {idx_t(10000), idx_t(100000), idx_t(300000), idx_t(1000000)}) {
			for (idx_t multiplicity : {idx_t(1), idx_t(8)}) {
				INFO(threads);
				INFO(rows);
				INFO(multiplicity);
				const auto distinct = rows / multiplicity;
				PrefixRangeFilter filter(LogicalType::BIGINT, double(multiplicity));
				filter.Initialize(*con.context, rows, Value::BIGINT(0), Value::BIGINT(domain - 1));
				filter.SetSingleWriter();
				DataChunk chunk;
				chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT});
				for (idx_t begin = 0; begin < rows; begin += STANDARD_VECTOR_SIZE) {
					const auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, rows - begin);
					for (idx_t i = 0; i < count; i++) {
						const auto rank = (begin + i) / multiplicity;
						const auto key = rank < distinct / 2 ? rank : domain - distinct + rank;
						chunk.SetValue(0, i, Value::BIGINT(key));
					}
					chunk.SetCardinality(count);
					filter.Insert(chunk, {0});
				}
				filter.Finalize(*con.context);
				REQUIRE(filter.IsValid());
				REQUIRE(!filter.NeedsBloomFallback());
				REQUIRE(filter.GetInfo().find("direct ranges=2") == 0);
				auto diagnostics = filter.GetDiagnostics();
				REQUIRE(diagnostics["initial_shift"] == "1");
				REQUIRE(diagnostics["estimated_distinct_keys"] == std::to_string(distinct));
				// Check boundary membership, holes, out-of-domain keys and NULL.
				vector<int64_t> probes {0,
				                        int64_t(distinct / 2 - 1),
				                        int64_t(distinct / 2 + 2),
				                        int64_t(domain / 2),
				                        int64_t(domain - distinct / 2),
				                        int64_t(domain - 1),
				                        -1,
				                        int64_t(domain)};
				for (idx_t i = 0; i < probes.size(); i++)
					chunk.SetValue(0, i, Value::BIGINT(probes[i]));
				chunk.SetValue(0, probes.size(), Value(LogicalType::BIGINT));
				chunk.SetCardinality(probes.size() + 1);
				vector<uint32_t> lookup(STANDARD_VECTOR_SIZE);
				filter.Lookup(chunk, lookup, {0});
				for (idx_t i = 0; i < probes.size(); i++) {
					const auto key = probes[i];
					REQUIRE(lookup[i] ==
					        uint32_t(key >= 0 && key < int64_t(domain) &&
					                 (key < int64_t(distinct / 2) || key >= int64_t(domain - distinct / 2))));
				}
				REQUIRE(lookup[probes.size()] == 0);
			}
		}
	}
}

TEST_CASE("RPT PRF multiplicity can trigger sector Bloom fallback", "[rpt_prf]") {
	DuckDB db(nullptr);
	Connection con(db);
	// A 70M span starts with width-two buckets. Each selected bucket contains
	// one key: a fractional source multiplicity must not count duplicates as
	// another occupied value. Ignoring multiplicity would accept this bitmap.
	PrefixRangeFilter filter(LogicalType::BIGINT, 1.5);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT});
	filter.Initialize(*con.context, 150000, Value::BIGINT(0), Value::BIGINT(99999 * 700));
	auto insert = [&]() {
		for (idx_t begin = 0; begin < 150000; begin += STANDARD_VECTOR_SIZE) {
			const auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, 150000 - begin);
			for (idx_t i = 0; i < count; i++)
				chunk.SetValue(0, i, Value::BIGINT(((begin + i) % 100000) * 700));
			chunk.SetCardinality(count);
			filter.Insert(chunk, {0});
		}
	};
	insert();
	filter.Finalize(*con.context);
	REQUIRE(filter.NeedsBloomFallback());
	REQUIRE(!filter.IsValid());
	insert();
	filter.CompleteBloomFallback();
	REQUIRE(filter.IsValid());
	idx_t negatives_passed = 0;
	vector<uint32_t> lookup(STANDARD_VECTOR_SIZE);
	for (idx_t begin = 0; begin < 200000; begin += STANDARD_VECTOR_SIZE) {
		const auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, 200000 - begin);
		for (idx_t i = 0; i < count; i++) {
			const auto rank = begin + i;
			chunk.SetValue(0, i, Value::BIGINT((rank / 2) * 700 + rank % 2));
		}
		chunk.SetCardinality(count);
		filter.Lookup(chunk, lookup, {0});
		for (idx_t i = 0; i < count; i++) {
			if ((begin + i) % 2 == 0)
				REQUIRE(lookup[i] == 1);
			else
				negatives_passed += lookup[i];
		}
	}
	// The PRF bitmap would accept virtually all these adjacent negative keys.
	REQUIRE(negatives_passed < 10000);
}

TEST_CASE("RPT PRF private construction and homogeneous range classification", "[rpt_prf]") {
	DuckDB db(nullptr);
	Connection con(db);
	PrefixRangeFilter filter(LogicalType::INTEGER);
	filter.Initialize(*con.context, 300001, Value::INTEGER(0), Value::INTEGER(600000));
	vector<unique_ptr<PrefixRangeFilter>> locals;
	for (idx_t i = 0; i < 4; i++)
		locals.push_back(filter.CreateLocal(*con.context));
	vector<std::thread> threads;
	for (idx_t task = 0; task < locals.size(); task++) {
		threads.emplace_back([&, task]() {
			DataChunk chunk;
			chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
			idx_t count = 0;
			for (idx_t i = task; i <= 600000; i += 4) {
				if (!(i < 200000 || (i >= 400000 && i % 2 == 0)))
					continue;
				chunk.SetValue(0, count++, Value::INTEGER(i));
				if (count == STANDARD_VECTOR_SIZE) {
					chunk.SetCardinality(count);
					locals[task]->Insert(chunk, {0});
					count = 0;
				}
			}
			chunk.SetCardinality(count);
			locals[task]->Insert(chunk, {0});
		});
	}
	for (auto &thread : threads)
		thread.join();
	for (auto &local : locals)
		filter.MergeLocal(*local);
	filter.Finalize(*con.context);
	REQUIRE(filter.GetInfo().find("shift=0 ") != string::npos);
	auto check = [&](int32_t lower, int32_t upper, bool nullable, FilterPropagateResult expected) {
		auto stats = NumericStats::CreateUnknown(LogicalType::INTEGER);
		NumericStats::SetMin(stats, Value::INTEGER(lower));
		NumericStats::SetMax(stats, Value::INTEGER(upper));
		stats.Set(nullable ? StatsInfo::CAN_HAVE_NULL_VALUES : StatsInfo::CANNOT_HAVE_NULL_VALUES);
		REQUIRE(filter.CheckStatistics(stats) == expected);
	};
	const auto yes = FilterPropagateResult::FILTER_ALWAYS_TRUE;
	const auto no = FilterPropagateResult::FILTER_ALWAYS_FALSE;
	const auto mixed = FilterPropagateResult::NO_PRUNING_POSSIBLE;
	check(0, 199999, false, yes);
	check(0, 199999, true, mixed);
	check(-1, 199999, false, mixed);
	check(200000, 399999, false, no);
	check(199999, 400000, false, mixed);
	check(400000, 600000, false, mixed);
	check(600001, 700000, false, no);
	// Boundary masks, partial words and short-range scanning.
	for (int32_t i : {0, 63, 64, 199999, 200000, 399999, 400000, 599999, 600000}) {
		check(i, i, false, i < 200000 || (i >= 400000 && i % 2 == 0) ? yes : no);
	}
}

TEST_CASE("PRF scan ownership survives checkpoints and repeated execution", "[rpt_prf]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto sql = [&](const string &query) {
		auto result = con.Query(query);
		INFO(query);
		if (result->HasError()) { INFO(result->GetError()); REQUIRE(!result->HasError()); }
		REQUIRE(!result->HasError());
		return result;
	};
	sql("SET rpt_filter='prf'");
	sql("SET threads=4");
	sql("PRAGMA verify_parallelism");
	sql("CREATE TABLE a AS SELECT (i%50000)::INTEGER id, i%7 payload FROM range(200000) t(i)");
	sql("CREATE TABLE b AS SELECT i::INTEGER id FROM range(100000) t(i)");
	sql("CREATE TABLE c AS SELECT i::INTEGER id FROM range(150000) t(i)");
	const string query = "SELECT count(*) FROM a JOIN b USING(id) JOIN c USING(id) WHERE a.payload=0";
	sql("PREPARE prf_scan_query AS " + query);
	for (idx_t stage = 0; stage < 3; stage++) {
		if (stage == 1) {
			sql("CHECKPOINT");
		} else if (stage == 2) {
			sql("BEGIN");
			// Stay within prepared-plan static statistics bounds. This fork also
			// retains stale static bounds with rpt_filter=none after domain expansion.
			sql("INSERT INTO a VALUES (49001,0),(NULL,0)");
			sql("INSERT INTO b VALUES (49001),(NULL)");
			sql("INSERT INTO c VALUES (49001),(NULL)");
			sql("UPDATE a SET id=49002 WHERE id=1");
			sql("DELETE FROM b WHERE id=2");
		}
		for (idx_t repeat = 0; repeat < 2; repeat++) {
			INFO(stage);
			INFO(repeat);
			auto actual = sql("EXECUTE prf_scan_query")->GetValue(0, 0);
			sql("SET rpt_filter='none'");
			auto expected = sql(query)->GetValue(0, 0);
			REQUIRE(actual == expected);
			sql("SET rpt_filter='prf'");
		}
		auto explain = sql("EXPLAIN ANALYZE " + query)->GetValue(1, 0).ToString();
		REQUIRE(explain.find("Tuple Placement: scan") != string::npos);
		sql("SET rpt_prf_scan_pushdown=false");
		auto fallback = sql("EXPLAIN ANALYZE " + query)->GetValue(1, 0).ToString();
		REQUIRE(fallback.find("Tuple Placement: scan") == string::npos);
		REQUIRE(fallback.find("Tuple Placement:") != string::npos);
		sql("SET rpt_prf_scan_pushdown=true");
	}
	sql("ROLLBACK");
}

TEST_CASE("RPT selectivity sampling counts scanned rows before compaction", "[rpt_prf]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto sql = [&](const string &query) {
		auto result = con.Query(query);
		INFO(query);
		if (result->HasError()) {
			INFO(result->GetError());
		}
		REQUIRE(!result->HasError());
		return result;
	};
	sql("SET rpt_filter='prf'");
	sql("SET rpt_prf_scan_pushdown=true");
	sql("SET threads=1");
	// This selective scan spans more than 32 row groups. Its output is compacted
	// into a small number of chunks, which must not be mistaken for the number of
	// physical input vectors when RPT samples build selectivity.
	sql("CREATE TABLE large_selective_build AS "
	    "SELECT i::INTEGER id, i%100=0 keep FROM range(5000000) t(i)");
	sql("CREATE TABLE selective_probe AS "
	    "SELECT (i%5000000)::INTEGER id FROM range(10000000) t(i)");
	const string query =
	    "SELECT count(*) FROM selective_probe p JOIN large_selective_build b USING(id) WHERE b.keep";
	REQUIRE(sql(query)->GetValue(0, 0) == Value::BIGINT(100000));
	auto explain = sql("EXPLAIN ANALYZE " + query)->GetValue(1, 0).ToString();
	REQUIRE(explain.find("bitmap buckets=") != string::npos);
	REQUIRE(explain.find("Tuple Placement: scan") != string::npos);
}

TEST_CASE("PRF storage selection composes repeatedly and grows beyond a vector", "[rpt_prf]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto filter = make_shared_ptr<PrefixRangeFilter>(LogicalType::INTEGER);
	DataChunk build;
	build.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	build.SetValue(0, 0, Value::INTEGER(7));
	build.SetCardinality(1);
	filter->Initialize(*con.context, 1, Value::INTEGER(7), Value::INTEGER(7));
	filter->Insert(build, {0});
	filter->Finalize(*con.context);
	PrefixRangeTableFilter wrapper(filter);
	auto state = TableFilterState::Initialize(*con.context, wrapper);
	Vector keys(Value::INTEGER(7));
	UnifiedVectorFormat data;
	keys.ToUnifiedFormat(STANDARD_VECTOR_SIZE, data);
	for (idx_t count : {idx_t(10), idx_t(STANDARD_VECTOR_SIZE * 3)}) {
		SelectionVector sel(0, count);
		for (idx_t repeat = 0; repeat < 4; repeat++) {
			idx_t approved = count;
			REQUIRE(ColumnSegment::FilterSelection(sel, keys, data, wrapper, *state, count, approved) == count);
			for (idx_t i = 0; i < count; i++) {
				REQUIRE(sel.get_index(i) == i);
			}
		}
	}
}

} // namespace duckdb
