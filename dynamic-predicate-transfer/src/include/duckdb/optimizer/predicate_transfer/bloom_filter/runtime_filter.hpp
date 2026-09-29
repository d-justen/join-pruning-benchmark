//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/predicate_transfer/bloom_filter/runtime_filter.hpp
//
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/common/enums/filter_propagate_result.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/planner/column_binding.hpp"
#include <atomic>

namespace duckdb {

class BaseStatistics;
class ClientContext;

enum class RuntimeFilterType { CASCADE, PREFIX_RANGE };

//! Per application, published by a native sequential source after filter merging.
//! A false value always means that the downstream operator must retain its probe.
struct RuntimeFilterScanApplication {
	std::atomic<bool> owned_by_scan {false};
};

//! Common run-time interface for RPT+ transfer filters.
class RuntimeFilter {
public:
	virtual ~RuntimeFilter() = default;

	virtual void Initialize(ClientContext &context, idx_t row_count, const Value &min, const Value &max) = 0;
	virtual void Insert(DataChunk &chunk, const vector<idx_t> &bound_cols_built) = 0;
	virtual int Lookup(DataChunk &chunk, vector<uint32_t> &results, const vector<idx_t> &bound_cols_applied) const = 0;
	//! Return positions in the input chunk. The default preserves Cascade's batched lookup.
	virtual idx_t Select(DataChunk &chunk, SelectionVector &sel, vector<uint32_t> &scratch,
	                     const vector<idx_t> &bound_cols_applied) const {
		Lookup(chunk, scratch, bound_cols_applied);
		idx_t selected_count = 0;
		for (idx_t i = 0; i < chunk.size(); i++) {
			sel.set_index(selected_count, i);
			selected_count += scratch[i];
		}
		return selected_count;
	}
	virtual void Finalize(ClientContext &context) = 0;

	virtual bool IsValid() const = 0;
	virtual bool RequiresBounds() const = 0;
	virtual RuntimeFilterType GetType() const = 0;
	virtual string GetName() const = 0;
	virtual string GetInfo() const = 0;
	virtual idx_t GetMemoryUsage() const = 0;

	virtual FilterPropagateResult CheckStatistics(BaseStatistics &stats) const {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
};

//! A filter plus the build/apply column bindings for one RPT+ transfer edge.
class RuntimeFilterUsage {
public:
	RuntimeFilterUsage(shared_ptr<RuntimeFilter> filter_p, const vector<idx_t> &applied, const vector<idx_t> &built)
	    : filter(std::move(filter_p)), bound_cols_applied(applied), bound_cols_built(built) {
	}

	bool IsValid() const {
		return filter->IsValid();
	}

	int Lookup(DataChunk &chunk, vector<uint32_t> &results) const {
		return filter->Lookup(chunk, results, bound_cols_applied);
	}

	idx_t Select(DataChunk &chunk, SelectionVector &sel, vector<uint32_t> &scratch) const {
		return filter->Select(chunk, sel, scratch, bound_cols_applied);
	}

	void Insert(DataChunk &chunk) const {
		filter->Insert(chunk, bound_cols_built);
	}

	shared_ptr<RuntimeFilter> filter;
	shared_ptr<RuntimeFilterScanApplication> scan_application;
	vector<idx_t> bound_cols_applied;
	vector<idx_t> bound_cols_built;
};

} // namespace duckdb
