//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/filter/prefix_range_filter.hpp
//
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/optimizer/predicate_transfer/bloom_filter/runtime_filter.hpp"
#include "duckdb/planner/table_filter.hpp"

namespace duckdb {

//! Statistics and tuple filter; OptionalFilter can restrict it to statistics.
class PrefixRangeTableFilter : public TableFilter {
public:
	static constexpr const TableFilterType TYPE = TableFilterType::PREFIX_RANGE_FILTER;

	explicit PrefixRangeTableFilter(shared_ptr<RuntimeFilter> filter_p = nullptr,
	                                shared_ptr<RuntimeFilterScanApplication> application_p = nullptr);

	shared_ptr<RuntimeFilter> filter;
	shared_ptr<RuntimeFilterScanApplication> application;

	FilterPropagateResult CheckStatistics(BaseStatistics &stats) const override;
	string ToString(const string &column_name) const override;
	unique_ptr<TableFilter> Copy() const override;
	bool Equals(const TableFilter &other) const override;
	unique_ptr<Expression> ToExpression(const Expression &column) const override;

	void Serialize(Serializer &serializer) const override;
	static unique_ptr<TableFilter> Deserialize(Deserializer &deserializer);
};

} // namespace duckdb
