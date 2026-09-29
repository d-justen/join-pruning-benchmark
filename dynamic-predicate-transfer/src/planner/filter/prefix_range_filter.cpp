#include "duckdb/planner/filter/prefix_range_filter.hpp"

#include "duckdb/planner/expression/bound_constant_expression.hpp"

namespace duckdb {

PrefixRangeTableFilter::PrefixRangeTableFilter(shared_ptr<RuntimeFilter> filter_p,
                                               shared_ptr<RuntimeFilterScanApplication> application_p)
    : TableFilter(TableFilterType::PREFIX_RANGE_FILTER), filter(std::move(filter_p)),
      application(std::move(application_p)) {
}

FilterPropagateResult PrefixRangeTableFilter::CheckStatistics(BaseStatistics &stats) const {
	if (!filter || !filter->IsValid()) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	return filter->CheckStatistics(stats);
}

string PrefixRangeTableFilter::ToString(const string &column_name) const {
	return column_name + " IN RPT_PREFIX_RANGE_FILTER";
}

unique_ptr<TableFilter> PrefixRangeTableFilter::Copy() const {
	return make_uniq<PrefixRangeTableFilter>(filter, application);
}

bool PrefixRangeTableFilter::Equals(const TableFilter &other_p) const {
	if (other_p.filter_type != TableFilterType::PREFIX_RANGE_FILTER) {
		return false;
	}
	auto &other = other_p.Cast<PrefixRangeTableFilter>();
	return filter.get() == other.filter.get() && application.get() == other.application.get();
}

unique_ptr<Expression> PrefixRangeTableFilter::ToExpression(const Expression &column) const {
	// Generic expression consumers do not claim scan ownership. The downstream
	// RPT operator retains its probe when a source cannot execute this filter.
	return make_uniq<BoundConstantExpression>(Value::BOOLEAN(true));
}

} // namespace duckdb
