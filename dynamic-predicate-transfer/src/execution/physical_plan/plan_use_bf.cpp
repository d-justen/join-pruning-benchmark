#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/operator/logical_use_bf.hpp"
#include "duckdb/execution/operator/filter/physical_use_bf.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/client_config.hpp"

namespace duckdb {

//! Find a source in this streaming pipeline. Materialization and joins are
//! boundaries: their earlier sources cannot accept a filter from this stage.
static PhysicalTableScan *FindPRFScan(PhysicalOperator &plan, idx_t &column, const LogicalType &type) {
	if (column >= plan.types.size() || plan.types[column] != type) {
		return nullptr;
	}
	switch (plan.type) {
	case PhysicalOperatorType::TABLE_SCAN: {
		auto &scan = plan.Cast<PhysicalTableScan>();
		if (scan.function.function != TableScanFunction::GetFunction().function || scan.extra_info.sample_options) {
			return nullptr;
		}
		if (!scan.projection_ids.empty()) {
			column = scan.projection_ids[column];
		}
		if (column >= scan.column_ids.size() || scan.column_ids[column].IsVirtualColumn() ||
		    !scan.column_ids[column].GetChildIndexes().empty() ||
		    scan.returned_types[scan.column_ids[column].GetPrimaryIndex()] != type) {
			return nullptr;
		}
		return &scan;
	}
	case PhysicalOperatorType::PROJECTION: {
		auto &projection = plan.Cast<PhysicalProjection>();
		// Moving a prefilter must not change evaluation of computed expressions.
		for (auto &expr : projection.select_list) {
			if (expr->GetExpressionClass() != ExpressionClass::BOUND_REF) {
				return nullptr;
			}
		}
		column = projection.select_list[column]->Cast<BoundReferenceExpression>().index;
		break;
	}
	case PhysicalOperatorType::FILTER:
	case PhysicalOperatorType::USE_BF:
		break;
	default:
		return nullptr;
	}
	return plan.children.size() == 1 ? FindPRFScan(plan.children[0], column, type) : nullptr;
}

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalUseBF &op) {
	auto &plan = CreatePlan(*op.children[0]); // Generate child plan
	auto bf_creator = CreatePlanFromRelated(*op.related_create_bf);

	auto &bf_plan = op.filter_plan;
	size_t target_idx = std::numeric_limits<size_t>::max();
	for (size_t i = 0; i < bf_creator->filter_plans.size(); i++) {
		auto &filter_plan = bf_creator->filter_plans[i];
		if (filter_plan->apply == bf_plan->apply) {
			target_idx = i;
			break; // Found the target, exit loop
		}
	}
	D_ASSERT(target_idx != std::numeric_limits<size_t>::max());
	if (ClientConfig::GetConfig(context).rpt_prf_scan_pushdown &&
	    bf_creator->plan_filters[target_idx]->GetType() == RuntimeFilterType::PREFIX_RANGE) {
		auto column = bf_plan->bound_cols_apply[0];
		auto scan = FindPRFScan(plan, column, bf_plan->return_types[0]);
		if (scan) {
			if (!scan->dynamic_filters) {
				scan->dynamic_filters = make_shared_ptr<DynamicTableFilterSet>();
			}
			auto &target = bf_creator->scan_targets[target_idx];
			target.filters = scan->dynamic_filters;
			target.column = column;
		}
	}

	auto &use_bf = Make<PhysicalUseBF>(plan.types, op.filter_plan, std::move(bf_creator->filters_to_create[target_idx]),
	                                   bf_creator, op.estimated_cardinality, op.below_join);
	use_bf.children.emplace_back(plan);
	return use_bf;
}

} // namespace duckdb
