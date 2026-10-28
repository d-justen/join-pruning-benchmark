"""Dimension panels for larger versions of the ordered join-pruning figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


def predicate_label(parameters, *, compact=True):
    kind = parameters['query_type']
    if not compact:
        return {'range': 'Range', 'shoulder': 'Shoulder', 'scattered': 'Scattered'}.get(
            kind, f"Seasonal {parameters.get('cluster_count', '')}")
    return {'range': 'R', 'shoulder': 'Sh', 'scattered': 'Sc'}.get(
        kind, f"Se{parameters.get('cluster_count', '')}")


def draw_ordered_panels(path, entries, variant_order, colors, display_variant):
    dimensions = sorted({e['query_parameters']['dimension_cardinality'] for e in entries})
    variants = [v for v in variant_order if any(s['variant_label'] == v for e in entries for s in e['series'])]
    fig, axes = plt.subplots(2, len(dimensions), squeeze=False, figsize=(3.6 * len(dimensions), 3.5),
                             sharey='row', gridspec_kw={'hspace': .06, 'wspace': .10})
    width = .8 / len(variants)
    bars = 0
    for di, dimension in enumerate(dimensions):
        selected = [e for e in entries if e['query_parameters']['dimension_cardinality'] == dimension]
        upper, lower = axes[:, di]
        for qi, entry in enumerate(selected):
            series = {s['variant_label']: s for s in entry['series']}
            for vi, variant in enumerate(variants):
                if variant not in series:
                    raise ValueError(f"missing figure measurement: {entry['query_name']}/{variant}")
                value = series[variant]
                x = qi + (vi - (len(variants) - 1) / 2) * width
                upper.bar(x, value['median_ms'], width=width, color=colors[variant], edgecolor='white', linewidth=.3)
                lower.bar(x, value['join_probe_rows'], width=width, color=colors[variant], linewidth=0)
                lower.bar(x, value['probe_rows_filtered_after_scan'], bottom=value['join_probe_rows'], width=width,
                          facecolor='white', edgecolor=colors[variant], hatch='....', linewidth=.5)
                bars += 1
        title = (f"{dimension / 1_000_000_000:g}B" if dimension >= 1_000_000_000 else
                 f"{dimension // 1_000_000}M" if dimension >= 1_000_000 else f"{dimension // 1000}K")
        upper.set_title(f"{title} dimension rows", fontsize=9)
        upper.set_yscale('log')
        upper.set_xticks([])
        lower.set_xticks(range(len(selected)), [predicate_label(e['query_parameters']) for e in selected], fontsize=8)
        lower.yaxis.set_major_formatter(FuncFormatter(lambda n, _: f'{n / 1e6:g}M'))
        lower.yaxis.set_major_locator(MaxNLocator(4))
        for ax in (upper, lower):
            ax.grid(axis='y', color='#e5e7eb', linewidth=.5)
            ax.set_axisbelow(True)
            ax.spines[['right', 'top']].set_visible(False)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_xlim(-.6, len(selected) - .4)
    axes[1, 0].invert_yaxis()
    axes[0, 0].set_ylabel('Execution time (ms)', fontsize=8)
    axes[1, 0].set_ylabel('Fact tuples', fontsize=8)
    handles = [Patch(facecolor=colors[v], label=display_variant(v)) for v in variants]
    handles += [Patch(facecolor='white', edgecolor='#666', hatch='....', label='Tuple-filtered tuples')]
    fig.legend(handles=handles, loc='upper center', ncol=min(7, len(handles)), frameon=False, fontsize=8)
    fig.subplots_adjust(left=.065, right=.995, top=.78, bottom=.10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return {'dimension_panels': len(dimensions), 'query_groups': len(entries), 'runtime_bars': bars}
