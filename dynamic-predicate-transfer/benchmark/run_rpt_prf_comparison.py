import argparse
import csv
import pathlib
import re
import statistics
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'build' / 'rpt-prf-performance'
OUT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('stage')
parser.add_argument('--rounds', type=int, default=2)
parser.add_argument('--scan-comparison', action='store_true')
args = parser.parse_args()
records = []
for repeat in range(args.rounds):
    modes = ('cascade', 'prf-operator', 'prf-scan', 'none') if args.scan_comparison else ('cascade', 'prf', 'none')
    if repeat % 2:
        modes = tuple(reversed(modes))
    for mode in modes:
        prefix = OUT / f'{args.stage}-{repeat}-{mode}'
        if prefix.with_suffix('.log').exists():
            raise RuntimeError(f'Refusing to overwrite {prefix}')
        print(f'Starting {args.stage} round {repeat + 1}: {mode}', flush=True)
        command = [str(ROOT / 'build/release/benchmark/benchmark_runner'),
                   'benchmark/imdb/.*.benchmark', '--threads=8', '--rpt_filter', mode.split('-')[0],
                   '--rpt_prf_scan_pushdown', 'false' if mode == 'prf-operator' else 'true',
                   '--disable-timeout']
        run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        log = run.stdout + '\n' + run.stderr
        prefix.with_suffix('.log').write_text(log)
        matches = re.findall(r'^benchmark/imdb/(\d+[a-z])\.benchmark\t(\d+)\t([0-9.]+)$', log, re.M)
        if run.returncode or len(matches) != 565 or re.search(r'\b(INCORRECT|ERROR|TIMEOUT)\b', log):
            raise RuntimeError(f'Benchmark failed or incomplete: {prefix}.log ({len(matches)} samples)')
        assert len({q for q, _, _ in matches}) == 113
        for q in {q for q, _, _ in matches}:
            assert sorted(int(i) for name, i, _ in matches if name == q) == [1, 2, 3, 4, 5]
        for query, iteration, timing in matches:
            records.append([args.stage, repeat, mode, query, int(iteration), float(timing)])
        with (OUT / f'{args.stage}.csv').open('w') as dest:
            writer = csv.writer(dest)
            writer.writerow(['stage', 'round', 'mode', 'query', 'iteration', 'seconds'])
            writer.writerows(records)
        total = sum(float(t) for _, _, t in matches)
        medians = sum(statistics.median(float(t) for name, _, t in matches if name == q)
                      for q in {q for q, _, _ in matches})
        print(f'{mode}: 565 verified samples; sum={total:.6f}s; sum of query medians={medians:.6f}s', flush=True)
