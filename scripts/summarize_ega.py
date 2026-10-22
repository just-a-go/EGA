#!/usr/bin/env python3
"""Export saved EGA metrics from runs and retained history; no metric recomputation."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    fields = ['scope', 'task', 'model', 'seed', 'run', 'seed_macro_f1', 'ega_macro_f1',
              'seed_auroc', 'ega_auroc', 'seed_brier', 'ega_brier']
    rows, counts, skipped = [], {'current': 0, 'history': 0}, 0
    for scope, base in [('current', ROOT/'runs/ega'), ('history', ROOT/'cases/history')]:
        for path in sorted(base.rglob('metrics.json')):
            metrics = read_json(path)
            if metrics is None:
                skipped += 1
                continue
            declaration = read_json(path.parent/'settings.json') or {}
            settings = declaration.get('settings', {})
            settings = settings if isinstance(settings, dict) else {}
            parts = path.relative_to(base).parts
            labels = {key: parts[i] if scope == 'current' and len(parts) >= 5
                      else settings.get(key, '') for i, key in enumerate(('task', 'model', 'seed'))}
            row = dict(labels, scope=scope, run=path.parent.relative_to(ROOT).as_posix())
            for condition in ('seed', 'ega'):
                saved = metrics.get(condition, {})
                for metric in ('macro_f1', 'auroc', 'brier'):
                    row[condition+'_'+metric] = saved.get(metric, '') if isinstance(saved, dict) else ''
            rows.append(row)
            counts[scope] += 1
    output = ROOT/'runs/summary.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} results (current={counts['current']}, history={counts['history']}; "
          f"unreadable metrics skipped={skipped}) to {output}")


if __name__ == '__main__':
    main()
