"""Generated test data only: no historical prompts, labels, or split fingerprints."""
import contextlib
import io
import json
from pathlib import Path
import tempfile

from prepare import prepare, load

SPLIT_NAMES = ('train', 'calibration', 'test')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def synthetic_splits():
    # Twelve independent groups, ten rows each. Distinct deterministic strings
    # keep accidental near-duplicate merging out of this count/isolation fixture.
    import random
    rng = random.Random(10)
    tasks = [{'id':f'fixture-{i}', 'group':f'fixture-group-{i//10}',
              'prompt':''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=100)),
              'synthetic':True} for i in range(120)]
    labels = [{'id':row['id'], 'tier':('fast','balanced','flagship')[i%3],
               'task_type':'implementation', 'needs_vision':('yes','no')[i%2],
               'label_source':'expert_provisional', 'evidence':'prompt_only',
               'confidence':1., 'rationale':'Synthetic fixture label; not measured evidence.'}
              for i,row in enumerate(tasks)]
    with tempfile.TemporaryDirectory(prefix='router-synthetic-') as directory:
        root = Path(directory)
        write_rows(root/'tasks.jsonl', tasks)
        for n in range(1,5):write_rows(root/f'expert-recovery-{n}.jsonl', labels[(n-1)*30:n*30])
        with contextlib.redirect_stdout(io.StringIO()):prepare(root)
        splits = {name:load(root/f'{name}.jsonl') for name in SPLIT_NAMES}
        report = json.loads((root/'split-report.json').read_text())
    return tasks, splits, report


def supplemental_rows():
    return [{'id':f'supplement-{i}', 'prompt':f'Synthetic supplemental exercise {i}.',
             'synthetic':True} for i in range(200)]
