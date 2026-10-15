#!/usr/bin/env python3
"""CPU-only aggregate report. Optional JSONL input must contain fixed test scores.

Outputs only numeric aggregates and local group/root aliases. No private row,
prompt, label record, source identifier, or path is echoed in the report.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from split_reporting import split_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-report',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,help='explicit test JSONL labels and optional fixed answers, with split_group; optional root field: group')
    parser.add_argument('--output',type=Path,help='new output path; existing paths are never overwritten (default: stdout)')
    args=parser.parse_args()
    source_bytes=args.split_report.read_bytes();rows=None
    inputs={'split_report_sha256':hashlib.sha256(source_bytes).hexdigest()}
    if args.predictions:
        data=args.predictions.read_bytes();inputs['prediction_input_sha256']=hashlib.sha256(data).hexdigest()
        rows=[json.loads(line) for line in data.splitlines() if line.strip()]
    report=split_report(json.loads(source_bytes),rows);report['input_digests']=inputs
    text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:stream.write(text)
    else:print(text,end='')


if __name__=='__main__':main()
