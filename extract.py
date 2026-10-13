"""Read-only extraction of task prompts; historical model selections are never exported."""
from __future__ import annotations
import argparse, hashlib, io, json, re
from collections import Counter
from pathlib import Path
import zstandard


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def redact(text: str) -> str:
    # Remove injected coordination boilerplate, not the delegated task.
    text = re.sub(r'\nYour parent agent id is [\s\S]*$', '', text)
    text = re.sub(r'/home/[^/\s]+/', '/home/USER/', text)
    text = re.sub(r"-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]", text)
    text = re.sub(r"\b(?:sk-|hf_|ghp_|github_pat_)[A-Za-z0-9_-]{12,}", "[REDACTED_TOKEN]", text)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)[\w.\-]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*)['\"]?[^\s'\";,}]{8,}", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)([?&](?:token|key|secret|password)=)[^&\s]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?<![\w])[^\s@]+@[^\s@]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    return text


def text_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")


def rows(path: Path):
    if path.name.endswith('.zstd'):
        with path.open('rb') as source, zstandard.ZstdDecompressor().stream_reader(source) as reader:
            for line in io.TextIOWrapper(reader):
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open() as reader:
            for line in reader:
                if line.strip():
                    yield json.loads(line)


def extract(manifest: Path, output: Path):
    candidates = [Path(line) for line in manifest.read_text().splitlines() if line.strip()]
    latest = {}
    for path in candidates:
        match = re.search(r'session(?:\.v(\d+))?\.jsonl', path.name)
        if not match:
            continue
        version = int(match.group(1) or 0)
        if path.parent not in latest or version > latest[path.parent][0]:
            latest[path.parent] = (version, path)
    headers = {}; events = {}; errors = []
    for _, path in latest.values():
        try:
            log = list(rows(path)); header = log[0]
            assert header.get('type') == 'session', 'unrecognized header'
            assert header.get('version') in [3, 4], 'unsupported historical schema'
            identity = header['id']; headers[identity] = header
            # A fork's inherited history is not a newly issued delegation.
            seed_cut = max((index for index, event in enumerate(log) if event.get('type') == 'session/end-seed'), default=0)
            events[identity] = log[seed_cut + 1:]
        except Exception as error:
            errors.append({'source': digest(str(path)), 'error': type(error).__name__})
    def root_id(identity):
        seen = set()
        while identity not in seen:
            seen.add(identity)
            parent = headers.get(identity, {}).get('parentSession')
            if not parent:
                break
            identity = parent
        return digest(identity)
    exported = []; counters = Counter()
    for identity, log in events.items():
        project = digest(headers[identity].get('cwd', '_no-cwd'))
        for event in log:
            if event.get('type') != 'tool/call':
                continue
            data = event.get('data', {}); name = data.get('name', '')
            if not (name.startswith('subagent') or name == 'spawn_teammate'):
                continue
            args = data.get('arguments', {})
            try:
                if isinstance(args, str):
                    args = json.loads(args)
                prompt = args.get('prompt')
                if not isinstance(prompt, str) or len(prompt.strip()) < 15:
                    continue
                prompt = redact(prompt.strip())
                exported.append({'id': digest(identity + ':' + str(data.get('callId', event.get('seq')))),
                    'prompt': prompt, 'description': redact(str(args.get('description', ''))),
                    'group': root_id(identity), 'project': project,
                    'provenance': {'kind': 'delegation_prompt', 'session': digest(identity), 'call': digest(str(data.get('callId', ''))), 'tool': name}})
                counters[name] += 1
            except (ValueError, TypeError, AttributeError):
                counters['invalid_arguments'] += 1
        # Workflow children do not persist their prompt in the parent agent-start event.
        # Recover only the first live user prompt, never responses or descriptors.
        if headers[identity].get('origin') == 'subagent':
            for event in log:
                if event.get('type') != 'user/message':
                    continue
                prompt = redact(text_content(event.get('data', {}).get('content')).strip())
                if len(prompt) >= 15:
                    exported.append({'id': digest(identity + ':initial-prompt'), 'prompt': prompt, 'description': '',
                        'group': root_id(identity), 'project': project,
                        'provenance': {'kind': 'child_initial_prompt', 'session': digest(identity)}})
                    counters['child_initial_prompt'] += 1
                break
    # Exact duplicate prompts always share a split group, including separate roots.
    normalized = {}; unique = []
    for record in exported:
        key = digest(re.sub(r'\s+', ' ', record['prompt']))
        if key in normalized:
            counters['exact_duplicates'] += 1
            continue
        normalized[key] = record
        record['prompt_hash'] = key
        unique.append(record)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'tasks.jsonl'
    path.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in unique))
    report = {'physical_files': len(candidates), 'selected_sessions': len(latest), 'read_sessions': len(headers),
        'tasks_before_dedup': len(exported), 'unique_tasks': len(unique), 'groups': len(set(row['group'] for row in unique)),
        'counts': dict(counters), 'errors': errors, 'historical_model_fields_exported': False,
        'privacy': 'local only; credential-pattern redaction is not a guarantee; review before sharing'}
    (output / 'extraction-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--manifest', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); extract(args.manifest, args.output)
