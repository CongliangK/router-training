"""Export reviewed PUBLIC source into a NEW directory INSIDE the private repo.
No private data/history is copied; no model binaries are exported or loaded.
"""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
PRIVATE = HERE.parents[1]
PUBLIC = PRIVATE.parent / 'router-training-public'
SHA = 'cd058a8093977865691b867c360bfc64b4e29f88'
TARGET = HERE / 'prepared/code-candidate'

def git(*args):
    return subprocess.check_output(['git', '-C', str(PUBLIC), *args])

def main():
    if git('rev-parse', 'HEAD').decode().strip() != SHA or git('status', '--porcelain').strip():
        raise ValueError('public source binding changed')
    if TARGET.exists() or not TARGET.resolve().is_relative_to(PRIVATE):
        raise ValueError('fresh private destination required')
    entries = {}
    for entry in git('ls-tree', '-rz', SHA).split(b'\0'):
        if not entry:
            continue
        metadata, name = entry.split(b'\t', 1)
        mode, kind, object_id = metadata.decode().split()
        name = name.decode('utf-8')
        path = PurePosixPath(name)
        if kind != 'blob' or mode not in ('100644', '100755') or path.is_absolute() or '..' in path.parts:
            raise ValueError('unsupported archive entry')
        if path.suffix.lower() in ('.safetensors', '.bin', '.pt', '.pth', '.ckpt'):
            raise ValueError('model or opaque binary export refused')
        if path.suffix == '.jsonl' and not name.startswith('data/synthetic-examples/'):
            raise ValueError('nonexample data export refused')
        entries[name] = object_id
    archive_bytes = git('archive', '--format=tar', '--prefix=code-candidate/', SHA)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:') as archive:
        files = {member.name.removeprefix('code-candidate/') for member in archive.getmembers() if member.isfile()}
        if files != set(entries):
            raise ValueError('archive manifest differs from reviewed source')
        for member in archive.getmembers():
            target = HERE / 'prepared' / member.name
            if not target.resolve().is_relative_to(TARGET.resolve()):
                raise ValueError('archive path outside candidate')
            if not (member.isdir() or member.isfile()):
                raise ValueError('archive links refused')
        archive.extractall(HERE/'prepared', filter='data')
    hashes = {}
    for name, object_id in entries.items():
        data = (TARGET/name).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != object_id:
            raise ValueError('candidate blob differs from reviewed source')
        hashes[name] = hashlib.sha256(data).hexdigest()
    lock = {'public_source_sha': SHA, 'public_tree_sha': git('rev-parse', SHA+'^{tree}').decode().strip(),
            'files': len(hashes), 'sha256_by_file': hashes, 'all_git_blobs_exact': True,
            'private_history_copied': False, 'private_dataset_attached': False,
            'models_or_weights_exported': False, 'formal_training_ready': False,
            'kind': 'source-only export, NOT a Git checkout or runnable approved experiment'}
    with (HERE/'prepared/SOURCE-LOCK.json').open('x') as handle:
        json.dump(lock, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: value for key, value in lock.items() if key != 'sha256_by_file'}, indent=2))

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'ok': False, 'error_type': type(error).__name__, 'detail_withheld': True}))
        raise SystemExit(1)
