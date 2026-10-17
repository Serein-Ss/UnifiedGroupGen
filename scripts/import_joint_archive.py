"""Import and verify joint evaluation artifacts from a server ZIP, without replacing local code."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.cli import artifact_path
from unifiedgroupgen.training import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', default='artifacts/imported_server/joint_best_seed142')
    parser.add_argument('--checkpoint-output', default='artifacts/imported_server/checkpoints/joint_best.pt')
    args = parser.parse_args()
    archive, output, checkpoint = Path(args.archive), artifact_path(args.output), artifact_path(args.checkpoint_output)
    if output.exists() or checkpoint.exists():
        raise ValueError('Use new import/checkpoint paths; original and previous evidence is immutable')
    output.mkdir(parents=True)
    with ZipFile(archive) as z:
        matches = [n for n in z.namelist() if n.endswith('artifacts/evaluations/joint_best_seed142/pipeline.json')]
        if len(matches) != 1:
            raise ValueError('Archive must contain exactly one requested joint evaluation')
        prefix = matches[0][:-len('pipeline.json')]
        state = json.loads(z.read(matches[0]))
        if state['state'] != 'complete':
            raise ValueError('The archived joint evaluation is not complete')
        written = 0
        for item in z.infolist():
            if not item.filename.startswith(prefix) or item.is_dir():
                continue
            relative = PurePosixPath(item.filename[len(prefix):])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe path in selected archive entries')
            target = output.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError('Selected archive path escapes the import directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(item))  # ZipFile verifies each selected entry's CRC.
            written += 1
        for name, job in state['jobs'].items():
            for filename, checksum in job['output_sha256'].items():
                candidate = (output/job['directory']/filename).resolve()
                if not candidate.is_relative_to(output.resolve()):
                    raise ValueError('Evaluation manifest path escapes the import directory')
                if file_hash(candidate) != checksum:
                    raise ValueError(f'Generation artifact fingerprint mismatch: {name}/{filename}')
        checkpoints = [n for n in z.namelist() if n.endswith('artifacts/runs/joint_space_layer_large_seed42/model.best.pt')]
        if len(checkpoints) != 1:
            raise ValueError('Expected one actual shared best checkpoint')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with z.open(checkpoints[0]) as src, checkpoint.open('wb') as dst:
            while block := src.read(1024*1024):
                dst.write(block)
        if file_hash(checkpoint) != state['identity']['checkpoint_sha256']:
            raise ValueError('Archived best checkpoint differs from the generation identity')
        root_prefix = prefix[:-len('artifacts/evaluations/joint_best_seed142/')]
        code_matches = {name: hashlib.sha256(z.read(root_prefix+name)).hexdigest() == checksum
                        for name, checksum in state['identity']['code_sha256'].items()}
    report = {'archive_sha256': file_hash(archive), 'files_imported': written,
        'pipeline_sha256': file_hash(output/'pipeline.json'), 'checkpoint_sha256': file_hash(checkpoint),
        'archived_code_matches_generation': code_matches, 'generation_outputs_verified': True,
        'scope': 'Only evaluation artifacts and the bound checkpoint imported. Local source and training data retained.'}
    if not all(code_matches.values()):
        raise ValueError('Some archived code differs from the code used at generation time')
    (output/'import_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'verified': True, 'files': written, 'checkpoint': str(checkpoint), 'output': str(output)}))


if __name__ == '__main__':
    main()
