#!/bin/bash
# One-time alignment of the already-uploaded TH-HPC6 checkout.
set -e

cd /fs6/home/zhongrsh/workspace/UnifiedGroupGen
baseline=6db528627b125786aef4288b15728dfefe4eee86
test "$(git symbolic-ref --short HEAD)" = main
if [ "$(git rev-parse HEAD)" != 9d8286704eaa3eb44e25ddac531212639b3a506c ]; then
    echo "This script only applies to the original uploaded 9d828670 checkout."
    exit 1
fi
git cat-file -e "$baseline^{commit}"
git merge-base --is-ancestor "$baseline" origin/main

paths=(
    .gitattributes
    .gitignore
    README.md
    configs/paths.yaml
    data/dataset/README.md
    unifiedgroupgen/common/configuration.py
    unifiedgroupgen/tests/legacy/paths.py
    unifiedgroupgen/tests/unit/test_main15_controls.py
    unifiedgroupgen/tests/unit/test_package_layout.py
    unifiedgroupgen/tests/unit/test_result_storage.py
    unifiedgroupgen/tests/unit/test_training.py
)

backup=$(mktemp -d /fs6/home/zhongrsh/workspace/UnifiedGroupGen_git_backup.XXXXXX)
echo "Backup directory: $backup"
cp -a .git "$backup/git"
tar -czf "$backup/files_before_sync.tar.gz" -- "${paths[@]}"

# Mixed reset updates HEAD/index while preserving every working-tree file.
git reset --mixed --quiet "$baseline"
# Only these 11 audited files are replaced; their previous bytes are backed up.
git restore --source="$baseline" --worktree -- "${paths[@]}"
git pull --ff-only origin main

echo "Alignment complete."
git rev-parse HEAD
git status --short --branch --untracked-files=no
