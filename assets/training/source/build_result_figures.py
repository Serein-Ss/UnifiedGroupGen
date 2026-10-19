"""Plot only the verified current joint snapshot; export PNG and source CSV."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'assets/training'


def main():
    manifest = json.loads((ROOT/'assets/results/current_snapshot.json').read_text(encoding='utf-8'))
    history = json.loads((ROOT/manifest['history']).read_text(encoding='utf-8'))
    validated = [r for r in history if r.get('val_loss') is not None]
    best = manifest['best_epoch_1based']
    completed = manifest['training']['completed_epochs']
    assert len(history) == completed
    assert min(validated, key=lambda r: r['val_loss'])['epoch']+1 == best
    fields = ['epoch', 'train_descriptor', 'train_flow', 'validated', 'val_descriptor', 'val_flow', 'val_total', 'learning_rate', 'seconds']
    with (OUT/'joint_training_history.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(fields)
        for row in history:
            val = row.get('val') or {}
            writer.writerow([row['epoch']+1, row['train']['descriptor'], row['train']['flow'],
                bool(val), val.get('descriptor', ''), val.get('flow', ''), row['val_loss'] if val else '',
                row['learning_rate'], row['seconds']])
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False, 'legend.frameon':False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    epochs = [r['epoch']+1 for r in history]
    for axis, key, title in zip(axes, ['total', 'descriptor', 'flow'],
            ['a  Joint training objective', 'b  Descriptor Transformer', 'c  OrbitFlow']):
        train = [r['train']['descriptor']+r['train']['flow'] if key == 'total' else r['train'][key] for r in history]
        values = [r['val_loss'] if key == 'total' else r['val'][key] for r in validated]
        axis.plot(epochs, train, color='#437da6', lw=1, label='Train')
        axis.plot([r['epoch']+1 for r in validated], values, color='#c57b40', marker='o', ms=2.3, lw=1.2, label='Validation')
        selected = next(i for i,r in enumerate(validated) if r['epoch']+1 == best)
        axis.scatter([best], [values[selected]], marker='*', s=85, color='#8b3434', zorder=4)
        axis.axvline(best, color='#8b3434', ls='--', lw=.8)
        axis.axvline(completed, color='#626262', ls=':', lw=.8)
        axis.set(title=title, xlabel='Completed epoch (1-based)', ylabel='Loss', xlim=(0, completed+5))
        axis.grid(axis='y', color='#e3e8eb', lw=.5)
    axes[0].legend(loc='center right', fontsize=9)
    fig.suptitle(f'Joint SG/LG model: early stopped at {completed}; {len(validated)} validation points; best at {best}', fontsize=12)
    fig.savefig(OUT/'training_history.png', dpi=260, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    (OUT/'README.md').write_text(f'# Current joint training history\n\nVerified snapshot: `{manifest["snapshot"]}`. Training completed by early stopping at {completed} epochs, with {len(validated)} actual validation observations. Best checkpoint: epoch {best}, loss {manifest["training"]["best_val_loss"]:.9f}.\n\n![Training history](training_history.png)\n\n[Source CSV](joint_training_history.csv). No smoothing or zero-filled missing validation. The best marker follows the joint objective, not each component minimum. Training and validation use different conditioning mechanisms; absolute losses are not material-quality metrics. Older single-source figures and PDF/SVG versions are archived outside the current assets.\n', encoding='utf-8')
    print(f'Rebuilt {completed} epochs / {len(validated)} validation observations')


if __name__ == '__main__':
    main()
