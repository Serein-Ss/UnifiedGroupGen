"""Standalone, read-only terminal panel for existing training logs (stdlib only)."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


class JsonlTail:
    """Read a bounded suffix once, then only appended, newline-complete records."""
    def __init__(self, path, steps=False, window=262144):
        self.path, self.steps, self.window = Path(path), steps, window
        self.identity, self.offset = None, 0
        self.reset, self.invalid = False, 0

    def _read(self, stream, start, size):
        if start:
            stream.seek(start - 1)
            if stream.read(1) != b'\n':
                stream.readline()  # Discard a leading partial record.
                start = stream.tell()
        stream.seek(start)
        data = stream.read(max(0, size - start))
        end = data.rfind(b'\n') + 1
        rows, invalid = [], 0
        for line in data[:end].splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    invalid += 1
            except (ValueError, UnicodeDecodeError):
                invalid += 1
        return rows, start + end, invalid

    def poll(self):
        self.reset = False
        try:
            with self.path.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                self.reset = identity != self.identity or stat.st_size < self.offset
                if self.reset:
                    window = self.window
                    while True:
                        start = max(0, stat.st_size - window)
                        rows, offset, invalid = self._read(stream, start, stat.st_size)
                        last = rows[-1] if rows else {}
                        # Opening during validation must also recover the frozen
                        # Train row, even when validation has thousands of batches.
                        need_train = (self.steps and last.get('phase') == 'validation'
                                      and last.get('event') == 'step'
                                      and not any(r.get('phase') == 'train'
                                                  and r.get('event') == 'step'
                                                  and r.get('epoch') == last.get('epoch') for r in rows))
                        if not start or (rows and not need_train):
                            break
                        window *= 2
                else:
                    rows, offset, invalid = self._read(stream, self.offset, stat.st_size)
                self.identity, self.offset = identity, offset
                self.invalid += invalid
                return rows
        except FileNotFoundError:
            return []


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stamp(row):
    if finite(row.get('updated')):
        return row['updated']
    try:
        return datetime.fromisoformat(row['time']).timestamp()
    except (KeyError, ValueError, TypeError):
        return 0


def metric(values, phase, key):
    name = phase + '/' + key
    return values.get(name + '_epoch', values.get(name))


class RunReader:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.steps = JsonlTail(self.directory / 'model.steps.jsonl', steps=True)
        self.history = JsonlTail(self.directory / 'model.history.jsonl')
        self.status, self.latest, self.batches, self.end, self.epoch_history = {}, {}, {}, {}, {}

    def poll(self):
        try:
            value = json.loads((self.directory / 'model.status.json').read_text(encoding='utf-8'))
            if isinstance(value, dict):
                self.status = value
        except (FileNotFoundError, ValueError):
            pass  # An old/incomplete status file does not hide the step log.
        history = self.history.poll()
        if self.history.reset:
            self.epoch_history = {}
        if history:
            self.epoch_history = history[-1]
        rows = self.steps.poll()
        if self.steps.reset:
            self.latest, self.batches, self.end = {}, {}, {}
        for row in rows:
            event = row.get('event')
            if event == 'start' or (row.get('epoch') is not None
                                    and row.get('epoch') != self.latest.get('epoch')
                                    and event not in ('complete', 'failed', 'interrupted')):
                self.batches, self.end = {}, {}
            self.latest = row
            if event == 'step' and row.get('phase') in ('train', 'validation'):
                self.batches[row['phase']] = row
            elif event == 'epoch_end':
                self.end = row
        return self.snapshot()

    def snapshot(self):
        status, last, history = self.status, self.latest, self.epoch_history
        newest = last if stamp(last) >= stamp(status) and last else status
        event = newest.get('event')
        phase = 'epoch_end' if event == 'epoch_end' else newest.get('phase', 'waiting')
        if event in ('complete', 'failed', 'interrupted'):
            phase = event
        elif newest is status and status.get('state') in ('complete', 'failed', 'interrupted'):
            phase = status['state']
        epoch = newest.get('epoch')
        if epoch is None and phase not in ('preparing_targets', 'building_model', 'waiting'):
            epoch = last.get('epoch')
        if phase == 'complete':
            epoch = (status['epochs_completed'] - 1 if finite(status.get('epochs_completed'))
                     else self.end.get('epoch', self.batches.get('train', {}).get('epoch', epoch)))
        summary = self.end if self.end and self.end.get('epoch', -1) >= history.get('epoch', -1) else history
        early = summary.get('early_stopping') or {}
        best = summary.get('best_validation_loss', early.get('best_score', status.get('best_validation_loss')))
        rates = history.get('learning_rates_used') or []
        lr = last.get('lr', self.batches.get('train', {}).get('lr', summary.get('lr', rates[0] if rates else None)))
        means = summary.get('metrics', {})
        ended = phase in ('epoch_end', 'complete', 'finished')
        train, validation = dict(self.batches.get('train', {})), dict(self.batches.get('validation', {}))
        if ended:
            for name, row, key in (('train', train, 'train_seconds'), ('val', validation, 'validation_seconds')):
                parts = summary.get('loss_parts', {}).get(name, {})
                for part in ('loss', 'continuous', 'element', 'jump'):
                    row[part] = (summary.get('loss' if name == 'train' else 'validation_loss')
                                 if part == 'loss' and 'loss_parts' in summary else parts.get(part, metric(means, name, part)))
                if key in summary:
                    row['phase_seconds'] = summary[key]
        # Status-only fallback: never fabricate within-epoch counts or timings.
        if not train and phase in ('training', 'train'):
            train['loss'] = status.get('loss')
        total_loss = summary.get('loss', metric(means, 'train', 'loss')) if ended else None
        val_loss = summary.get('validation_loss', metric(means, 'val', 'loss')) if ended else None
        latest_time = max(stamp(status), stamp(last))
        return dict(name=self.directory.parent.name, path=str(self.directory), phase=phase,
                    epoch=epoch, max_epochs=status.get('max_epochs'),
                    batch_size=status.get('batch_size', last.get('batch_size')),
                    global_step=newest.get('global_step'), best=best, early=early, lr=lr,
                    loss=total_loss, val=val_loss, train=train, validation=validation,
                    updated=latest_time, invalid=self.steps.invalid + self.history.invalid)


def number(value):
    if not finite(value):
        return '-'
    return f'{value:.4f}' if abs(value) < 10000 else f'{value:.2e}'


def duration(seconds):
    if not finite(seconds) or seconds < 0:
        return '-'
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


def phase_lines(label, row, width):
    n, total, seconds = (row.get(k) for k in ('batch', 'batches', 'phase_seconds'))
    known = finite(n) and finite(total) and total > 0
    ratio = min(1., max(0., n / total)) if known else 0
    bar = '#' * int(ratio * 10) + '-' * (10 - int(ratio * 10))
    percent = f'{ratio:4.0%}' if known else '   -'
    count = f'{int(n):5}/{int(total):<5}' if known else '    -/-    '
    speed = n / seconds if known and finite(seconds) and seconds > 0 else None
    eta = max(0, total - n) / speed if speed else None
    main = (f'{label:<8} | {percent} [{bar}] | {count} | time={duration(seconds):>8}'
            f' | ETA={duration(eta):>8} | {(f"{speed:.2f}" if speed else "-"):>6}it/s')
    parts = ' | '.join(f'{key}={number(row.get(source)):>7}' for key, source in
                       (('loss', 'loss'), ('geom', 'continuous'), ('elem', 'element'), ('jump', 'jump')))
    return [main + ' | ' + parts] if len(main) + len(parts) + 3 <= width else [main, ' ' * 8 + ' | ' + parts]


def render(snapshots, width=180, now=None):
    now = time.time() if now is None else now
    lines = []
    for s in snapshots:
        age = f'{max(0, now - s["updated"]):.0f}s' if s['updated'] else '-'
        lines.append(f'{s["name"]} | stage={s["phase"]} | log_age={age}')
        epoch = s['epoch'] + 1 if finite(s['epoch']) else '-'
        if s['phase'] == 'complete' and finite(s['max_epochs']) and finite(epoch):
            epoch = min(epoch, s['max_epochs'])
        early = s['early']
        wait = f'{early.get("wait_count", "-")}/{early.get("patience", "-")}' if early else '-'
        header = f'Epoch {epoch}/{s["max_epochs"] or "-"} (bs={s["batch_size"] or "-"})'
        lr = f'{s["lr"]:.2g}' if finite(s['lr']) else '-'
        fields = (f'step={s["global_step"] if s["global_step"] is not None else "-"}'
                  f' | loss={number(s["loss"])} | val={number(s["val"])}'
                  f' | best={number(s["best"])} | early={wait} | lr={lr}')
        lines.extend([header + ' | ' + fields] if len(header) + len(fields) + 3 <= width else [header, fields])
        lines.extend(phase_lines('Train', s['train'], width))
        if s['phase'] == 'validation':
            lines.extend(phase_lines('Validate', s['validation'], width))
        if s['invalid']:
            lines.append(f'Warning: skipped {s["invalid"]} malformed complete JSONL records')
        lines.append('')
    lines.append('Read-only log observations; not scheduler status. Ctrl-C exits this panel only.')
    return '\n'.join(lines)


def watch(readers, interval=2., once=False):
    terminal = bool(getattr(sys.stdout, 'isatty', lambda: False)())
    if not once and not terminal:
        raise ValueError('Live mode needs an interactive terminal; use --once for plain text.')
    try:
        if not once:
            sys.stdout.write('\x1b[?1049h\x1b[?25l')
        while True:
            snapshots = [reader.poll() for reader in readers]
            display = render(snapshots, shutil.get_terminal_size((180, 40)).columns)
            sys.stdout.write(('' if once else '\x1b[H\x1b[J') + display + '\n')
            sys.stdout.flush()
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        if not once:
            sys.stdout.write('\x1b[?25h\x1b[?1049l')
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='Existing output run directories, DNG then CSP')
    parser.add_argument('--interval', type=float, default=2., help='Refresh interval in seconds (default: 2)')
    parser.add_argument('--once', action='store_true', help='Print one plain-text snapshot without ANSI codes')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < .5:
        parser.error('--interval must be finite and at least 0.5 seconds')
    if any(not p.is_dir() for p in args.runs):
        parser.error('Every run must be an existing directory; this viewer creates no files')
    try:
        watch([RunReader(p) for p in args.runs], args.interval, args.once)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Panel error: {error}\n')


if __name__ == '__main__':
    main()
