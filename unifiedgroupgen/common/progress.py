"""TTY progress bars and buffered JSONL observations; no model or RNG state."""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import time

from tqdm.auto import tqdm


BAR_PREFIX = '{desc:<32.32} | {percentage:3.0f}% |{bar:18}| {n_fmt:>6}/{total_fmt:<6}'
BAR_FORMAT = BAR_PREFIX+' | time={elapsed:>8} | ETA={remaining:>8} | {rate_fmt:>20}{fields}'


def format_metrics(values):
    widths = dict(loss=8, val=8, geom=8, elem=8, jump=8, best=8, early=7, lr=8, t=5, atoms=5, accepted=6, failed=6,
                  rejected=6, stage=24, step=8, age=8)
    return ''.join(f' | {key}={str(value):>{widths.get(key, 0)}}' for key, value in values.items())


class PhaseClock:
    """Monotonic active-stage time; paused intervals never count."""
    def __init__(self):
        self.reset(running=False)

    def reset(self, running=True):
        self._seconds=0.
        self._started=time.monotonic() if running else None

    def pause(self):
        if self._started is not None:
            self._seconds+=time.monotonic()-self._started
            self._started=None

    def resume(self):
        if self._started is None:
            self._started=time.monotonic()

    @property
    def elapsed(self):
        return self._seconds+(time.monotonic()-self._started if self._started is not None else 0.)


class AlignedTqdm(tqdm):
    """Use the same columns and pipe-separated fields for every progress bar."""
    def __init__(self, *args, phase_timing=False, **kwargs):
        self._fields = ''
        self.phase_clock=PhaseClock() if phase_timing else None
        if self.phase_clock is not None:
            self.phase_clock.reset()
        kwargs.setdefault('bar_format', BAR_FORMAT)
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        values = super().format_dict
        values['fields'] = self._fields
        if self.phase_clock is not None:
            elapsed=self.phase_clock.elapsed
            count=self.n-getattr(self,'initial',0)
            values['elapsed']=elapsed
            values['rate']=count/elapsed if elapsed>0 and count>0 else None
        return values

    def reset(self, total=None):
        if self.phase_clock is not None:
            self.phase_clock.reset()
        super().reset(total)

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        self._fields = format_metrics({**(ordered_dict or {}), **kwargs})
        if refresh:
            self.refresh()


def interactive():
    return bool(getattr(sys.stdout, 'isatty', lambda: False)())


def scalar(value):
    if value is None:
        return None
    value = float(value.detach() if hasattr(value, 'detach') else value)
    return value if math.isfinite(value) else None


class StepLog:
    """Append-only step records, flushed at least once per second and on close."""
    def __init__(self, path):
        self.path = Path(path) if path is not None else None
        self.stream = None
        self.flushed = time.monotonic()

    def write(self, phase, event='step', **values):
        if self.path is None:
            return
        if self.stream is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open('a', encoding='utf-8')
        row = dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='milliseconds'),
                   phase=phase, event=event, **values)
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        if event in ('start', 'complete', 'failed', 'interrupted', 'epoch_end') or time.monotonic()-self.flushed >= 1:
            self.stream.flush()
            self.flushed = time.monotonic()

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class Progress:
    def __init__(self, description, total, *, log=None, unit='it', initial=0):
        self.description, self.total, self.completed = description, total, initial
        self.unit = unit
        self.log = StepLog(log)
        self.started = time.monotonic()
        self.last_refresh = 0.
        self.metrics = {}
        self.bar = None

    def __enter__(self):
        self.bar = AlignedTqdm(total=self.total, initial=self.completed, desc=self.description,
                        unit=' '+self.unit, file=sys.stdout, disable=not interactive(),
                        dynamic_ncols=True, leave=False, mininterval=.2)
        self.log.write(self.description, 'start', completed=self.completed, total=self.total)
        return self

    def observe(self, **values):
        """Record an inner sampling step while the completed-structure count stays fixed."""
        self.log.write(self.description, completed=self.completed, total=self.total, **values)
        visible = {k: v for k, v in values.items() if k in ('t', 'atoms', 'accepted', 'failed')}
        if visible:
            self.metrics.update(visible)
            self.bar.set_postfix(self.metrics, refresh=False)
            if time.monotonic()-self.last_refresh >= .2:
                self.bar.refresh()
                self.last_refresh = time.monotonic()

    def update(self, amount=1, **metrics):
        self.completed += amount
        self.metrics.update(metrics)
        self.bar.set_postfix(self.metrics, refresh=False)
        self.bar.update(amount)
        self.log.write(self.description, completed=self.completed, total=self.total,
                       elapsed_seconds=time.monotonic()-self.started, **metrics)

    def __exit__(self, kind, error, traceback):
        state = ('interrupted' if kind and issubclass(kind, (KeyboardInterrupt, SystemExit, GeneratorExit))
                 else 'failed' if kind else 'complete')
        try:
            self.log.write(self.description, state, completed=self.completed, total=self.total,
                           elapsed_seconds=time.monotonic()-self.started,
                           **({'error': str(error)} if error is not None else {}))
        finally:
            self.log.close()
            self.bar.close()
        details = format_metrics(self.metrics)
        if kind is not GeneratorExit:
            tqdm.write(f'{self.description:<32.32} | {self.completed:>6}/{self.total:<6} | {state:<11} '
                       f'| time={time.monotonic()-self.started:.1f}s'+details, file=sys.stdout)


def track(items, description, *, total=None, log=None, unit='it', initial=0):
    """Advance after each consumed item, including handled failures and continues."""
    total = len(items)+initial if total is None else total
    with Progress(description, total, log=log, unit=unit, initial=initial) as progress:
        for item in items:
            yield item
            progress.update()


def watch(paths):
    """Read existing status files; Ctrl-C stops this viewer, never a training job."""
    if not interactive():
        raise ValueError('The read-only progress viewer requires an interactive terminal')
    paths = [Path(path) for path in paths]
    bars = [AlignedTqdm(total=0, desc=path.parent.parent.name, position=i, file=sys.stdout,
                 dynamic_ncols=True, unit='(bs=-) epochs',bar_format=BAR_PREFIX+' {unit}{fields}') for i, path in enumerate(paths)]
    try:
        while True:
            finished = []
            for path, bar in zip(paths, bars):
                try:
                    state = json.loads(path.read_text(encoding='utf-8'))
                except (FileNotFoundError, json.JSONDecodeError):
                    bar.set_postfix(stage='waiting for status', refresh=True)
                    finished.append(False)
                    continue
                done = state.get('state') in ('complete', 'failed', 'interrupted')
                bar.total = state.get('max_epochs') or 0
                bar.n = state.get('epochs_completed', state.get('epoch')) or 0
                bar.unit = f'(bs={state.get("batch_size", "-")}) epochs'
                loss = scalar(state.get('loss')); val = scalar(state.get('validation_loss'))
                age = max(0, time.time()-state['updated']) if state.get('updated') else None
                bar.set_postfix(stage=state.get('state') if done else state.get('phase', 'waiting'),
                    step=str(state.get('global_step', '-')),
                    loss=f'{loss:.4f}' if loss is not None else '-',
                    val=f'{val:.4f}' if val is not None else '-',
                    age=f'{age:.0f}s' if age is not None else '-', refresh=True)
                finished.append(done)
            if all(finished):
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        for bar in bars:
            bar.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read-only training progress from model.status.json files')
    parser.add_argument('status', nargs='+', help='One or more existing model.status.json paths')
    watch(parser.parse_args().status)
