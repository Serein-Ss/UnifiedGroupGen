"""One-shot, read-only inspection of a local generation/evaluation campaign."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import psutil


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def inspect(output, verify=False):
    output=Path(output).resolve()
    if not (output/'pipeline.json').exists():
        submitted=read(output/'submitted.json')
        print('Initial checkpoint/data verification; task status has not been written yet.')
        print('Submitted:',submitted['submitted_at'],'| PID:',submitted['pid'],
              '| process exists:',psutil.pid_exists(submitted['pid']))
        log=output/'driver.stderr.log'
        if log.exists(): print(log.read_text(encoding='utf-8',errors='replace')[-2000:])
        if verify: raise ValueError('Campaign has not started its task list; no complete outputs to verify')
        return
    state=read(output/'pipeline.json')
    jobs=state['identity']['jobs']; entries={r['name']:r for r in state['jobs']}
    pid=state['pid']; alive=psutil.pid_exists(pid)
    print(time.strftime('%Y-%m-%d %H:%M:%S'),output)
    print('State:',state['state'],'| PID:',pid,'| alive:',alive,'| phase:',state.get('phase'))
    print('Current:',state.get('current'),'| status age seconds:',round(time.time()-state.get('updated',state['started_at']),1))
    print('Job states:',dict(Counter(entries.get(j['model']+'/'+j['name'],{}).get('state','pending') for j in jobs)))
    if alive:
        try:
            process=psutil.Process(pid)
            for child in [process,*process.children(recursive=True)]:
                try: print('Process:',child.pid,'RSS MiB:',round(child.memory_info().rss/2**20,1),' '.join(child.cmdline()))
                except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        except psutil.NoSuchProcess: pass
    rates=defaultdict(list); snapshots={}; counts=Counter()
    for job in jobs:
        key=job['model']+'/'+job['name']; folder=output/key
        if job['task']=='components': continue
        requested=len(job['case_ids'])*job['k'] if job['task']=='csp' else job['count']
        path=folder/'sampling.status.json'; s=read(path) if path.exists() else {}
        attempted=s.get('attempted',0); reused=s.get('reused',0)
        seconds=s.get('sampling_seconds',0)
        snapshots[key]=(job,requested,attempted,reused,seconds)
        counts.update(requested=requested,attempted=attempted,reused=reused,generated=s.get('generated',0))
        if attempted>reused and seconds>0:
            rates[(job['kind'],bool(job.get('conditional')))].append(seconds/(attempted-reused))
        if key==state.get('current'):
            print('Sampling:',s)
            for name in ('inference.log','components.log','metrics.log'):
                log=folder/name
                if log.exists():
                    print('Log:',name,'age seconds:',round(time.time()-log.stat().st_mtime,1))
                    with log.open(encoding='utf-8',errors='replace') as stream:
                        from collections import deque
                        print(''.join(deque(stream,maxlen=3)).rstrip())
    print('Structure counts (attempts include failures):',dict(counts))
    seconds=0.; unknown=0
    for key,(job,requested,attempted,reused,elapsed) in snapshots.items():
        remaining=requested-attempted
        if remaining<=0: continue
        rate=elapsed/(attempted-reused) if attempted>reused and elapsed else None
        candidates=rates.get((job['kind'],bool(job.get('conditional'))),[])
        rate=rate or (statistics.median(candidates) if candidates else None)
        if rate: seconds+=remaining*rate
        else: unknown+=1
    metric_times=[e['evaluation_seconds'] for e in entries.values() if e.get('evaluation_seconds') and
                  not e.get('reused_component_report') and '/components' not in e['name']]
    print('Remaining generation with measured rates: %.2f h; unmeasured batches: %d'%(seconds/3600,unknown))
    print('Completed metric batches:',len(metric_times),'; median seconds:',round(statistics.median(metric_times),1) if metric_times else 'not measured')
    print('This is a partial serial ETA; missing component, property and fingerprint timings prevent a reliable total ETA.')
    if state.get('error'): print('ERROR:',state['error'])
    if verify:
        from unifiedgroupgen.common.configuration import resolve_path
        inputs={}
        for job in jobs:
            inputs[job['checkpoint']]=job['checkpoint_sha256']
            inputs.update(job['data_sha256'])
            prior=job.get('prior_samples')
            if prior:
                inputs[prior['samples']]=prior['samples_sha256']
                inputs.update(prior.get('metadata_sha256',{}))
            component=job.get('prior_components')
            if component:
                inputs.update({r['path']:r['sha256'] for r in component['files'].values()})
                inputs[component['data_identity']['path']]=component['data_identity']['sha256']
        inputs.update(state['identity']['code'])
        for name,expected in inputs.items():
            path=resolve_path(name)
            if not path.is_file() or digest(path)!=expected: raise ValueError('Campaign input/code changed: '+str(path))
        print('Verified input/code files:',len(inputs))
        checked=records=failed=0
        for job in jobs:
            key=job['model']+'/'+job['name']; entry=entries.get(key,{})
            folder=output/key
            receipt=entry if entry.get('state')=='complete' else entry.get('generation',{})
            if receipt.get('state')!='complete': continue
            for name,expected in receipt.get('outputs',{}).items():
                path=folder/name
                if not path.is_file() or digest(path)!=expected: raise ValueError('Output hash mismatch: '+str(path))
                checked+=1
            ledger=folder/'samples.jsonl'
            if not ledger.exists(): continue
            rows=[json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines() if line.strip()]
            requested=len(job['case_ids'])*job['k'] if job['task']=='csp' else job['count']
            if len(rows)!=requested or [r['index'] for r in rows]!=list(range(requested)):
                raise ValueError('Incomplete denominator or index sequence: '+key)
            for row in rows:
                if row['status']=='generated':
                    if digest_bytes(row['cif'].encode('utf-8'))!=row['cif_sha256']:
                        raise ValueError('Inline CIF hash mismatch: '+key)
                else: failed+=1
            records+=len(rows)
        print('Verified output files:',checked,'| complete ledger records:',records,'| recorded generation failures:',failed)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args(); inspect(args.output,args.verify)


if __name__=='__main__': main()
