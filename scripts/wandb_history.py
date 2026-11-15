"""Read-only history planning, immutable snapshots, and separate W&B backfill runs.

No model/checkpoint imports, scheduler commands, or training-process interaction.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import uuid


SCHEMA='ugg_wandb_history_snapshot_v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value,exclusive=False):
    with Path(path).open('x' if exclusive else 'w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def number(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def metrics(row):
    values=row.get('metrics',{})
    result={}
    for phase in ('train','val'):
        for name in ('loss','continuous','element','jump'):
            key=phase+'/'+name
            value=values.get(key+'_epoch',values.get(key))
            if value is not None:
                if not number(value):raise ValueError('Nonfinite/nonnumeric epoch metric: '+key)
                result[key]=value
    if not {'train/loss','val/loss'} <= result.keys():
        raise ValueError('History needs epoch training and validation loss')
    rates=row.get('learning_rates_used',[])
    rate=values.get('train/learning_rate',rates[0] if rates else None)
    if rate is not None:
        if not number(rate):raise ValueError('Invalid learning rate')
        result['train/learning_rate']=rate
    for key,value in (row.get('early_stopping') or {}).items():
        if key in ('wait_count','patience','best_score') and number(value):result['early_stopping/'+key]=value
    return result


def history(path):
    """Take one bounded read; omit only an unfinished final JSONL record."""
    path=Path(path)
    with path.open('rb') as stream:
        size=os.fstat(stream.fileno()).st_size
        data=stream.read(size)
    end=data.rfind(b'\n')+1
    complete=data[:end]
    rows=[]
    for index,line in enumerate(complete.splitlines(),1):
        if not line.strip():continue
        try:
            row=json.loads(line)
            for key in ('epoch','global_step'):
                if type(row[key]) is not int or row[key]<0:raise ValueError('Invalid '+key)
            rows.append(dict(epoch=row['epoch'],epoch_number=row['epoch']+1,
                             global_step=row['global_step'],**metrics(row)))
        except (ValueError,KeyError,TypeError) as error:
            raise ValueError(f'{path}:{index}: {error}') from error
    return complete,rows,dict(path=str(path.resolve()),complete_bytes=len(complete),
        sha256=sha(complete),ignored_tail_bytes=len(data)-end)


def collect(campaign):
    campaign=Path(campaign).resolve();recovery=read(campaign)
    if recovery.get('schema')!='ugg_joint_flow_repaired_resume_v1':
        raise ValueError('Expected a repaired-resume campaign')
    chain=[(campaign,recovery)];seen={campaign}
    while chain[-1][1].get('schema')=='ugg_joint_flow_repaired_resume_v1':
        child=chain[-1][1];original=Path(child['original']).resolve()
        if original in seen:raise ValueError('Recovery ancestry contains a cycle')
        seen.add(original)
        if sha(original.read_bytes())!=child['original_manifest_sha256']:raise ValueError('Original campaign hash changed')
        chain.append((original,read(original)))
    chain.reverse();old=chain[0][1]
    if old.get('schema')!='ugg_joint_flow_full_5090_v2':raise ValueError('Unexpected original campaign')
    report=dict(schema=SCHEMA,campaign=str(campaign),campaign_sha256=sha(campaign.read_bytes()),
        original_revision=old['revision'],resume_revision=recovery['revision'],
        campaign_name=recovery['name'],models={},training_started=False)
    files={};protected=[path.parent for path,_ in chain]
    for model in recovery['models']:
        if not model or Path(model).name!=model or any(c in model for c in '/\\:'):
            raise ValueError('Unsafe model name')
        if model not in old['models']:raise ValueError('Model missing from original campaign')
        boundaries=[child['compatibility']['models'][model]['global_step'] for _,child in chain[1:]]
        labels=['original']+['resume'+str(i) for i in range(1,len(chain)-1)]+['resume']
        merged={};sources=[];duplicates=0
        for index,(_,source_campaign) in enumerate(chain):
            segment=labels[index]
            directory=Path(source_campaign['root'])/'outputs'/model/source_campaign['name']
            protected.append(directory.resolve())
            data,rows,source=history(directory/'model.history.jsonl')
            source['excluded_after_checkpoint']=0
            for row in rows:
                if index<len(boundaries) and row['global_step']>boundaries[index]:
                    source['excluded_after_checkpoint']+=1;continue
                epoch=row['epoch']
                row['segment']=next((labels[i] for i,limit in enumerate(boundaries) if row['global_step']<=limit),labels[-1])
                if epoch in merged:
                    if merged[epoch]!=row:raise ValueError(f'Conflicting epoch {epoch} for {model}')
                    duplicates+=1
                merged[epoch]=row
            name=f'inputs/{model}.{segment}.jsonl';files[name]=data
            source['snapshot']=name;sources.append(source)
        rows=[merged[key] for key in sorted(merged)]
        if not rows:raise ValueError('No completed epochs for '+model)
        if any(b['global_step']<=a['global_step'] for a,b in zip(rows,rows[1:])):
            raise ValueError('Non-increasing global steps for '+model)
        name=model+'.jsonl'
        files[name]=(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows)).encode()
        report['models'][model]=dict(rows=len(rows),first_epoch=rows[0]['epoch'],last_epoch=rows[-1]['epoch'],
            first_global_step=rows[0]['global_step'],last_global_step=rows[-1]['global_step'],
            missing_epochs=[i for i in range(rows[0]['epoch'],rows[-1]['epoch']+1) if i not in merged],
            duplicates_removed=duplicates,sources=sources,history=name)
    report['files']={name:sha(data) for name,data in files.items()}
    return report,files,protected


def prepare(campaign,output,project='UnifiedGroupGen'):
    output=Path(output).resolve()
    if output.exists() or output.is_symlink():raise FileExistsError('Snapshot output already exists: '+str(output))
    report,files,protected=collect(campaign)
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError('Snapshot output must be separate from source runs and frozen campaigns')
    output.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        path=output/name;path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
    report.update(created=datetime.now(timezone.utc).isoformat(),project=project,
        runs={model:uuid.uuid4().hex for model in report['models']})
    write(output/'snapshot.json',report,exclusive=True)
    return report


def verify(snapshot):
    snapshot=Path(snapshot).resolve();report=read(snapshot/'snapshot.json')
    if report.get('schema')!=SCHEMA:raise ValueError('Unexpected snapshot schema')
    for name,expected in report['files'].items():
        path=(snapshot/name).resolve()
        if not path.is_relative_to(snapshot) or sha(path.read_bytes())!=expected:
            raise ValueError('Snapshot file/hash mismatch: '+name)
    for model,info in report['models'].items():
        if info['history'] not in report['files'] or model not in report['runs']:
            raise ValueError('Incomplete snapshot model')
    return snapshot,report


def publish(snapshot,mode='offline',entity=None):
    snapshot,report=verify(snapshot)
    if mode not in ('offline','online'):raise ValueError('Mode must be offline or online')
    if mode=='online' and not entity:raise ValueError('Online upload requires explicit --entity')
    import wandb
    if wandb.run is not None:raise ValueError('Run backfill in its own process, outside training')
    results={}
    for model,info in report['models'].items():
        receipt=snapshot/(model+'.receipt.json')
        if receipt.exists():
            prior=read(receipt)
            if prior['state']=='complete' and prior['mode']==mode and prior.get('entity')==entity:
                results[model]=prior;continue
            raise ValueError('Prior/incomplete publication exists; preserve it and prepare a new snapshot')
        rows=[json.loads(line) for line in (snapshot/info['history']).read_text().splitlines()]
        state=dict(state='publishing',mode=mode,entity=entity,project=report['project'],
                   run_id=report['runs'][model],rows=len(rows))
        write(receipt,state,exclusive=True)
        run=None
        try:
            options=dict(project=report['project'],entity=entity,id=state['run_id'],mode=mode,
                name=model+'-history-'+snapshot.name,group=report['campaign_name'],job_type='history-backfill',
                dir=str(snapshot),config=dict(model=model,history_backfill=True,epoch_only=True,
                    original_revision=report['original_revision'],resume_revision=report['resume_revision'],
                    history_sha256=report['files'][info['history']]),
                settings=wandb.Settings(disable_git=True,disable_code=True,x_disable_stats=True))
            if mode=='online':options['resume']='never'
            run=wandb.init(**options)
            run.define_metric('epoch_number')
            for prefix in ('train/*','val/*','early_stopping/*'):
                run.define_metric(prefix,step_metric='epoch_number')
            for row in rows:run.log(row,step=row['global_step'])
            url=run.url if mode=='online' else None
            run.finish()
            state.update(state='complete',url=url)
        except BaseException as error:
            state.update(state='failed',error=type(error).__name__)
            if run is not None:run.finish(exit_code=1)
            raise
        finally:
            write(receipt,state)
        results[model]=state
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan',help='Read-only source validation; no W&B import or upload')
    p.add_argument('--campaign',required=True)
    p=sub.add_parser('prepare',help='Copy completed history into a new standalone snapshot')
    p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
    p.add_argument('--project',default='UnifiedGroupGen')
    p=sub.add_parser('publish',help='Publish only verified snapshot rows into separate W&B runs')
    p.add_argument('--snapshot',required=True);p.add_argument('--mode',choices=['offline','online'],default='offline')
    p.add_argument('--entity',help='Required W&B account/team for online upload')
    args=parser.parse_args(argv)
    if args.action=='plan':result=collect(args.campaign)[0]
    elif args.action=='prepare':result=prepare(args.campaign,args.output,args.project)
    else:result=publish(args.snapshot,args.mode,args.entity)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
