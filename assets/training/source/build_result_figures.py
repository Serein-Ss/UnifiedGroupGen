"""Plot the stopped-run snapshot without interpreting untested endpoints as zero."""
from pathlib import Path
import json
import hashlib
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT=Path(__file__).resolve().parents[3]
AUDIT=PROJECT.parent/'tmp/UnifiedGroupGen_remote_audit_20261017'
OUT=PROJECT/'assets/training'
RESULTS=PROJECT/'assets/results'
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,
    'axes.titlesize':11,'legend.fontsize':9,'svg.fonttype':'none','pdf.fonttype':42})


def export(fig,folder,name):
    folder.mkdir(exist_ok=True,parents=True)
    fig.savefig(folder/f'{name}.png',dpi=260,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    a=json.loads((AUDIT/'audit_summary.json').read_text(encoding='utf-8'))
    fig,axs=plt.subplots(1,3,figsize=(11.8,3.55),layout='constrained')
    for ax,(name,r) in zip(axs,a['runs'].items()):
        rows=[json.loads(s) for s in (AUDIT/'evidence'/r['path']/'model.history.jsonl').read_text().splitlines() if s.strip()]
        rows=list({r['epoch']:r for r in rows}.values())
        epochs=np.array([r['epoch']+1 for r in rows])
        train=np.array([r['train']['descriptor']+r['train']['flow'] for r in rows])
        vr=[r for r in rows if r.get('val_loss') is not None]
        ax.plot(epochs,train,color='#A5B6C3',lw=.85,label='Train')
        ax.plot([r['epoch']+1 for r in vr],[r['val_loss'] for r in vr],color='#3468A5',lw=1.7,label='Validation')
        b=r['best'];ax.scatter(b['epoch']+1,b['val_loss'],marker='*',s=95,color='#C96C46',zorder=5,label='Best validation')
        ax.axvline(r['status']['completed_epochs'],color='#6B7280',ls=':',lw=1)
        ax.set_title(name+(' (stopped)' if name=='Joint' else ' (early stopped)'),loc='left')
        ax.set_xlabel('Completed epoch (1-based)');ax.set_ylabel('Descriptor + flow loss')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E3E8EB',lw=.6)
        ax.set_xlim(left=0)
    axs[0].legend(frameon=False)
    export(fig,OUT,'training_history')
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.35),layout='constrained',gridspec_kw={'width_ratios':[2.1,1]})
    gs=a['generation']
    labels=[r['model']+' / '+('uncond.' if r['mode']=='unconditional' else 'cond.') for r in gs]
    mat=np.array([[r['generated'],r['exact_group'],r['counts']['distance_valid'],r['counts']['no_coincident_orbits']] for r in gs])
    im=axs[0].imshow(mat/64*100,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    axs[0].set_xticks(range(4),['Output','Exact SG','Distance ≥ 0.6 Å','No orbit overlap'])
    axs[0].set_yticks(range(4),labels)
    for i in range(4):
        for j in range(4):axs[0].text(j,i,f'{mat[i,j]}/64',ha='center',va='center',color='white')
    axs[0].set_title('a  Archived reference-generation checks',loc='left')
    axs[1].axis('off');axs[1].set_title('b  Evidence still missing',loc='left')
    for y,title in zip([.83,.59,.35,.11],['Shared SG/LG generation','MPTS52 chemical screening','Property hit rate','Physical stability']):
        axs[1].text(0,y,title,fontsize=11,color='#243444')
        axs[1].text(0,y-.12,'Not evaluated',fontsize=10,color='#7E8B95')
    export(fig,RESULTS,'reference_generation_summary')
    pd.DataFrame(mat,index=labels,columns=['output','exact_group','distance_0p6','no_orbit_overlap']).to_csv(RESULTS/'reference_generation_counts.csv')
    # Reuse a previous local audit only after binding all of its CIFs to this snapshot.
    prior=PROJECT/'artifacts/audits/mp20_generation_20261008'
    old_zip=PROJECT/'tmp/mp20_generation_audit.zip'
    if prior.exists() and old_zip.exists():
        matched=0
        with zipfile.ZipFile(old_zip) as z:
            cifs=[n for n in z.namelist() if n.endswith('.cif')]
            for n in cifs:
                parts=n.replace('\\','/').split('/')
                start=next(i for i,x in enumerate(parts) if x in ('conditional','unconditional'))
                target=AUDIT/'evidence/artifacts/runs/local_large_seed42/mp20'/('/'.join(parts[start:]))
                assert hashlib.sha256(z.read(n)).digest()==hashlib.sha256(target.read_bytes()).digest()
                matched+=1
        assert matched==128
        extra=json.loads((prior/'summary.json').read_text())
        manifest=json.loads((prior/'manifest.json').read_text())
        assert manifest['remote_checkpoint_sha256']==a['runs']['MP20']['test']['checkpoint_sha256']
        inv={x['path']:x['sha256'] for x in json.loads((AUDIT/'inventory.json').read_text())}
        for split,sha in extra['reference_sha256'].items():
            assert inv[f'artifacts/full/mp_20/{split}.jsonl']==sha
        counts=[extra['modes'][m]['counts'] for m in ['unconditional','conditional']]
        keys=['exact_group','distance_0p6_pass','radius_ratio_0p6_pass','smact14_pass','diagnostic_joint_pass']
        labels=['Exact SG','Distance ≥ 0.6 Å','Radius ratio ≥ 0.6','SMACT screening','Combined screen']
        fig,ax=plt.subplots(figsize=(8.1,3.7),layout='constrained')
        x=np.arange(len(keys))
        for i,(row,color,name) in enumerate(zip(counts,['#3468A5','#C96C46'],['Unconditional','Conditional'])):
            vals=[row[k] for k in keys]
            ax.bar(x+(i-.5)*.35,vals,width=.35,color=color,label=name,zorder=3)
            for xx,v in zip(x+(i-.5)*.35,vals):ax.text(xx,v+1,f'{v}/64',ha='center',fontsize=8.5)
        ax.set_xticks(x,labels);ax.set_ylim(0,80);ax.set_ylabel('Samples passing (out of 64)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E3E8EB',lw=.6,zorder=0)
        ax.legend(frameon=False,ncol=2,loc='upper right');ax.set_title('MP20: additional local screening of the same 128 CIFs',loc='left')
        export(fig,RESULTS,'mp20_additional_screening')
        proof={'matching_cifs':matched,'checkpoint_sha256':manifest['remote_checkpoint_sha256'],
               'reference_split_hashes_match':True,'audit_source':'artifacts/audits/mp20_generation_20261008/summary.json',
               'modes':{m:extra['modes'][m] for m in ['unconditional','conditional']},
               'scope':'Historical screening reused after exact CIF, checkpoint and reference-data identity checks. No new chemistry calculation; not stability or property certification.'}
        (RESULTS/'mp20_additional_screening.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'README.md').write_text('# 训练曲线\n\n用户已确认服务器停止，图中Joint为停止快照。横轴为完成轮数，从1开始。MP20重复epoch保留最后记录；未验证轮次不填0，验证曲线只连接真实验证点。竖虚线为已完成轮数；Joint第199轮的部分批次不作为完整epoch。\n\n损失为描述与流损失之和；训练/验证条件机制不同，绝对值不用于不同数据集的性能排名。星号为各自最佳验证点。\n\n![训练曲线](training_history.png)\n\n[PDF](training_history.pdf) · [SVG](training_history.svg)\n',encoding='utf-8')
    print('Saved training and reference-generation figures')


if __name__=='__main__':main()
