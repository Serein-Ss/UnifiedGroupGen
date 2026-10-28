"""PNG-only quantitative figures; unknown quantities are never drawn as zero."""
from pathlib import Path
import numpy as np


def figures(rows, report, references, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    fields = ['generation_success', 'parse_success', 'periodic_distance_valid', 'chemical_valid',
              'basic_valid', 'fingerprint_success', 'exact_target_group', 'target_operation_containment']
    values = [report['common'][k]['rate_over_all_attempts'] for k in fields]
    for i, (key, value) in enumerate(zip(fields, values)):
        if value is not None:
            axes[0].barh(i, value, color='#407c9d')
            metric = report['common'][key]
            axes[0].text(min(value+.02, .7), i, f"{metric['passed']}/{metric['attempted']}", va='center', fontsize=8)
        else:
            axes[0].text(.03, i, 'Not assessed', va='center', color='gray')
    axes[0].set(yticks=range(len(fields)), yticklabels=[x.replace('_', ' ') for x in fields],
                xlim=(0, 1.15), xlabel='Fraction of ALL attempts', title='Common validity / constraints')
    axes[0].invert_yaxis()
    if report['task']=='csp':
        for profile, values in report['csp']['profiles'].items():
            budgets = sorted(int(k) for k in values if k.isdigit())
            axes[1].plot(budgets, [values[str(k)]['match_rate'] for k in budgets], 'o-', label=profile)
        axes[1].set(xlabel='Candidates per reference case', ylabel='Match@K (all cases)', ylim=(0, 1.05))
        axes[1].legend(fontsize=8)
    else:
        fields = ['unique_valid', 'novel_valid', 'vun_yield']
        for i, key in enumerate(fields):
            value = report['dng'][key]['rate_over_all_attempts']
            if value is not None:
                axes[1].bar(i, value, color='#dc9d4c')
            else:
                axes[1].text(i, .1, 'N/A', ha='center')
        axes[1].set(xticks=range(3), xticklabels=['Unique valid', 'Novel valid', 'VUN'], ylim=(0, 1.05),
                    ylabel='Fraction of ALL attempts')
    axes[1].set_title('CSP reconstruction' if report['task']=='csp' else 'De novo population')
    quantities = report['properties']['quantities']
    for i, (name, value) in enumerate(quantities.items()):
        hit = value['valid_hit']['rate_over_all_attempts']
        if hit is not None:
            axes[2].bar(i, hit, color='#679975')
        else:
            axes[2].text(i, .1, 'Not assessed', ha='center', rotation=90)
    if quantities:
        axes[2].set(xticks=range(len(quantities)), xticklabels=list(quantities), ylim=(0, 1.05))
        axes[2].tick_params(axis='x', rotation=25)
    else:
        axes[2].text(.5, .5, 'No property targets / predictions', ha='center', transform=axes[2].transAxes)
    axes[2].set(title='Independent proxy: valid target hits', ylabel='Fraction of targeted attempts')
    fig.suptitle(f"{report['dataset'] or report['kind']} | {report['task'].upper()} | N={report['attempted']} | {report['protocol_id']}")
    fig.savefig(output/'metrics.png', dpi=200, facecolor='white'); plt.close(fig)
    keys = [('atoms','Atoms/cell'), ('num_elements','Distinct elements'),
            ('area_per_atom_angstrom2','Area/atom (A2)') if report['kind']=='layer' else ('volume_per_atom_angstrom3','Volume/atom (A3)'),
            ('layer_thickness_angstrom','Layer thickness (A)') if report['kind']=='layer' else ('density_g_cm3','Density (g/cm3)')]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    for ax, (key, label) in zip(axes, keys):
        for data, name, color in [(rows,'Valid generated','#407c9d'), (references,'Reference','#dc9d4c')]:
            values = np.sort([r[key] for r in data if r.get(key) is not None and (name=='Reference' or r.get('basic_valid'))])
            if len(values):
                ax.step(values, np.arange(1,len(values)+1)/len(values), where='post', color=color, marker='o' if len(values)<3 else None,
                        markersize=3,label=f'{name}: {len(values)}')
        ax.set(xlabel=label, ylabel='Cumulative fraction', ylim=(0,1.05))
        if ax.lines:
            ax.legend(fontsize=7)
    fig.suptitle('Raw structural distributions; valid-generated denominator stated in legends')
    fig.savefig(output/'distributions.png', dpi=200, facecolor='white'); plt.close(fig)
    if report['groups']:
        fig, ax = plt.subplots(figsize=(12,3.5), constrained_layout=True)
        groups = report['groups']
        for field, label, marker in [('exact','Exact reidentified group','o'),('containment','Target operations contained','x'),('basic_valid','Basic valid','.')]:
            selected = [(g['group'],g[field]['rate_over_all_attempts']) for g in groups if g[field]['rate_over_all_attempts'] is not None]
            if selected:
                x, y = zip(*selected); ax.scatter(x,y,label=label,marker=marker,s=18)
        ax.set(xlabel='Space / layer group number', ylabel='Fraction of group attempts', ylim=(-.03,1.08))
        if ax.collections:
            ax.legend(fontsize=8)
        fig.savefig(output/'groups.png', dpi=200, facecolor='white'); plt.close(fig)
    assessed=[p for p,v in quantities.items() if v['predictions']['evaluated']]
    if assessed:
        fig,axes=plt.subplots(1,len(assessed),figsize=(4.5*len(assessed),3.5),squeeze=False,constrained_layout=True)
        for ax,p in zip(axes.ravel(),assessed):
            for source,name,color in [(rows,'Generated','#407c9d'),(references,'Reference','#dc9d4c')]:
                values=np.sort([r['predicted_'+p] for r in source if r.get('predicted_'+p) is not None])
                if len(values): ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',marker='o' if len(values)<3 else None,
                                        markersize=3,label=f'{name}: {len(values)}',color=color)
            ax.set(xlabel=p+' (independent proxy)',ylabel='Cumulative fraction');ax.legend(fontsize=8)
        fig.savefig(output/'properties.png',dpi=200);plt.close(fig)


def campaign_summary(output, state):
    import csv
    from .evaluation_io import write_json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);rows=[]
    for entry in state['jobs']:
        model,job=entry['name'].split('/',1)
        path=output/entry['name']/'metrics/summary.json'
        if not path.exists(): continue
        report=__import__('json').loads(path.read_text(encoding='utf-8'))
        row={'model':model,'job':job,'task':report['task'],'kind':report['kind'],'dataset':report['dataset'],
             'attempted':report['attempted'],'seconds':entry['seconds'],'protocol':report['protocol_id']}
        for key in ('generation_success','basic_valid','legacy_joint_valid','exact_target_group','target_operation_containment','composition_satisfied'):
            row[key]=report['common'][key]['rate_over_all_attempts']
        if report['task']=='csp':
            for profile,values in report['csp']['profiles'].items():
                for k,v in values.items():
                    if k.isdigit(): row[profile+'_match_at_'+k]=v['match_rate']
        else:
            row['vun_yield']=report['dng']['vun_yield']['rate_over_all_attempts']
        row['valid_joint_property_hit']=report['properties']['valid_joint_hit']['rate_over_all_attempts']
        rows.append(row)
    fields=sorted({k for r in rows for k in r})
    with (output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or ['model']);writer.writeheader();writer.writerows(rows)
    write_json(output/'summary.json',{'state':state['state'],'jobs':rows,'scope':'Raw CSP/DNG metrics; no physical validation'})
    # Primary no-property DNG comparison; all conditional jobs remain in the table.
    selected=[r for r in rows if r['job'].endswith('_dng')]
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,key in zip(axes,('basic_valid','exact_target_group','vun_yield')):
        for i,r in enumerate(selected):
            if r.get(key) is not None: ax.bar(i,r[key],color='#407c9d')
            else: ax.text(i,.05,'N/A',ha='center',rotation=90)
        ax.set(xticks=range(len(selected)),xticklabels=[r['model']+'/'+r['dataset'] for r in selected],
               ylabel='Fraction of ALL attempts',title=key.replace('_',' '),ylim=(0,1.05))
        ax.tick_params(axis='x',labelrotation=90,labelsize=7)
    fig.savefig(output/'model_comparison.png',dpi=200);plt.close(fig)
