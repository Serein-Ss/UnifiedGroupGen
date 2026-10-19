"""Dataset figures from the archived prepared records; no model or data mutations.

Run with the project's environment. Input defaults to ../../tmp/UnifiedGroupGen.zip
relative to the project. Exact plots are regenerated from the source records.
CCFA grouped-bar/heatmap grammar is adapted to Matplotlib for PNG exports.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import os
import sys
import zipfile

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / 'assets/datasets'
sys.path.insert(0, str(PROJECT / 'src'))
from unifiedgroupgen.symmetry import wp_arrays, metric_chart
from pymatgen.core import Element

NAMES = ['MP20', 'MPTS52', 'C2DB']
PATHS = ['artifacts/full/mp_20', 'artifacts/full_tol0p01/mpts_52',
         'artifacts/full_official_tol0p1_v7/c2db_51']
KEYS = ['mp_20', 'mpts_52', 'c2db_51']
COLORS = ['#3468A5', '#C96C46', '#167D8D']
SPLITS = ['train', 'val', 'test']
PROP = ['formation_energy', 'band_gap', 'energy_above_hull']
P_LABEL = ['Formation energy (eV/atom)', 'Band gap (eV)', 'Energy above hull (eV/atom)']
CAPTIONS = {}


def savefig(fig, name, caption):
    fig.savefig(OUT / f'{name}.png', dpi=260, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    CAPTIONS[name] = caption


def style_axes(ax, xlabel=None, ylabel=None):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#E3E8EB', linewidth=.6, zorder=0)
    ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)


def ecdf(ax, arrays, xlabel, zero=True):
    for name, color, v in zip(NAMES, COLORS, arrays):
        x = np.sort(np.asarray(v, dtype=float))
        ax.step(x, np.arange(1, len(x)+1)/len(x), where='post', color=color,
                label=name, linewidth=1.65)
    style_axes(ax, xlabel, 'Cumulative fraction')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, 1.02)
    if zero: ax.set_xlim(left=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=PROJECT.parent/'tmp/UnifiedGroupGen.zip')
    args = parser.parse_args()
    (OUT/'tables').mkdir(parents=True, exist_ok=True)
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman', 'font.size':10,
        'axes.titlesize':11, 'axes.labelsize':10, 'xtick.labelsize':9,
        'ytick.labelsize':9, 'legend.fontsize':9, 'pdf.fonttype':42,
        'svg.fonttype':'none', 'mathtext.fontset':'stix', 'axes.unicode_minus':False})
    data=[]; elements=collections.defaultdict(collections.Counter)
    ids={}; hashes={}; patterns={}; wp_seen={}; joint_counts=[]; source_checks=[]
    with zipfile.ZipFile(args.archive) as z:
        for name, prefix in zip(NAMES, PATHS):
            ids[name]=set();hashes[name]=set();patterns[name]=set();wp_seen[name]=set()
            for split in SPLITS:
                path=prefix+'/'+split+'.jsonl';raw=z.read('UnifiedGroupGen/'+path)
                source_checks.append({'path':path,'sha256':hashlib.sha256(raw).hexdigest()})
                for line in raw.splitlines():
                    if not line.strip():continue
                    r=json.loads(line);d=r['descriptor'];pv=r['provenance'];orb=d['orbits']
                    root,_,bases=metric_chart(d['kind'],d['number'])
                    kd=len(bases);qd=0;n=0;heights=[]
                    for o in orb:
                        aa,bb,axes=wp_arrays(d['kind'],d['number'],o['letter'])
                        if d['kind']=='layer':
                            q=np.asarray(r['state'][kd+qd:kd+qd+len(axes)])
                            heights.extend((bb[:,2]+aa[:,2,:]@q).tolist())
                        qd+=len(axes);n+=len(aa)
                        wp_seen[name].add((d['number'],o['letter']))
                    assert len(r['state'])==qd+kd, (name,r['id'],'DOF')
                    assert n==pv['expanded_atoms'], (name,r['id'],'atoms')
                    zs=set(o['atomic_number'] for o in orb)
                    for el in zs:elements[name][el]+=1
                    ids[name].add(r['id']);hashes[name].add(pv['source_cif_sha256'])
                    pattern=(d['number'],tuple(sorted(o['letter'] for o in orb)))
                    patterns[name].add(pattern)
                    row={'dataset':name,'split':split,'id':r['id'],'kind':d['kind'],
                         'group':d['number'],'source_atoms':pv['source_atoms'],
                         'expanded_atoms':n,'orbits':len(orb),'elements':len(zs),
                         'lattice_dof':kd,'coordinate_dof':qd,'state_dof':kd+qd,
                         'state_fraction':(kd+qd)/(3*n+(6 if d['kind']=='space' else 3)),
                         'source_cif_sha256':pv['source_cif_sha256']}
                    measure=np.linalg.det(root)*np.exp(.5*np.dot(r['state'][:kd],np.trace(bases,axis1=1,axis2=2)))
                    row['volume_per_atom_A3']=measure/n if d['kind']=='space' else np.nan
                    row['area_per_atom_A2']=measure/n if d['kind']=='layer' else np.nan
                    row['layer_thickness_A']=float(np.ptp(heights)) if heights else np.nan
                    for prop in PROP:
                        v=r.get('properties',{}).get(prop)
                        row[prop]=float(v) if v is not None and np.isfinite(float(v)) else np.nan
                    data.append(row)
            print('Read and checked',name,flush=True)
        for name,key in zip(NAMES,KEYS):
            for split in SPLITS:
                path=f'artifacts/full_joint_sourcegrouped/{key}/{split}.jsonl'
                with z.open('UnifiedGroupGen/'+path) as f:
                    count=sum(bool(l.strip()) for l in f)
                joint_counts.append({'dataset':name,'split':split,'records':count})
    df=pd.DataFrame(data)
    assert len(df)==102610
    df.to_csv(OUT/'tables/record_statistics.csv.gz',index=False,compression='gzip')
    joint=pd.DataFrame(joint_counts)
    joint.to_csv(OUT/'tables/joint_split_counts.csv',index=False)
    assert joint.records.sum()==102610
    counts=df.groupby(['dataset','split']).size().unstack().reindex(NAMES)[SPLITS]
    counts.to_csv(OUT/'tables/reference_split_counts.csv')
    totals=df.groupby('dataset').size().reindex(NAMES)
    summary=[]
    for name in NAMES:
        d=df[df.dataset==name]
        row={'dataset':name,'records':len(d),'unique_ids':len(ids[name]),
             'unique_source_cif_hashes':len(hashes[name]),'groups_observed':d.group.nunique(),
             'groups_catalog':80 if name=='C2DB' else 230,
             'wyckoff_types_observed':len(wp_seen[name]),
             'orbit_patterns_undecorated':len(patterns[name]),
             'unique_elements':len(elements[name]),'zero_coordinate_dof':int((d.coordinate_dof==0).sum())}
        for col in ['source_atoms','expanded_atoms','orbits','lattice_dof','coordinate_dof','state_dof','state_fraction','volume_per_atom_A3','area_per_atom_A2','layer_thickness_A']:
            values=d[col].dropna()
            if values.empty:
                for label in ['min','median','p95','max']:row[col+'_'+label]=None
            else:
                for label,v in [('min',values.min()),('median',values.median()),('p95',values.quantile(.95)),('max',values.max())]:row[col+'_'+label]=float(v)
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'tables/dataset_summary.csv',index=False)
    groupcounts=df.groupby(['dataset','group']).size().rename('records').reset_index()
    groupcounts.to_csv(OUT/'tables/group_counts.csv',index=False)
    elrows=[{'dataset':name,'atomic_number':el,'element':Element.from_Z(el).symbol,
             'records_with_element':v,'fraction':v/totals[name]} for name in NAMES for el,v in sorted(elements[name].items())]
    pd.DataFrame(elrows).to_csv(OUT/'tables/element_prevalence.csv',index=False)
    coverage=np.array([[df.loc[df.dataset==name,p].notna().mean()*100 for p in PROP] for name in NAMES])
    pd.DataFrame(coverage,index=NAMES,columns=PROP).to_csv(OUT/'tables/property_coverage_percent.csv')
    propstats=[]
    for name in NAMES:
        for prop in PROP:
            v=df.loc[df.dataset==name,prop].dropna()
            propstats.append({'dataset':name,'property':prop,'n':len(v),'missing':int(totals[name]-len(v)),
                'min':v.min(),'p01':v.quantile(.01),'median':v.median(),'p99':v.quantile(.99),'max':v.max(),'zero_fraction':float((v==0).mean())})
    pd.DataFrame(propstats).to_csv(OUT/'tables/property_statistics.csv',index=False)

    # Figure 1: dataset size and the two different split schemes.
    fig,axs=plt.subplots(1,3,figsize=(12.4,3.45),layout='constrained')
    for ax,frame,title in [(axs[0],counts,'a  Original reference splits'),
        (axs[1],joint.pivot(index='dataset',columns='split',values='records').reindex(NAMES)[SPLITS],'b  Shared-model splits')]:
        bottom=np.zeros(3)
        for split,color in zip(SPLITS,['#3468A5','#91B6C7','#D5DEE7']):
            ax.bar(NAMES,frame[split],bottom=bottom,color=color,label=split,width=.6)
            bottom+=frame[split].values
        for x,y in enumerate(bottom):ax.text(x,y+900,f'{int(y):,}',ha='center',fontsize=9)
        ax.set_ylim(0,51000);style_axes(ax,ylabel='Source records');ax.set_title(title,loc='left')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(3,3));ax.legend(frameon=False,loc='upper right')
    for i,row in enumerate(summary):
        axs[2].bar(i,row['groups_observed']/row['groups_catalog']*100,color=COLORS[i],width=.6)
        axs[2].text(i,3,f"{row['groups_observed']} / {row['groups_catalog']}",ha='center',color='white',weight='bold')
    axs[2].set_xticks(range(3),['MP20\nSpace groups','MPTS52\nSpace groups','C2DB\nLayer groups'])
    axs[2].set_ylim(0,105);style_axes(axs[2],ylabel='Catalog coverage (%)');axs[2].set_title('c  Observed symmetry groups',loc='left')
    savefig(fig,'01_dataset_overview','原始参考划分与联合重划分分别统计；总量均按来源记录计数。空间群和层群使用各自230/80的分母，编号不视为同一类别空间。')

    # Figure 2: conventional-cell expansion and continuous dimensions.
    fig,axs=plt.subplots(2,3,figsize=(11.8,6.6),layout='constrained')
    cols=['source_atoms','expanded_atoms','orbits','coordinate_dof','state_dof','state_fraction']
    labels=['Atoms in source cell','Atoms in standardized cell','Orbit instances per structure',
            'Independent coordinate DOF','Total state DOF: dim(k, q)','dim(k, q) / ambient DOF']
    for ax,col,label,letter in zip(axs.flat,cols,labels,'abcdef'):
        ecdf(ax,[df.loc[df.dataset==n,col] for n in NAMES],label)
        ax.set_title(letter+'  '+label,loc='left');ax.set_xlabel('Value')
        if col=='state_fraction':ax.set_xlim(0,1.02)
    axs[0,0].legend(frameon=False,loc='lower right')
    savefig(fig,'02_structure_complexity','全102610条记录的经验累积分布，未截断尾部。ambient DOF=3N+6（三维）或3N+3（层状，含高度）；该比值仅描述参数化维数减少，不是速度提升倍数。')

    # Figure 3: full catalogs; independent SG and LG axes.
    fig,axs=plt.subplots(3,1,figsize=(11.2,5.1),layout='constrained')
    for ax,name,color in zip(axs,NAMES,COLORS):
        limit=80 if name=='C2DB' else 230
        v=df[df.dataset==name].group.value_counts().reindex(range(1,limit+1),fill_value=0)
        ax.bar(v.index,v.values/len(df[df.dataset==name])*100,color=color,width=.85)
        ax.set_xlim(.3,limit+.7);style_axes(ax,ylabel='Records (%)')
        ax.set_title(f'{name}: {int((v>0).sum())}/{limit} observed',loc='left')
        ax.set_xlabel('Layer-group number' if name=='C2DB' else 'Space-group number')
    savefig(fig,'03_symmetry_distribution','各自全群编号范围上的来源记录比例；空白编号表示未在此数据集出现。空间群与层群轴分别解释。')

    # Figure 4: per-structure element prevalence, not counts of expanded atoms.
    allz=sorted(set().union(*(set(elements[n]) for n in NAMES)))
    fig,axs=plt.subplots(2,1,figsize=(12.3,4.1),layout='constrained')
    for ax,zs in zip(axs,np.array_split(allz,2)):
        mat=np.array([[elements[n][int(el)]/totals[n]*100 for el in zs] for n in NAMES])
        im=ax.imshow(mat,aspect='auto',cmap='Blues',vmin=0,vmax=50)
        ax.set_xticks(range(len(zs)),[Element.from_Z(int(el)).symbol for el in zs],fontsize=8)
        ax.set_yticks(range(3),NAMES);ax.set_xlabel('Element (ordered by atomic number)')
        ax.set_xticks(np.arange(-.5,len(zs),1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.45);ax.tick_params(which='minor',length=0)
    fig.colorbar(im,ax=axs,label='Structures containing the element (%)',shrink=.85)
    savefig(fig,'04_element_coverage','分母为各数据集结构记录数，每种元素在每条记录中只计一次。白色表示0；非原子丰度分布，避免标准胞扩展倍数影响。')

    # Figure 5: missing labels distinguished from physical zeros.
    fig=plt.figure(figsize=(12.1,6.3),layout='constrained');gs=fig.add_gridspec(2,3,height_ratios=[1,1.55])
    ax=fig.add_subplot(gs[0,:]);im=ax.imshow(coverage,aspect='auto',vmin=0,vmax=100,cmap='Blues')
    ax.set_yticks(range(3),NAMES);ax.set_xticks(range(3),['Formation energy','Band gap','Energy above hull'])
    ax.set_title('a  Finite property-label coverage',loc='left')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{coverage[i,j]:.1f}%',ha='center',va='center',color='white' if coverage[i,j]>55 else '#243444')
    fig.colorbar(im,ax=ax,label='Records (%)',shrink=.8)
    for j,(prop,label) in enumerate(zip(PROP,P_LABEL)):
        ax=fig.add_subplot(gs[1,j])
        vals=[df.loc[df.dataset==n,prop].dropna() for n in NAMES]
        ecdf(ax,vals,label,zero=False);ax.set_title(chr(98+j)+'  '+label.split(' (')[0],loc='left')
        if j==0:ax.legend(frameon=False,loc='lower right')
        if j==1:ax.text(.97,.06,'MPTS52: no labels',transform=ax.transAxes,ha='right',color='#6B7280',fontsize=9)
    savefig(fig,'05_property_labels','上：有限数值标签的覆盖率，缺失不补零。下：有标签记录的完整经验分布；各数据源计算流程及能量参考可能不同，叠图仅描述数据，不表示标签完全可交换。完整范围含异常长尾。')

    # Figure 6: overlap and dimensional constraints.
    fig,axs=plt.subplots(1,3,figsize=(11.8,3.5),layout='constrained')
    overlaps=[]
    for a in NAMES:
        for b in NAMES:
            overlaps.append({'dataset_a':a,'dataset_b':b,'shared_material_ids':len(ids[a]&ids[b]),'shared_source_cif_hashes':len(hashes[a]&hashes[b])})
    ov=pd.DataFrame(overlaps);ov.to_csv(OUT/'tables/source_overlap.csv',index=False)
    for ax,col,title in [(axs[0],'shared_material_ids','a  Shared material IDs'),(axs[1],'shared_source_cif_hashes','b  Identical source CIF hashes')]:
        mat=ov.pivot(index='dataset_a',columns='dataset_b',values=col).reindex(index=NAMES,columns=NAMES).values
        ax.imshow(np.log10(mat+1),cmap='Blues',vmin=0,vmax=5)
        ax.set_xticks(range(3),NAMES);ax.set_yticks(range(3),NAMES);ax.set_title(title,loc='left')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{mat[i,j]:,}',ha='center',va='center',color='white' if mat[i,j]>5000 else '#243444')
    for i,name in enumerate(NAMES):
        v=(df.loc[df.dataset==name,'coordinate_dof']==0).mean()*100
        axs[2].bar(i,v,color=COLORS[i],width=.6);axs[2].text(i,v+.6,f'{v:.1f}%',ha='center')
    axs[2].set_xticks(range(3),NAMES);axs[2].set_ylim(0,100);style_axes(axs[2],ylabel='Records (%)')
    axs[2].set_title('c  Zero coordinate DOF',loc='left')
    savefig(fig,'06_overlap_and_fixed_sites','前两图为去重ID与原始CIF字节哈希的集合交集，数值为真实计数、颜色为log10(count+1)，不等于完整结构匹配。相同哈希证明原始字节相同；不同哈希不能排除等价结构。')

    # Figure 7: do not compare a vacuum-dependent 3D volume to a 2D area.
    fig,axs=plt.subplots(1,3,figsize=(11.8,3.6),layout='constrained')
    for name,color in zip(NAMES[:2],COLORS[:2]):
        x=np.sort(df.loc[df.dataset==name,'volume_per_atom_A3'].dropna())
        axs[0].step(x,np.arange(1,len(x)+1)/len(x),where='post',color=color,label=name,lw=1.7)
    axs[0].set_xscale('log');axs[0].legend(frameon=False,loc='lower right')
    for ax,col in zip(axs[1:],['area_per_atom_A2','layer_thickness_A']):
        x=np.sort(df.loc[df.dataset=='C2DB',col].dropna())
        ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=COLORS[2],lw=1.7)
        ax.set_xlim(left=0)
    for ax,title,xlabel in zip(axs,['a  3D crystals','b  2D in-plane cell','c  2D out-of-plane extent'],
                              [r'Volume per atom ($\mathrm{\AA}^3$, log axis)',r'Area per atom ($\mathrm{\AA}^2$)',r'Layer thickness ($\mathrm{\AA}$)']):
        ax.set_title(title,loc='left');style_axes(ax,xlabel,'Cumulative fraction')
        ax.set_ylim(0,1.02);ax.yaxis.set_major_formatter(PercentFormatter(1))
    savefig(fig,'07_lattice_and_layer_geometry','由存档k和群相容度量解码周期测度，V或A=sqrt(det H)。3D体积/原子与2D面积/原子分开；2D层厚=max(h)-min(h)，不包含真空，也不加范德华原子半径。保留完整分布，3D体积横轴用对数。')

    fig,ax=plt.subplots(figsize=(8,3.4),layout='constrained')
    ax.imshow(coverage,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    ax.set_xticks(range(3),['Formation energy','Band gap','Above-hull energy'])
    ax.set_yticks(range(3),NAMES)
    ax.set_title('Property-label coverage (%)')
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f'{coverage[i,j]:.1f}',ha='center',va='center',color='white' if coverage[i,j]>65 else '#243746')
    fig.savefig(OUT/'property_coverage_recipe.png',dpi=260,facecolor='white')
    plt.close(fig)

    metadata={'archive':str(args.archive),'source_files':source_checks,'rows':len(df),
        'validation':'All records checked: state DOF and expanded atom count match group tables.',
        'dataset_summary':summary,'captions':CAPTIONS,
        'scope':'Prepared source records across all original splits; shared splits counted separately. No structural matching or property re-evaluation.'}
    (OUT/'tables/statistics.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    text=['# 数据集统计与图注','',
        '统计对象是服务器备份中实际用于预处理的MP20、MPTS52、C2DB来源记录；全量统计合并原始训练、验证、测试。联合模型采用另行分组划分，图1单独列出。','',
        '所有102610条记录均核对了群表轨道重数、位置自由度、晶格自由度与state长度。这里的全量覆盖统计不能替代训练集覆盖统计。','',
        '| 数据集 | 记录数 | 群覆盖 | 元素种类 | 未装饰轨道组合 | 原始/标准胞原子数中位数 | 状态维数中位数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:text.append(f"| {r['dataset']} | {r['records']:,} | {r['groups_observed']}/{r['groups_catalog']} | {r['unique_elements']} | {r['orbit_patterns_undecorated']} | {r['source_atoms_median']:g}/{r['expanded_atoms_median']:g} | {r['state_dof_median']:g} |")
    for name,caption in CAPTIONS.items():text += ['',f'## {name}','',caption,'',f'![{name}]({name}.png)']
    text += ['', '## 可复现性与口径','',
        '- `source/build_dataset_figures.py`生成所有图，`tables/`保留逐记录压缩CSV、汇总表、标签覆盖、元素频率、群分布和交集计数。',
        '- 轨道组合定义为群号及带重复的Wyckoff letter排序列表，不含元素或连续参数；不同晶胞设置不混用。',
        '- 绘图只输出PNG，使用Times New Roman；旧PDF/SVG已移出当前图集并归档。',
        '- C2DB这里是本项目实际恢复的16905条记录子集，不是宣称覆盖整个实时C2DB数据库。',
        '- 结构数是来源记录数。MP20与MPTS52存在来源重叠，直接相加不等于独立材料总数。',
        '- 同一材料的群号、Wyckoff位点及原子数取决于标准胞和识别容差。配置容差为MP20 0.001 Å、MPTS52 0.01 Å、C2DB层群识别0.1 Å。',
        '- 形成能/凸包能单位为eV/atom、带隙为eV；分布来自存档标签，没有重新计算或统一跨来源的能量参考。',
        '- 图5反映记录中存储的标签，不代表每个参考配置都使用了全部标签；如MP20参考配置只条件化形成能和带隙。',
        '- 图6中不同数据源ID交集以精确字符串为准，MP数据源内可解释为共享材料ID；这不是原型去重或结构匹配结果。']
    (OUT/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
