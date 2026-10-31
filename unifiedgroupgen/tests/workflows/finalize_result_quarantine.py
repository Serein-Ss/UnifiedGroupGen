"""One-time current-only result curation after the explicit quarantine has verified."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/'old_results'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_text(relative, value):
    path=ROOT/relative;path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding='utf-8')


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def retire_file(relative, backup=None):
    source=(ROOT/relative).resolve();target=(OLD/(backup or relative)).resolve()
    assert source.is_relative_to(ROOT) and target.is_relative_to(OLD) and not target.exists()
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True);source.rename(target)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==before
    with (OLD/'late_moves.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'source':relative,'archive':target.relative_to(OLD).as_posix(),
                                 'sha256':before,'state':'verified'},ensure_ascii=False)+'\n')


def inventories(snapshot):
    models=snapshot['models']
    inventory={'scope':'Current main15 models only; checkpoint and history evidence, not blanket generation-quality certification.',
        'model_count':len(models),'checkpoint_count':2*len(models),'models':models,
        'total_completed_epochs':sum(m['completed_epochs'] for m in models),
        'evaluated_model_names':['mp20_u','c2db51_u','joint_u'],
        'evaluation_scope':'The retained 256-scale evaluation is hash-bound to these three best checkpoints only.',
        'import_date_note':'Date directories record import time, not training start time.'}
    write_json(ROOT/'outputs/model_inventory.json',inventory)
    write_json(ROOT/'assets/results/main15/model_inventory.json',inventory)
    fields=['name','directory','status','completed_epochs','best_epoch','best_val_loss','validation_points',
            'train_records','val_records','descriptor_parameters','flow_parameters','requires_source_context',
            'target_properties','best_checkpoint_sha256','last_checkpoint_sha256','date_note']
    for path in (ROOT/'outputs/model_inventory.csv',ROOT/'assets/results/main15/model_inventory.csv'):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for model in models:
                row={k:model[k] for k in fields};row['target_properties']=';'.join(model['target_properties'])
                writer.writerow(row)
    rows=[]
    for model in models:
        for line in (ROOT/model['directory']/'model.history.jsonl').read_text().splitlines():
            if not line.strip():continue
            entry=json.loads(line);row={'model':model['name'],'epoch':entry['epoch']+1}
            for phase in ('train','val'):
                for key,value in (entry.get(phase) or {}).items():row[phase+'_'+key]=value
            for key in ('train_loss','val_loss','training_seconds','validation_seconds','checkpoint_seconds','wall_seconds',
                        'learning_rate','peak_cuda_bytes','peak_cuda_reserved_bytes','coordinate_length_scale_angstrom'):
                row[key]=entry.get(key)
            rows.append(row)
    path=ROOT/'assets/training/main15/training_history.csv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','epoch',*sorted({k for r in rows for k in r}-{'model','epoch'})])
        writer.writeheader();writer.writerows(rows)
    assert len(rows)==inventory['total_completed_epochs']
    return inventory


def curate_assets(inventory):
    before=load(OLD/'assets/results/main15_baselines_20261025/summary.json')
    hashes={m['name']:m['best_checkpoint_sha256'] for m in inventory['models']}
    assert all(hashes[m['name']] in json.dumps(m) for m in before['models'])
    for name in ('01_training_losses.png','02_continuous_and_gradients.png','03_runtime_and_memory.png',
                 'training_history.csv','training_summary.csv'):
        copy_file(OLD/'assets/results/main15_baselines_20261025'/name,ROOT/'assets/training/main15_baselines'/name)
    write_json(ROOT/'assets/training/main15_baselines/provenance.json',{
        'models':['mp20_u','c2db51_u','joint_u'], 'best_checkpoints':{n:hashes[n] for n in ('mp20_u','c2db51_u','joint_u')},
        'scope':'Unchanged figures for the three retained baseline models only; not curves for all 15 models.'})
    write_text('assets/training/README.md','# 当前训练结果\n\nmain15/training_history.csv包含新版15套模型的2520轮原始日志展开；完整日志与权重保存在outputs的各实验日期目录。main15_baselines中的3张PNG只展示其中3个基础模型，检查点哈希已核对，不代表15套曲线。\n')
    manifest=load(OLD/'assets/structures/manifest.json')
    manifest['figures']=[f for f in manifest['figures'] if f['scope']=='Real encoded dataset example']
    write_json(ROOT/'assets/structures/manifest.json',manifest)
    write_text('assets/structures/README.md','# 真实结构示例\n\n当前仅保留TiO₂与MoS₂真实数据示例，共4个视图。它们是编码后的真实结构，不是模型生成结构。新版基础模型的生成结构总图在assets/results/baselines_256_local_seed142_20261026/06_generated_structures.png。\n')
    write_json(ROOT/'assets/paper/figure_data_manifest.json',{
        'scope':'Current main15 only; theory/examples are independent of learned checkpoints.',
        'method':['fig01_unified_method.png','fig01_material_examples.png'],
        'model_inventory':'../results/main15/model_inventory.json',
        'verified_baseline_results':'../results/baselines_256_local_seed142_20261026/manifest.json',
        'other_model_quality':'pending; no old-model values substituted'})
    write_text('assets/paper/README.md','# 当前文章图件\n\n图1保留统一方法与TiO₂/MoS₂真实示例。数据图在assets/datasets；训练表在assets/training/main15；3个基础模型的6张生成评估图在assets/results/baselines_256_local_seed142_20261026。其余12套模型的生成质量与性质响应待补。没有用撤出的结果填补这些空缺。\n')
    write_text('assets/README.md','# 当前图表范围\n\n保留统一方法图、真实数据集图、来源一致性数据、新版15套模型的训练清单，以及检查点哈希匹配的3个基础模型评估。训练PNG仅覆盖3个基础模型；15套完整训练日志见outputs，2520轮表见training/main15。旧模型图表、视频和混合汇总已撤出当前目录。\n')
    write_text('assets/results/RESULTS_PLAN.md','# 当前结果与待补内容\n\n新版15套均已训练完成；当前可引用的独立生成质量只覆盖mp20_u、c2db51_u、joint_u，合计5404次生成与24854条测试请求。其余12套的模块测试、CSP、从头生成、逐群控制与性质评价待补。训练完成不等于这些性能已评价通过。\n')


def source_audit_without_model_results():
    for name in ('summary.json','README.md','01_full_source_audit.png'):
        if (ROOT/'assets/source_integrity'/name).exists():retire_file('assets/source_integrity/'+name)
    original=load(OLD/'assets/source_integrity/summary.json')
    summary={k:v for k,v in original.items() if k not in ('model_training_affected','models_require_clean_retraining',
        'clean_revision_models_trained','joint_historical_split_impact')}
    write_json(ROOT/'assets/source_integrity/summary.json',summary)
    write_text('assets/source_integrity/README.md','# 当前数据来源一致性\n\n保留全部原始CIF审计明细、68条处理清单及修订后102608条可用记录的摘要。这里评价的是输入数据的组成和结构等价性，不是训练模型的性能。包含旧模型影响柱状图的图件已撤出。\n')
    return summary


def update_documents(inventory):
    original=(OLD/'docs/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md').read_text(encoding='utf-8')
    start=original.index('<!-- BASELINE256_RESULTS_START -->');end=original.index('<!-- BASELINE256_RESULTS_END -->')+len('<!-- BASELINE256_RESULTS_END -->')
    baseline=original[start:end].replace('### 9.11','### 9.6').replace('第9.9节','第9.5节')
    rows='\n'.join(f"| {m['name']} | {m['completed_epochs']} | {m['best_epoch']} | {m['best_val_loss']:.6f} | {'、'.join(m['target_properties']) or '无目标性质'} |" for m in inventory['models'])
    section9='''## 9 Data and Current Evidence 数据与当前结果

本章只使用当前保留的15套模型及其哈希绑定的评估，不使用其他模型的指标补缺。

### 9.1 数据与实验范围

主实验为MP20、C2DB及两者联合，每种数据设置包含基础、形成能、带隙、凸包能及三性质联合5种训练配置，共15套。MPTS52仍保留为数据来源审计与统计参考，不属于当前15套训练。

| 数据设置 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| MP20 | 27136 | 9047 | 9046 |
| C2DB | 10143 | 3381 | 3381 |
| MP20+C2DB | 37279 | 12428 | 12427 |

### 9.2 来源与编码验证

全量来源审计处理102610条输入，其中66条修复、2条MPTS52输入隔离，得到102608条核验记录。原始数据、修复明细与隔离记录保留。当前15套配置的数据门禁已通过；这项证据不代替生成结构评价或物理稳定性验证。数据图见附录D。

### 9.3 条件与共享边界

五套joint模型仍将source_domain作为必要上下文输入两个网络。基础模型不输入目标性质，但联合基础模型仍不能称为完全无来源条件生成。性质可见掩码统一与来源输入移除仍是后续建模调整，不在本次结果清理中实施。

### 9.4 模型与检查点身份

15套模型分别保留最优和末轮检查点、完整训练历史、状态、日志与导入溯源。30份检查点哈希与原清单一致，15份最优检查点严格加载通过。日期目录使用导入时间，非训练开始时间。完整清单见[模型索引](../outputs/model_inventory.csv)。

### 9.5 十五套模型的训练

累计2520轮。下表为同一次训练的完成轮次与按总验证损失选择的最佳轮次，不拼接两个模块各自最优的时刻。不同数据与性质配置的损失不能仅按绝对值排名。

| 模型 | 完成轮次 | 最优轮次 | 最优验证损失 | 目标性质 |
|---|---:|---:|---:|---|
'''+rows+'''

全部逐轮结果见[2520轮训练表](../assets/training/main15/training_history.csv)。下图仅覆盖其中3个基础模型，不代表全部15套曲线。

![三个当前基础模型的总目标、描述Transformer和OrbitFlow训练曲线](../assets/training/main15_baselines/01_training_losses.png)

'''+baseline+'''

### 9.7 尚未补齐的性能评价

其余12套模型的模块测试、指定组成CSP、从头生成、指定群及性质控制评价待补；没有将其他模型的已有指标迁移为它们的结果。性质性能需要独立预测器，物理验证另行设计。

'''
    text=original[:original.index('## 9 Data and Current Evidence')]+section9+original[original.index('## 10 Experiments'):]
    text=text[text.index('## 摘要'):]
    abstract='''# UnifiedGroupGen 基于群作用与轨道自由度的统一对称性约束材料生成

**当前模型限定稿 · 2026-10-30**

当前结果只对应MP20、C2DB及两者联合的15套模型。数据集、数学推导与真实结构示例保留；此前混合展示的其他模型结果已撤出。联合模型仍保留来源上下文，未把这一建模限制隐藏为完全无条件生成。

## 摘要

本工作将固定群作用与轨道规则编译为合法连续状态，由共享OrbitFlow在展开原子图上预测候选变化，并归约为合法独立速度。位置采用仿射轨道表示，相容晶格采用不变对数度量。以条件流匹配实现，我们在MP20、C2DB及两者联合数据上完成15套训练，比较无目标性质、单性质与多性质训练配置。全量来源一致性审计得到102608条核验记录，其中主实验使用MP20和C2DB。当前独立生成评价只覆盖3个基础模型，包括完整测试请求、5404次生成、指定组成与固定描述重建及逐群诊断。其余模型的生成质量、性质响应及公平外部对照仍待补齐；训练完成不等于这些性能已验证。

'''
    text=abstract+text[text.index('**关键词：**'):]
    text=text.replace('在MP_20、MPTS_52和C2DB上验证该构造','在MP_20和C2DB上验证该构造')
    c0=text.index('## 12 Conclusion');c1=text.index('## 附录 A',c0)
    text=text[:c0]+'''## 12 Conclusion 总结

当前保留15套完整模型及与其中3个基础检查点绑定的生成评价。群—轨道构造、数据来源一致性、学习效果和材料质量分别提供证据。联合模型仍保留来源条件；其余12套生成性能和性质控制结果尚未补齐，不能由训练损失或不同模型的结果替代。

'''+text[c1:]
    d0=text.index('## 附录 D');d1=text.index('## 附录 F',d0)
    catalog=load(OLD/'web/symmetry-lab/gallery_catalog.json')
    dataset_figures=[f for f in catalog['figures'] if f['section']=='数据集']
    appendix='## 附录 D 当前数据图\n\n'
    for i,item in enumerate(dataset_figures,1):
        appendix+=f"### D.{i} {item['title']}\n\n![{item['content']}](../assets/{item['asset']})\n\n{item['result']}\n\n"
    appendix+='''## 附录 E 当前证据与待补结果

| 内容 | 当前状态 |
|---|---|
| 统一方法与真实示例 | 图1与TiO₂/MoS₂推导保留 |
| 数据统计与来源检查 | 保留修订后数据图与原始明细 |
| 15套训练 | 2520轮日志、30份检查点与完整清单 |
| 3个基础模型评价 | 模块、CSP、固定描述、从头生成、逐群诊断及6张PNG |
| 其余12套质量与性质控制 | 待补，不使用其他模型结果代替 |
| 匹配基线、消融与物理验证 | 待补 |

'''
    text=text[:d0]+appendix+text[d1:]
    text=text.replace('第9.9节','第9.5节').replace('第9.11节','第9.6节')
    write_text('docs/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md',text)
    write_text('docs/TRAINING_OUTPUT_ARCHIVE.md','# 当前15套模型\n\n唯一权重目录为outputs/<实验名称>/<导入日期>/。每套包含model.best.pt和model.pt；两者是同一次训练的检查点，不重复计为模型。原导入清单、日志和来源配置保留。当前15套完整清单见[模型索引](../outputs/model_inventory.csv)，日期不是训练开始时间。\n')
    write_text('docs/EXISTING_OUTPUT_LOCATIONS.md','# 当前结果位置\n\n| 内容 | 位置 |\n|---|---|\n| 15套权重与完整日志 | outputs/<实验>/<日期>/ |\n| 模型身份和检查点哈希 | outputs/model_inventory.json与CSV |\n| 2520轮合并训练表 | assets/training/main15/training_history.csv |\n| 基础模型训练PNG | assets/training/main15_baselines/（仅3套） |\n| 基础模型原始生成与指标 | artifacts/evaluations/baselines_256_local_seed142_20261026/ |\n| 6张基础模型结果PNG和表 | assets/results/baselines_256_local_seed142_20261026/ |\n| 撤出的内容 | old_results/（仅待核对归档，不作为当前证据） |\n')
    write_text('docs/PAPER_EVIDENCE_PLAN.md','# 当前文章证据计划\n\n以中文论文长稿第9节为准，只使用当前15套模型。训练信息15套齐全；生成指标只有3个基础模型已完成。下一步补齐其余12套的模块、CSP、从头生成、逐群及性质任务，再做同预算基线与消融。不得把训练完成、规则保持或计算成功表述为性质命中或稳定性。\n')
    write_text('docs/MODEL_SCALE.md','# 当前15套模型参数\n\n| 模型 | 描述Transformer | OrbitFlow | 总参数 |\n|---|---:|---:|---:|\n'+'\n'.join(f"| {m['name']} | {m['descriptor_parameters']} | {m['flow_parameters']} | {m['descriptor_parameters']+m['flow_parameters']} |" for m in inventory['models'])+'\n\n数量由实际最优检查点配置构造模型后统计，并严格加载权重验证；编译器没有训练参数。参数量本身不代表生成质量或计算速度。\n')
    write_text('outputs/README.md','# 当前模型结果\n\n仅保留15套正式模型，各自最优／末轮权重与日志不变。model_inventory.json和CSV给出身份、哈希、轮次、性质及实际参数。日期目录为导入时间，不是训练开始时间。2520轮表见assets/training/main15。其余12套尚未得到生成质量评价。\n')


def update_site(inventory):
    site=ROOT/'web/symmetry-lab';previous=OLD/'web/symmetry-lab'
    raw=(previous/'dist/data.js').read_text(encoding='utf-8')
    payload=json.loads(raw[raw.index('=')+1:].strip().rstrip(';'))
    write_json(site/'materials.json',payload['materials'])
    catalog=load(previous/'gallery_catalog.json')
    selected=[]
    for item in catalog['figures']:
        item=dict(item)
        if item.get('evidence_family')=='main15_baseline':
            item['asset']=item['asset'].replace('results/main15_baselines_20261025/','training/main15_baselines/')
        elif item.get('evidence_family')!='baseline256' and item['section'] not in ('数据集','模型架构'):
            continue
        if item['section'] in ('数据集','模型架构'):item['models']=[]
        if (ROOT/'assets'/item['asset']).is_file():selected.append(item)
    for section in catalog['sections']:
        if section['id']=='training':
            section.update(heading='3  当前15套模型训练',intro='15套均完成训练；表格包含全部模型，PNG曲线仅覆盖3个基础模型。')
        elif section['id']=='generation':
            section.update(intro='仅展示与当前3个基础模型检查点绑定的评价；其余12套保持待补。')
        elif section['id']=='pending':
            section.update(intro='其余12套的生成质量、性质响应、公平基线和物理验证待补；不填入其他模型的结果。')
    catalog.update(updated='2026-10-30',figures=selected)
    write_json(site/'gallery_catalog.json',catalog)
    copy_file(Path(__file__).with_name('current_site_export.py'),site/'export_data.py')
    out=site/'dist';out.mkdir(exist_ok=True)
    (out/'tables').mkdir(exist_ok=True)
    for name in ('h2_orbit_derivations.json','h3_metric_derivations.json'):
        copy_file(previous/'dist/tables'/name,out/'tables'/name)
    app=(previous/'dist/app.js').read_text(encoding='utf-8')
    bodies={
        'renderModelInventory':"function renderModelInventory(){const target=$('trained-models');if(target)target.innerHTML=currentTrainingHTML(D.latest_snapshot);}\n",
        'renderGenerationSettings':"function renderGenerationSettings(){const target=$('generation-settings');if(target)target.innerHTML=currentGenerationHTML();}\n",
        'renderConstructionNotes':'''function renderConstructionNotes(){const target=$('construction-notes');if(!target)return;const a=D.source_integrity;target.innerHTML=`<div class="metric-explanation"><h3>数据来源与编码验证</h3><p>来源一致性审计核对102610条输入，修复66条，隔离2条，保留102608条核验记录。这是数据与表示证据，不是生成质量或物理稳定性。</p><div class="table-scroll"><table class="model-table"><thead><tr><th>来源</th><th>原始输入</th><th>需处理</th><th>隔离</th><th>核验记录</th></tr></thead><tbody>${Object.entries(a.datasets).map(([name,r])=>`<tr><th>${name}</th><td>${r.records}</td><td>${r.affected}</td><td>${r.errors}</td><td>${r.repaired_passed}</td></tr>`).join('')}</tbody></table></div><p>当前15套训练数据门禁通过。群操作保持、精确群、化学合理性与物理稳定性分别评价。</p><a href="tables/source_integrity_summary.json" download>下载来源核验摘要</a></div>`;}\n'''}
    for name,next_name in (('renderModelInventory','renderGenerationSettings'),('renderGenerationSettings','renderConstructionNotes'),('renderConstructionNotes','setupGallery')):
        pattern=r'function '+name+r'\(\)\{[\s\S]*?(?=function '+next_name+r'\()'
        app,count=re.subn(pattern,lambda _:bodies[name]+'\n',app,count=1);assert count==1,name
    step='''if(step===5){explanation=`<h3>同一模型在合法空间中学习</h3><p>训练输入真实描述与独立状态，编译器构造合法路径；OrbitFlow预测候选速度，再归约到允许的晶格和位置变化。当前页面仅计算数学展开，不在浏览器运行神经网络。</p><div class="equation">a → 编译器 → Φₐ(zₜ) → OrbitFlow → 合法速度 → 积分</div><p>当前保留15套模型。基础模型不输入目标性质；联合模型仍有来源上下文。生成评价只覆盖其中3个基础检查点，其他模型不借用其结果。</p>`;numeric=`<h3>当前证据范围</h3><table><tr><th>完成训练</th><td>${D.models.length}套</td></tr><tr><th>累计训练</th><td>${D.latest_snapshot.total_completed_epochs}轮</td></tr><tr><th>已有基础评价</th><td>3套／5404次生成</td></tr><tr><th>其他生成质量</th><td>12套待补</td></tr></table><button class="primary jump-gallery-inline">查看当前结果图集 →</button>`;}'''
    app,count=re.subn(r'^if\(step===5\)\{.*\}$',lambda _:step,app,count=1,flags=re.M);assert count==1
    app=app.replace("if(!figures.length)return '';","if(!figures.length&&!(['construction','pending'].includes(section.id)&&(category==='全部'||category===section.title)&&!term))return '';")
    (out/'app.js').write_text(app,encoding='utf-8')
    current=r''' 'use strict';
function currentTrainingHTML(snapshot){
 const names={formation_energy:'形成能',band_gap:'带隙',energy_above_hull:'凸包能'};
 return `<h3>当前保留15套完整模型</h3><p>累计${snapshot.total_completed_epochs}轮；每套保留最优和末轮检查点。日期目录是导入时间，不是训练开始时间。</p><div class="table-scroll"><table class="model-table"><thead><tr><th>模型</th><th>性质</th><th>完成／最优轮次</th><th>最优验证损失</th><th>描述／连续参数</th><th>来源上下文</th></tr></thead><tbody>${snapshot.models.map(m=>`<tr data-model-id="${m.name}"><th>${m.name}</th><td>${m.target_properties.map(p=>names[p]||p).join('＋')||'无目标性质'}</td><td>${m.completed_epochs}／${m.best_epoch}</td><td>${m.best_val_loss.toFixed(6)}</td><td>${m.descriptor_parameters.toLocaleString()}／${m.flow_parameters.toLocaleString()}</td><td>${m.requires_source_context?'保留':'无'}</td></tr>`).join('')}</tbody></table></div><p>三套基础模型已完成独立生成评价；其余12套不能仅因训练完成就称为性能通过。下方训练PNG只覆盖3个基础模型，全部15套日志展开为2520轮CSV。</p><p><a href="tables/model_inventory.csv" download>模型清单</a> · <a href="tables/training_history.csv" download>全部训练轨迹</a> · <a href="tables/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md" download>当前文章</a></p>`;
}
function currentGenerationHTML(){return baselineEvaluationHTML(window.UGG_DATA.baseline_evaluation);}
'''
    (out/'current_results.js').write_text(current,encoding='utf-8')
    html=(previous/'dist/index.html').read_text(encoding='utf-8')
    html=re.sub(r'<p>历史三来源联合模型.*?</p>', '<p>当前保留15套完整模型，来源、性质、轮次与检查点身份见研究图集。生成评价只覆盖3个基础模型；群操作保持、精确群与材料质量分开报告。</p>',html)
    html=re.sub(r'(<section id="gallery-view".*?<p class="lede">).*?(</p>)',r'\1只展示当前15套模型与核验过的结果。3个基础模型已有模块、CSP、从头生成和逐群评价，其余12套待补。\2',html,flags=re.S)
    html=html.replace('结果归档 · 2026.10.26','当前15套 · 2026.10.30').replace('结果归档 2026-10-26 · 新基础完整评估与历史结果分开报告','当前15套 · 仅引用核验过的对应结果')
    (out/'index.html').write_text(html,encoding='utf-8')
    theory=(previous/'theory_content.md').read_text(encoding='utf-8')
    theory=re.sub(r'^目前已有102,610条准备记录.*$', '数学展开与投影由真实TiO₂/MoS₂的参考状态检查；数据来源审计与生成评价分别报告。当前模型与检查点身份以研究图集中的15套清单为准。',theory,flags=re.M)
    theory=re.sub(r'^训练结果已更新为245轮.*$', '当前15套训练已完成。只有3个基础模型的生成评价已核验，不将训练完成解释为性质或稳定性通过。',theory,flags=re.M)
    (site/'theory_content.md').write_text(theory,encoding='utf-8')
    (site/'qa').mkdir(exist_ok=True)
    write_text('web/symmetry-lab/README.md','# 本地对称性实验室\n\n交互、H.1/H.2/H.3推导和真实TiO₂/MoS₂示例保留。研究图集只使用当前15套模型；生成质量只覆盖3个基础模型。\n\n从项目根目录执行python web/symmetry-lab/export_data.py，再执行node web/symmetry-lab/build_theory.mjs。export_data读取当前模型索引、图集和材料参数，不需要撤出的结果。此轮只更新本地，未发布线上站点。\n')


def current_readmes(inventory):
    retire_file('README.md','before_edit/README.md')
    write_text('README.md','''# UnifiedGroupGen

通过群—轨道编译与流匹配生成三维晶体和二维材料。当前仅保留MP20、C2DB及两者联合的15套完整模型。

```bash
conda env create -f unifiedgroupgen/environment.yml
conda activate unifiedgroupgen
python -m pip install --no-deps --no-build-isolation -e ./unifiedgroupgen
ugg-train experiment=mp20_u task=plan
ugg-generate --help
ugg-evaluate --help
```

模型权重、日志及导入溯源在outputs/<实验>/<日期>/，日期为导入时间。模型清单见[model_inventory.csv](outputs/model_inventory.csv)。各模型最优和末轮检查点属于同一次训练，共15套、30份权重。

只有3个基础模型已有哈希绑定的独立生成评价：原始结果在artifacts/evaluations/baselines_256_local_seed142_20261026，6张PNG与表在assets/results/baselines_256_local_seed142_20261026。其余12套的生成质量和性质控制待补。

[当前文章](docs/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md) · [代码组织](docs/CODE_ORGANIZATION.md) · [环境](docs/ENVIRONMENT.md) · [实验配置](docs/EXPERIMENT_CONFIGURATION.md) · [当前结果位置](docs/EXISTING_OUTPUT_LOCATIONS.md)

数据、真实结构示例、群约束和模型代码保留。当前5套联合模型仍有source_domain上下文，这一建模边界没有在目录清理中改变。
''')
    copy_file(ROOT/'.gitignore',OLD/'before_edit/.gitignore')
    with (ROOT/'.gitignore').open('a',encoding='utf-8') as stream:stream.write('\nold_results/\n')
    previous=load(OLD/'outputs/tests/fresh_package_cleanup/26-10-30-20-16-37/verification.json')
    write_json(ROOT/'outputs/validation/code_verification_summary.json',{
        'scope':'Software verification summary, independent of model-quality evidence.',
        'tests':previous['tests'],'paired_probe':previous['paired_probe'],
        'environment':{'windows_local_verified':True,'linux_fresh_install_verified':False},
        'raw_records_quarantined':True})
    write_text('docs/VALIDATION.md','# 当前验证范围\n\n当前保留15套模型的30份检查点哈希，15份最优权重严格加载通过。3个基础模型的256规模评价与当前检查点哈希一致。工程验证摘要见outputs/validation/code_verification_summary.json；它不是材料生成质量证据。此轮结果整理的检查与迁移日志保存在隔离目录，最终检查会验证模型、数据门禁、文章及本地网页不依赖被撤出的结果。\n')



def cleanup_other_documents():
    copy_file(ROOT/'assets/architecture/source/architecture_spec.md',OLD/'before_edit/assets/architecture/source/architecture_spec.md')
    write_text('assets/architecture/source/architecture_spec.md', "# 当前统一方法示意图\n\n保留的PNG只表达方法，不是训练或生成质量结果。当前15套配置的描述Transformer为4层、256通道，OrbitFlow为6层、512通道，与图中骨架标注一致；实际参数量见outputs/model_inventory.csv。\n\n训练连续模型使用真实描述，采样从先验开始。群—轨道编译器给出F=b+Bq和参考度量白化后的不变对数晶格空间。候选速度经合法映射后积分。二维使用面内2×2度量，3×3示意不表示对辅助真空做三维周期处理。图中的组成入口表示可选的采样约束，不代表当前描述网络已学得组成条件分布；联合版还保留来源上下文，完整输入边界以文章为准。\n\n原子图为示意，不是稳定材料证据。TiO₂与MoS₂真实参数及交互推导在examples和web/symmetry-lab。\n")
    for name in ('README.before_cleanup.md','JOINT_EVALUATION.md','LEGACY_4096_EVALUATION.md',
                 'PAPER_WORKFLOW.md','VALIDATION.md'):
        retire_file('unifiedgroupgen/tests/legacy/docs/'+name)
    for relative in ('docs/SOURCE_INTEGRITY_AUDIT_20261022.md','docs/EVALUATION_PROTOCOL_V2.md'):
        copy_file(ROOT/relative,OLD/'before_edit'/relative)
    path=ROOT/'docs/SOURCE_INTEGRITY_AUDIT_20261022.md'
    text=path.read_text(encoding='utf-8')
    text=text[:text.index('## 5. 对历史模型的影响')]
    text=text.replace('三个来源的原始CSV、原始编码目录、旧划分、检查点及历史生成输出均保留。新目录为artifacts/full_verified_v1与artifacts/joint_verified_v1。旧配置仍可用于解释历史检查点，正式训练门禁已禁止旧数据目录。',
        '原始来源与逐记录审计保留。正式数据位于data/dataset/full_verified_v1与data/dataset/joint_verified_v1；当前模型清单与配置见outputs/model_inventory.json和configs/experiments.yaml。')
    text+='## 5. 当前证据位置\n\n原始核对记录与修复明细在assets/source_integrity。当前15套模型使用修订后MP20、C2DB或两者联合数据；MPTS52只作为数据审计参考。逐群闭合与来源结构等价性是不同的检查，均不能代替生成质量或物理稳定性。\n'
    path.write_text(text,encoding='utf-8')
    path=ROOT/'docs/EVALUATION_PROTOCOL_V2.md';text=path.read_text(encoding='utf-8')
    a=text.index('## “历史”指什么');b=text.index('## 部分结构补全',a)
    text=text[:a]+'## 当前结果与协议\n\n当前模型身份以outputs/model_inventory.json为准。只有3个基础模型已有哈希绑定的5404次生成评估；其余12套生成结果待补。保留旧几何检查与匹配参数作为可复现的指标协议，不引用撤出的模型数值。\n\n'+text[b:]
    path.write_text(text,encoding='utf-8')


def main():
    assert load(OLD/'migration.json')['state']=='verified'
    snapshot=load(OLD/'_audit/retained_models_before.json')
    inventory=inventories(snapshot)
    curate_assets(inventory)
    source_audit_without_model_results()
    update_documents(inventory)
    update_site(inventory)
    current_readmes(inventory)
    cleanup_other_documents()
    print(json.dumps({'models':15,'training_epochs':inventory['total_completed_epochs'],'current_baseline_evaluation_retained':True},ensure_ascii=False))


if __name__=='__main__':main()
