"""Export current model identities, verified baseline results and real-material fixtures."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

from unifiedgroupgen.common.configuration import project_root


ROOT=project_root()
SITE=ROOT/'web/symmetry-lab'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)


def build():
    out=SITE/'dist';out.mkdir(exist_ok=True)
    inventory=load(ROOT/'outputs/model_inventory.json')
    catalog=load(SITE/'gallery_catalog.json')
    materials=load(SITE/'materials.json')
    baseline=ROOT/'assets/results/baselines_256_local_seed142_20261026'
    binding=load(baseline/'manifest.json')['study_identity']['checkpoints']
    assert all(next(m['best_checkpoint_sha256'] for m in inventory['models'] if m['name']==name)==value
               for name,value in binding.items())
    gallery=[]
    for item in catalog['figures']:
        source=ROOT/'assets'/item['asset']
        assert source.is_file(),source
        target=out/'figures'/item['asset'];copy(source,target)
        label=('真实数据／结构示例' if item['section']=='数据集' else '统一模型方法' if item['section']=='模型架构'
               else '当前基础模型：mp20_u／c2db51_u／joint_u；联合版保留来源条件')
        gallery.append({**item,'category':item['section'],'status':'真实数据' if item['section']=='数据集' else '已完成',
            'historical':False,'path':'figures/'+item['asset'],'original':'assets/'+item['asset'],
            'sha256':digest(source),'model_label':label,
            'caption':'内容：'+item['content']+' 结果：'+item['result']+' 边界：'+item['limits']})
    copy(ROOT/'examples/mp20_walkthrough/source.cif',out/'figures/mp20_source.cif')
    copy(ROOT/'examples/c2db_walkthrough/source.cif',out/'figures/mos2_source.cif')
    tables=out/'tables';tables.mkdir(exist_ok=True)
    for name in ('model_inventory.json','model_inventory.csv'):copy(ROOT/'outputs'/name,tables/name)
    copy(ROOT/'assets/training/main15/training_history.csv',tables/'training_history.csv')
    copy(ROOT/'docs/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md',tables/'UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md')
    for path in baseline.iterdir():
        if path.is_file() and (path.suffix=='.csv' or path.name in ('manifest.json','validation.json','RESULTS_ZH.md')):
            copy(path,tables/('baseline256_'+path.name))
    for name in ('summary.json','affected_records.csv','all_records.csv.gz'):
        copy(ROOT/'assets/source_integrity'/name,tables/('source_integrity_'+name))
    def rows(name):
        with (baseline/name).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
    evaluation={'validation':load(baseline/'validation.json'), 'component_tests':rows('component_tests.csv'),
        'generation_quality':rows('generation_quality.csv'),'paired_testcase_comparison':rows('paired_testcase_comparison.csv')}
    payload={'models':inventory['models'],'pending_models':[],'date':'2026-10-30','annotation_date':'2026-10-30',
        'sections':catalog['sections'],'latest_snapshot':inventory,'baseline_evaluation':evaluation,
        'materials':materials,'gallery':gallery,'source_integrity':load(ROOT/'assets/source_integrity/summary.json'),
        'scope':'Current 15 models; verified generation evaluation only for the three bound baselines. No other model results substituted.'}
    (out/'data.js').write_text('window.UGG_DATA = '+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    (out/'data_manifest.json').write_text(json.dumps({'model_inventory':inventory,'figures':gallery,
        'baseline_checkpoints':binding,'tables':{p.name:digest(p) for p in tables.iterdir() if p.is_file()}},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (SITE/'model_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'models':len(inventory['models']),'materials':len(materials),'figures':len(gallery),
                      'baseline_attempts':evaluation['validation']['sample_attempts_verified']},ensure_ascii=False))


if __name__=='__main__':build()
