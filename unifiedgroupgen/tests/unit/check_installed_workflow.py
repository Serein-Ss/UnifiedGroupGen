"""Run a two-sample fixture through an independently installed wheel's workflows."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args();root=args.workspace.resolve();root.mkdir(parents=True,exist_ok=True)
    os.environ['UGG_PROJECT_ROOT']=str(root)
    from unifiedgroupgen.pl_modules.factory import build_models

    from unifiedgroupgen.common.structures import write_json
    import unifiedgroupgen
    torch.set_num_threads(1);torch.manual_seed(142)
    record={'id':'fixture','dataset':'mp_20','properties':{},
            'descriptor':{'kind':'space','number':2,'orbits':[{'letter':'i','atomic_number':6}]},
            'state':[0.]*6+[.2,.3,.4]}
    source=root/'data/dataset/fixture.jsonl';source.parent.mkdir(parents=True,exist_ok=True)
    source.write_text(json.dumps(record)+'\n',encoding='utf-8')
    config={'experiment_id':'mp20_u','dataset':'mp_20','kind':'space','groups':[['space',2]],
            'properties':{},'composition_probability':0.,'observation_probability':0.,
            'allowed_elements':[6],'max_atoms':2,'max_orbits':1,'require_data_audit':False,
            'model':{'proposal_hidden':16,'proposal_layers':1,'heads':2,'flow_hidden':16,'flow_layers':1,'frequencies':2}}
    for split in ('train','val','test'):config[split+'_files']=[str(source)]
    (root/'configs').mkdir(exist_ok=True)
    write_json(root/'configs/fixture.yaml', config)
    write_json(root/'configs/experiments.yaml', {'experiments': {'mp20_u': {'config': 'configs/fixture.yaml'}}})
    proposal,flow=build_models(config,'cpu')
    checkpoint=root/'checkpoints/mp20_u/model.best.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'config':config,'proposal':proposal.state_dict(),'flow':flow.state_dict(),'epoch':0,
                'runtime_options':{'coordinate_length_scale':3.}},checkpoint)
    command=[sys.executable,'-m','unifiedgroupgen.tests.legacy.evaluation_before_tasks.suite','--models','mp20_u',
             '--checkpoint-root',str(checkpoint.parents[1]),'--output',str(root/'evaluation'),
             '--tasks','dng','fixed_descriptor','--conditional-tasks','--count','1','--csp-cases','1',
             '--csp-k','1','--steps','2','--threads','1','--device','cpu','--no-fingerprints']
    for flags in (['--stage','generate','--run','--resume'],['--verify-generated'],['--stage','metrics','--run','--resume']):
        result=subprocess.run(command+flags,cwd=root,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
    state=json.loads((root/'evaluation/pipeline.json').read_text(encoding='utf-8'))
    assert state['state']=='complete'
    assert len(list((root/'evaluation').rglob('samples.jsonl')))==2
    assert not list((root/'evaluation').rglob('*.cif'))
    write_json(root/'validation.json',{'passed':True,'package_import':unifiedgroupgen.__file__,
        'generation_jobs':2,'standalone_generation_and_metrics':True,'scope':'Random small fixture; not material-quality evidence'})
    print('Installed wheel generation -> hash verification -> metrics: PASS')


if __name__=='__main__':main()
