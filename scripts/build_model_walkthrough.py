"""Replay a trained checkpoint and render an evidence-backed Chinese MP4.

No optimizer updates are performed. Historical curves come from saved logs;
hidden features and gradients are a deterministic probe of the saved checkpoint.
Install imageio-ffmpeg separately for video encoding; it is not a training dependency.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'assets/videos/model_walkthrough'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def collect(checkpoint, history):
    import numpy as np
    import torch
    from unifiedgroupgen.cli import build_models
    from unifiedgroupgen.symmetry import Descriptor, compile_descriptor
    from unifiedgroupgen.flow import conditional_path, velocity_loss, sample_flow
    from unifiedgroupgen.evaluation import validate, to_structure

    torch.set_num_threads(4)
    torch.manual_seed(314159)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    proposal, flow = build_models(config, 'cpu')
    proposal.load_state_dict(saved['proposal'], strict=True)
    flow.load_state_dict(saved['flow'], strict=True)
    proposal.eval(); flow.eval()
    flow.checkpoint_layers = False
    best_epoch = saved['epoch'] + 1
    del saved
    examples = json.loads((ROOT/'assets/paper/source/material_examples.json').read_text())['examples']
    trace = {'checkpoint_sha256': sha256(checkpoint), 'best_epoch': best_epoch,
             'proposal_parameters': sum(p.numel() for p in proposal.parameters()),
             'flow_parameters': sum(p.numel() for p in flow.parameters()),
             'scope': 'Checkpoint replay, no optimizer step; historical curves are separate saved observations.',
             'seeds': {'probe': 271828, 'fixed_sampling': 142, 'descriptor_sampling': 5142},
             'steps': 64, 'guidance': 1.0, 'examples': []}
    losses = []
    def array(tensor):
        return tensor.detach().cpu().double().numpy().tolist()

    def tokens_report(descriptor, properties):
        sequence = proposal.vocab.encode(descriptor)
        with torch.no_grad():
            logits = proposal(sequence[:-1], properties)
        reports = []
        for i, target in enumerate(sequence[1:]):
            legal = proposal.legality.allowed(sequence[:i+1], kind=descriptor.kind)
            probabilities = logits[i].masked_fill(~legal, -torch.inf).softmax(-1)
            vals, ids = probabilities.topk(min(3, int(legal.sum())))
            reports.append({'prefix': [proposal.vocab.tokens[j] for j in sequence[:i+1]],
                            'target': proposal.vocab.tokens[target], 'target_probability': float(probabilities[target]),
                            'legal_choices': int(legal.sum()),
                            'top': [[proposal.vocab.tokens[j], float(v)] for j, v in zip(ids.tolist(), vals)]})
        return reports

    for index, example in enumerate(examples):
        print('Collecting real model trace:', example['id'], flush=True)
        record = None
        for split in ('train', 'val', 'test'):
            path = ROOT/'artifacts/full_joint_sourcegrouped'/example['dataset']/f'{split}.jsonl'
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['id'] == example['id']:
                        record = row; break
            if record is not None:
                break
        if record is None:
            raise ValueError('Real example absent from prepared data')
        descriptor = Descriptor.from_dict(record['descriptor'])
        c = compile_descriptor(descriptor)
        endpoint = torch.tensor(record['state'], dtype=torch.float32)
        assert np.allclose(record['state'], example['state'], atol=1e-7)
        properties = dict(record['properties'])
        properties['source_domain'] = config['dataset_domains'][example['dataset']]
        start, target = conditional_path(c, endpoint, 0., generator=torch.Generator().manual_seed(271828+index))
        probe_time = .55
        state = start + probe_time*target
        features, candidates = [], []
        hooks = [layer.register_forward_hook(lambda m, a, o: features.append(array(o))) for layer in flow.layers]
        hooks.append(flow.coordinate_head.register_forward_hook(lambda m, a, o: candidates.append(array(o))))
        prediction = flow(c, state, probe_time, properties)
        for hook in hooks:
            hook.remove()
        loss = velocity_loss(c, state, target, prediction)
        discrete = proposal.loss(descriptor, properties)
        losses.append(discrete + loss['loss'])
        lattice = c.expand(state)['lattice']
        legal = c.coordinate_velocity(prediction[c.lattice_dof:])
        item = {'id': record['id'], 'dataset': example['dataset'], 'split': split,
                'descriptor': record['descriptor'], 'properties': properties, 'state': array(endpoint),
                'lattice_dof': c.lattice_dof, 'coordinate_dof': c.coordinate_dof, 'atoms': c.num_atoms,
                'elements': c.elements.tolist(), 'B': c.coordinate_basis.tolist(), 'b': c.b.tolist(),
                'H0': (c.root@c.root).tolist(), 'metric_basis': c.metric_basis.tolist(),
                'training': {'start': array(start), 'target': array(target), 't': probe_time,
                             'state': array(state), 'prediction': array(prediction), 'features': features,
                             'candidate_mixed': candidates[0], 'candidate_cartesian': array(torch.tensor(candidates[0], dtype=lattice.dtype)@lattice.T),
                             'legal_cartesian': array(legal@lattice.T),
                             'loss': {k: float(v.detach()) for k, v in loss.items()},
                             'descriptor_loss': float(discrete.detach())},
                'teacher_forcing': tokens_report(descriptor, properties)}
        with torch.no_grad():
            generated = sample_flow(flow, c, {'source_domain': properties['source_domain']}, steps=64,
                                    guidance=1., generator=torch.Generator().manual_seed(142+index), return_trajectory=True)
        item['fixed_generation'] = {'trajectory': array(generated['trajectory']),
                                    'validation': validate(c, generated),
                                    'conditioning': 'Real descriptor fixed; source context only; no property targets.'}
        to_structure(c, generated).to(filename=str(OUT/f'outputs/fixed_{descriptor.kind}.cif'))
        # Also execute the actual autoregressive frontend with an explicit composition.
        composition = {int(z): int(n) for z, n in zip(*np.unique(c.elements, return_counts=True))}
        frontend_logits = []
        hook = proposal.head.register_forward_hook(lambda m, a, o: frontend_logits.append(o[-1].detach().clone()))
        fresh = proposal.sample({'source_domain': properties['source_domain']}, kind=descriptor.kind,
                                composition=composition, guidance=1., generator=torch.Generator().manual_seed(5142+index))
        hook.remove()
        sequence = proposal.vocab.encode(fresh)
        assert len(frontend_logits) == len(sequence)-1
        choices = []
        for i, logits in enumerate(frontend_logits):
            legal = proposal.legality.allowed(sequence[:i+1], composition, fresh.kind)
            probs = logits.masked_fill(~legal, -torch.inf).softmax(-1)
            choices.append({'token': proposal.vocab.tokens[sequence[i+1]], 'probability': float(probs[sequence[i+1]]),
                            'legal_choices': int(legal.sum())})
        fresh_c = compile_descriptor(fresh)
        with torch.no_grad():
            fresh_generated = sample_flow(flow, fresh_c, {'source_domain': properties['source_domain']}, steps=64,
                                          guidance=1., composition=composition,
                                          generator=torch.Generator().manual_seed(6142+index), return_trajectory=True)
        item['full_generation'] = {'descriptor': fresh.to_dict(), 'tokens': choices, 'composition': composition,
                                   'trajectory': array(fresh_generated['trajectory']), 'validation': validate(fresh_c, fresh_generated)}
        to_structure(fresh_c, fresh_generated).to(filename=str(OUT/f'outputs/full_{descriptor.kind}.cif'))
        trace['examples'].append(item)
    total = torch.stack(losses).mean()
    total.backward()
    trace['gradient_probe'] = {'loss': float(total.detach()), 'optimizer_updates': 0,
                              'mode': 'eval, gradients enabled, no condition dropout; two exemplars, no composition/observation mask',
                              'norms': {name: math.sqrt(sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None))
                                        for name, module in [('Transformer', proposal), ('Flow', flow)]}}
    rows = [json.loads(s) for s in history.read_text().splitlines() if s.strip()]
    trace['history'] = list({r['epoch']: r for r in rows}.values())
    trace['history_sha256'] = sha256(history)
    trace['config'] = config
    trace['source_sha256'] = {name: sha256(ROOT/'src/unifiedgroupgen'/name) for name in ('flow.py', 'proposal.py', 'symmetry.py')}
    save_json(OUT/'data/model_trace.json', trace)
    print('Trace saved; all four actual trajectories and two backward probes complete.', flush=True)


# Pillow animation keeps numerical frames reproducible without saving thousands of PNGs.
def render(trace, fps, ffmpeg):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from unifiedgroupgen.symmetry import Descriptor
    from unifiedgroupgen.representation_audit import compiled_chart, expand_numpy

    W, H = 1600, 900
    BG, INK, MUTED = '#f6f8fa', '#183247', '#5b6d7b'
    BLUE, TEAL, RED, GOLD = '#437da6', '#167d8d', '#ce6d51', '#c79b2b'
    FONT = 'C:/Windows/Fonts/msyh.ttc'
    fonts = {size: ImageFont.truetype(FONT, size) for size in (18, 20, 22, 24, 26, 28, 32, 36, 44, 52)}
    # Twelve short chapters: 4m24s. No unobserved historical hidden states are invented.
    chapters = [
        (16, '从真实材料到合法结构生成', '同一个联合检查点：三维空间群 + 二维层群'),
        (24, '01  真实输入数据是什么', '训练样本 = 离散规则 a + 独立连续状态 z + 可用性质标签'),
        (24, '02  群—轨道编译器：减少自由度', '确定性展开 F = b + Bq；晶格 H = H0^(1/2) exp(S) H0^(1/2)'),
        (22, '03  描述 Transformer 如何学习规则', '用真实规则序列做 teacher forcing；只在合法候选中计算概率'),
        (26, '04  一次训练样本的流匹配路径', '随机先验 z0 → 中间状态 zt → 真实数据 z1；不是扩散的逐步反向去噪'),
        (24, '05  同一个网络内部发生了什么', '真实前向钩子：6 层消息传递，显示前 32 / 512 个特征通道'),
        (22, '06  候选速度如何变成合法速度', '独立自由度中的速度，展开后自动满足既定群作用'),
        (18, '07  损失函数与反向传播', '在已训练检查点上重放训练目标；计算梯度，但不更新权重'),
        (20, '08  历史训练过程：真实日志', '曲线来自服务器归档；内部特征来自当前检查点，二者时间含义不同'),
        (28, '09  固定规则的真实连续生成', '64 步 Heun 积分；从随机合法状态得到新结构，规则固定、几何变化'),
        (24, '10  从组成到规则再到几何', '额外演示：给定组成和材料类型，前端实际抽样规则，再由同一流网络生成'),
        (16, '11  最终输出与证据边界', '保存完整轨迹、CIF、模型预测和检验；几何合法不等于稳定或性质命中'),
    ]
    total_seconds = sum(c[0] for c in chapters)
    boundaries = np.cumsum([0]+[c[0] for c in chapters])
    examples = trace['examples']
    charts = [compiled_chart(Descriptor.from_dict(e['descriptor'])) for e in examples]
    fresh_charts = [compiled_chart(Descriptor.from_dict(e['full_generation']['descriptor'])) for e in examples]
    # Precompute expanded trajectory states once; frame interpolation remains in independent z.
    def txt(draw, xy, value, size=26, fill=INK):
        # YaHei lacks several mathematical super/subscripts; keep all labels readable.
        text = str(value)
        for original, replacement in [('q̇','qdot'), ('Ḟ','Fdot'), ('v̂','v_hat'),
                                      ('ᵀ','^T'), ('⁻¹','^(-1)'), ('₂','2'), ('₄','4')]:
            text = text.replace(original, replacement)
        draw.text(xy, text, font=fonts[size], fill=fill)

    def lines(draw, xy, values, size=26, gap=44, fill=INK):
        for i, value in enumerate(values):
            txt(draw, (xy[0], xy[1]+i*gap), value, size, fill)

    def box(draw, bounds, title, color=BLUE):
        draw.rounded_rectangle(bounds, radius=18, fill='white', outline='#d9e2e9', width=2)
        txt(draw, (bounds[0]+24, bounds[1]+18), title, 28, color)

    def number(v):
        return '['+', '.join(f'{x:.3f}' for x in v)+']'

    from unifiedgroupgen.structure_rendering import render_structure, element_color

    def structure(canvas, c, z, bounds, angle=.45, vectors=None):
        lattice, coords, _ = expand_numpy(c, z)
        x0, y0, x1, y1 = bounds
        image = render_structure(lattice.T, coords, c.elements, periodic=tuple(c.periodic),
            repeat=(2, 2, 1) if c.descriptor.kind == 'layer' else (1, 1, 1),
            size=(x1-x0, y1-y0), angle=angle, vectors=vectors)
        canvas.paste(image, (x0, y0))

    def atom_legend(draw, x, y, index):
        labels = [('Ti',22),('O',8)] if index==0 else [('Mo',42),('S',16)]
        for k,(label,number) in enumerate(labels):
            xx=x+k*100
            draw.ellipse((xx,y+4,xx+18,y+22),fill=element_color(number))
            txt(draw,(xx+27,y),label,22)

    def panel(draw, index, title=None):
        x = 48+index*768
        box(draw,(x,170,x+736,776),title or ('TiO₂ | MP20' if index==0 else 'MoS₂ | C2DB'),BLUE if index==0 else TEAL)
        atom_legend(draw,x+490,194,index)
        return x

    def curve(draw, rect, xs, ys, color, yrange=None):
        x0,y0,x1,y1=rect
        xs,ys=np.asarray(xs),np.asarray(ys)
        ymin,ymax=yrange or (float(ys.min()),float(ys.max()))
        pts=np.column_stack([x0+(xs-xs.min())/max(float(np.ptp(xs)),1e-9)*(x1-x0),
                             y1-(ys-ymin)/max(ymax-ymin,1e-9)*(y1-y0)])
        if len(pts)>1: draw.line([tuple(p) for p in pts],fill=color,width=4)
        return pts

    def frame(chapter, u, elapsed):
        im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
        txt(d,(48,30),chapters[chapter][1],44)
        txt(d,(48,98),chapters[chapter][2],26,MUTED)
        d.line((48,143,1552,143),fill='#d9e2e9',width=2)
        caption=''
        if chapter==0:
            lines(d,(60,210),['UnifiedGroupGen', '同一套规则编译接口，同一套连续网络',
                             f'最优检查点：第 {trace["best_epoch"]} 轮',
                             f'描述模型 {trace["proposal_parameters"]:,} + 流模型 {trace["flow_parameters"]:,} 参数'],32,65)
            structure(im,charts[0],examples[0]['state'],(70,480,760,765),angle=.45+u*.5)
            structure(im,charts[1],examples[1]['state'],(850,250,1520,720),angle=.45+u*.5)
            txt(d,(900,730),'TiO₂ / MoS₂：真实数据样本',24,MUTED)
            caption='先理解输入和训练目标，再观察真实网络中间值，最后执行一次新的材料生成。'
        elif chapter==1:
            for i,e in enumerate(examples):
                x=panel(d,i)
                rule='SG 136 / Ti:2a + O:4f' if i==0 else 'LG 78 / Mo:1a + S:2f'
                lines(d,(x+25,245),[f'id: {e["id"]}  |  split: {e["split"]}',f'a = {rule}',
                      f'z = {number(e["state"])}',f'展开后的原子元素: {e["elements"]}',
                      '周期边界: x,y,z' if i==0 else '周期边界: x,y；面外高度以 Å 计',
                      f'性质标签: Ef={e["properties"]["formation_energy"]:.3f} eV/atom; Eg={e["properties"]["band_gap"]:.3f} eV'],24,43)
                structure(im,charts[i],e['state'],(x+35,505,x+700,750))
            caption='每个样本的 z 长度可以不同：TiO₂ 为 3 维，MoS₂ 为 2 维；模型读取的是展开后的变长原子图。'
        elif chapter==2:
            for i,e in enumerate(examples):
                x=panel(d,i); c=charts[i]
                z=np.array(e['state']); z[-1]+=(.055 if i==0 else .45)*math.sin(2*math.pi*u)
                lines(d,(x+25,245),[f'{e["lattice_dof"]} 个晶格参数 + {e["coordinate_dof"]} 个位置参数',
                      'Ti 固定；四个 O 由同一个 u 联动' if i==0 else 'Mo 固定；两个 S 的高度为 +h / -h',
                      f'q = {z[-1]:.4f} '+('（分数坐标）' if i==0 else 'Å'),
                      f'k = {number(z[:c.lattice_dof])}; S = Σ k_i E_i',
                      '此处为规则展开演示，尚未调用神经网络'],24,43)
                structure(im,c,z,(x+35,490,x+700,740))
            caption='B 是由群操作和轨道表确定的基底。演示中只改变独立参数，完整原子轨道始终一起变化。'
        elif chapter==3:
            for i,e in enumerate(examples):
                x=panel(d,i); reports=e['teacher_forcing']; j=min(int(u*len(reports)),len(reports)-1); r=reports[j]
                lines(d,(x+25,245),['真实序列：BOS → G → W → Z → W → Z → STOP',
                    '当前输入: '+' '.join(r['prefix']),f'下一真实 token: {r["target"]}',
                    f'合法候选: {r["legal_choices"]}  |  目标概率: {r["target_probability"]:.3f}',
                    '当前预测中概率最高的候选：'],24,48)
                for k,(name,p) in enumerate(r['top']):
                    yy=510+k*68
                    txt(d,(x+25,yy),name,24)
                    d.rounded_rectangle((x+225,yy+5,x+225+max(2,420*p),yy+32),radius=4,fill=BLUE if i==0 else TEAL)
                    txt(d,(x+650,yy),f'{p:.2f}',22)
            caption='这是当前检查点对真实标签的预测，不代表模型自由抽样一定会选到该标签；非法 token 先被屏蔽。'
        elif chapter==4:
            for i,e in enumerate(examples):
                x=panel(d,i); t=e['training']; z=np.array(t['start'])+u*np.array(t['target'])
                lines(d,(x+25,245),[f't = {u:.2f}  |  zt = z0 + t × Δz',f'z0 = {number(t["start"])}',
                     f'Δz = {number(t["target"])}',f'zt = {number(z)}'],24,43)
                structure(im,charts[i],z,(x+35,440,x+700,740))
            caption='目标速度为连接先验和真实样本的 Δz；周期坐标走最短提升路径。每个中间态都通过编译器合法展开。'
        elif chapter==5:
            layer=min(int(u*6),5)
            for i,e in enumerate(examples):
                x=panel(d,i); matrix=np.array(e['training']['features'][layer])[:, :32]
                maxabs=max(abs(v) for ex in examples for a in ex['training']['features'] for row in a for v in row[:32]) or 1.
                normalized=np.sign(matrix)*np.log1p(np.abs(matrix)/.001)/np.log1p(maxabs/.001)
                cells=np.empty((*matrix.shape,3),dtype=np.uint8)
                cells[:]=np.where(normalized[...,None]>=0,
                    np.array([248,250,252])+(np.array([67,125,166])-np.array([248,250,252]))*normalized[...,None],
                    np.array([248,250,252])+(np.array([206,109,81])-np.array([248,250,252]))*(-normalized[...,None]))
                heat=Image.fromarray(cells).resize((640,170),Image.Resampling.NEAREST)
                im.paste(heat,(x+48,380)); d=ImageDraw.Draw(im)
                lines(d,(x+25,245),[f't = 0.55  |  消息传递第 {layer+1} / 6 层',
                      '节点编码：元素/轨道 + 时间/晶格/性质',
                      f'{e["atoms"]} 个节点 × 512 通道；{e["atoms"]*(e["atoms"]-1)} 条有向边'],24,43)
                txt(d,(x+48,565),'横轴：前 32 个特征通道；纵轴：原子',22)
                lines(d,(x+25,620),['蓝正橙负；共享对称对数色标，线性尺度 0.001',
                      f'原始值范围: {-maxabs:.2f} 到 {maxabs:.2f}',
                      '学习特征不是原子坐标，也不是注意力权重'],22,40)
            caption='所有颜色都来自真实激活值。本次样本的部分深层特征非常接近；是否存在过平滑，需要另外诊断。'
        elif chapter==6:
            for i,e in enumerate(examples):
                x=panel(d,i); t=e['training']; z=t['state']; use_legal=u>=.5
                vectors=t['legal_cartesian'] if use_legal else t['candidate_cartesian']
                lines(d,(x+25,245),['后半段：合法展开速度' if use_legal else '前半段：网络候选速度',
                    f'预测独立速度: {number(t["prediction"])}',
                    'q̇ = (BᵀMB)⁻¹ BᵀM v̂；M = LᵀL',
                    'Ḟ = Bq̇；红箭头表示笛卡尔速度'],24,43)
                structure(im,charts[i],z,(x+35,445,x+700,735),vectors=vectors)
            caption='投影使用晶格诱导的物理度量；箭头是当前预测值，采用固定显示比例，不代表模型自动保证化学合理性。'
        elif chapter==7:
            for i,e in enumerate(examples):
                x=panel(d,i); t=e['training']
                lines(d,(x+25,255),['L = L描述 + L晶格 + L位置',
                    f'L描述（合法 token 交叉熵）= {t["descriptor_loss"]:.4f}',
                    f'L晶格（独立速度误差）= {t["loss"]["lattice"]:.4f}',
                    f'L位置（笛卡尔速度误差）= {t["loss"]["coordinates"]:.4f}',
                    f'目标速度: {number(t["target"])}',f'预测速度: {number(t["prediction"])}'],24,58)
                txt(d,(x+25,660),'两样本平均目标值: '+f'{trace["gradient_probe"]["loss"]:.4f}',26,RED)
                txt(d,(x+25,710),'梯度范数: Transformer '+f'{trace["gradient_probe"]["norms"]["Transformer"]:.3f}'+
                    '  /  Flow '+f'{trace["gradient_probe"]["norms"]["Flow"]:.3f}',22)
            caption='已实际执行反向传播。演示未更新权重，梯度存在并不说明训练收敛；本次单样本损失也不是历史验证损失。'
        elif chapter==8:
            rows=trace['history']; epochs=np.array([r['epoch']+1 for r in rows])
            train=np.array([r['train']['descriptor']+r['train']['flow'] for r in rows])
            vr=[r for r in rows if r.get('val_loss') is not None]
            rect=(125,230,1470,650); yrange=(0,max(float(train.max()),max(r['val_loss'] for r in vr))*1.05)
            for y in range(0,int(yrange[1]),10):
                yy=650-y/yrange[1]*420; d.line((125,yy,1470,yy),fill='#dfe6eb',width=1); txt(d,(65,yy-15),y,22,MUTED)
            visible=max(2,int(u*len(rows)))
            d.rectangle(rect,fill='white')
            for y in range(0,int(yrange[1]),10):
                yy=650-y/yrange[1]*420; d.line((125,yy,1470,yy),fill='#e1e7ec',width=1)
            curve(d,rect,epochs,train,BLUE,yrange)
            vx=np.array([r['epoch']+1 for r in vr]); vy=np.array([r['val_loss'] for r in vr])
            vpoints=np.column_stack([125+(vx-epochs[0])/(epochs[-1]-epochs[0])*1345,
                                     650-vy/yrange[1]*420])
            d.line([tuple(p) for p in vpoints],fill=RED,width=4)
            cut=125+(visible/len(rows))*1345
            if cut<1470:d.rectangle((cut,230,1471,651),fill=BG)
            for epoch in (1,50,100,150,int(epochs[-1])):
                xx=125+(epoch-epochs[0])/(epochs[-1]-epochs[0])*1345;txt(d,(xx-20,665),epoch,22,MUTED)
            lines(d,(125,715),[f'历史记录 {len(rows)} 轮；当前最优检查点在第 {trace["best_epoch"]} 轮',
                  '蓝：训练总损失    橙：按计划验证的损失    横轴：epoch'],26,43)
            caption='这是真实服务器日志；当前已保存的最优权重只对应一个训练时刻，无法还原所有历史隐藏特征。'
        elif chapter in (9,10):
            for i,e in enumerate(examples):
                fresh=chapter==10; g=e['full_generation'] if fresh else e['fixed_generation']; c=fresh_charts[i] if fresh else charts[i]
                x=panel(d,i)
                progress=max(0,(u-.3)/.7) if fresh else u
                position=progress*64; j=min(int(position),63); alpha=position-j
                z=(1-alpha)*np.array(g['trajectory'][j])+alpha*np.array(g['trajectory'][j+1])
                if fresh:
                    chosen=g['tokens']; count=max(1,min(len(chosen),int(u/.3*len(chosen))))
                    token_text=' → '.join(r['token'] for r in chosen[:count])
                    lines(d,(x+25,245),['给定组成: Ti₂O₄' if i==0 else '给定组成: MoS₂',
                         f'实际抽样: {"SG" if i==0 else "LG"} {c.descriptor.number}',token_text,
                         f'连续积分: {min(64,int(position)):02d} / 64'],24,43)
                else:
                    lines(d,(x+25,245),[f'固定 {"SG 136" if i==0 else "LG 78"} 与元素/Wyckoff 列表',
                        f'Heun 步数: {min(64,int(position)):02d} / 64  |  t = {progress:.2f}',
                        f'z(t) = {number(z)}','仅 source_domain 上下文；无目标性质'],24,43)
                structure(im,c,z,(x+35,440,x+700,735))
            caption='实际保存 65 个积分状态；步间只为视频做独立参数插值。二维展示 2×2 超胞，法向不使用周期边界。'
        else:
            for i,e in enumerate(examples):
                x=panel(d,i); g=e['full_generation']; r=g['validation']; c=fresh_charts[i]
                lines(d,(x+25,245),[f'生成规则: {"SG" if i==0 else "LG"} {c.descriptor.number}',
                     f'群操作闭合: {r["group_operation_closure"]}',
                     f'识别到目标群: {r.get("exact_target_group")}',
                     f'最短原子间距: {r["minimum_distance_angstrom"]:.3f} Å',
                     f'晶格相容残差: {r["metric_residual"]:.2e}'],24,40)
                structure(im,c,g['trajectory'][-1],(x+35,455,x+700,740),angle=.45+u*.5)
            caption='这两例仅展示模型运作。新颖性、化学有效性、稳定性及性质命中率，需要样本级的独立标准评估。'
        d.rounded_rectangle((32,801,1568,864),radius=10,fill='#e7eef3')
        caption_lines=['']
        for char in caption:
            if d.textlength(caption_lines[-1]+char,font=fonts[22])>1490:
                caption_lines.append('')
            caption_lines[-1]+=char
        if len(caption_lines)>2:
            raise ValueError('Caption needs more than two lines')
        for row,text in enumerate(caption_lines):
            txt(d,(52,809+row*26 if len(caption_lines)>1 else 817),text,22)
        txt(d,(50,875),'真实检查点回放  ·  无权重更新  ·  无稳定性认证',18,MUTED)
        txt(d,(1350,875),f'{int(elapsed)//60:02d}:{int(elapsed)%60:02d} / {total_seconds//60:02d}:{total_seconds%60:02d}',18,MUTED)
        d.rectangle((0,895,int(W*elapsed/total_seconds),899),fill=TEAL)
        return im

    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'keyframes').mkdir(exist_ok=True)
    video=OUT/'UnifiedGroupGen_walkthrough_zh.mp4'
    command=[ffmpeg,'-y','-loglevel','warning','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
             '-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','22',
             '-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    log_path=ROOT/'artifacts/video_walkthrough/encoding.log'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for chapter,(duration,title,_) in enumerate(chapters):
                print('Rendering:',title,flush=True)
                midpoint=frame(chapter,.5,float(boundaries[chapter]+duration*.5))
                midpoint.save(OUT/f'keyframes/{chapter:02d}.png')
                for f in range(duration*fps):
                    elapsed=float(boundaries[chapter])+f/fps
                    process.stdin.write(frame(chapter,f/(duration*fps-1),elapsed).tobytes())
            process.stdin.close()
            if process.wait()!=0: raise RuntimeError('Video encoder failed; see encoding.log')
        except BaseException:
            process.kill();process.wait();raise
    save_json(OUT/'data/video_manifest.json',{'width':W,'height':H,'fps':fps,'duration_seconds':total_seconds,
             'frames':total_seconds*fps,'video_sha256':sha256(video),'trace_sha256':sha256(OUT/'data/model_trace.json'),
             'structure_style':'Reference matched: VESTA, matte 0.5 A spheres, translucent coordination hulls, split-color visual contacts and axis compass',
             'structure_renderer_sha256':hashlib.sha256((ROOT/'src/unifiedgroupgen/structure_rendering.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
             'chapters':[{'start_seconds':int(boundaries[i]),'duration_seconds':c[0],'title':c[1]} for i,c in enumerate(chapters)]})
    print('Video saved:',video,flush=True)


def verify(ffmpeg):
    """Verify recorded targets/projections, all trajectory states, and the encoded MP4."""
    import numpy as np
    import imageio_ffmpeg
    from unifiedgroupgen.symmetry import Descriptor
    from unifiedgroupgen.representation_audit import compiled_chart, expand_numpy, action_statistics
    trace = json.loads((OUT/'data/model_trace.json').read_text(encoding='utf-8'))
    manifest = json.loads((OUT/'data/video_manifest.json').read_text(encoding='utf-8'))
    assert manifest['trace_sha256'] == sha256(OUT/'data/model_trace.json')
    assert manifest['video_sha256'] == sha256(OUT/'UnifiedGroupGen_walkthrough_zh.mp4')
    assert manifest['structure_renderer_sha256'] == hashlib.sha256((ROOT/'src/unifiedgroupgen/structure_rendering.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    errors, statistics = [], []
    for e in trace['examples']:
        c = compiled_chart(Descriptor.from_dict(e['descriptor']))
        t = e['training']; start, target, endpoint = map(np.array, (t['start'],t['target'],e['state']))
        difference = start + target - endpoint
        difference[c.state_periodic] -= np.round(difference[c.state_periodic])
        assert np.max(np.abs(difference)) < 1e-6
        assert np.allclose(t['state'], start+t['t']*target, atol=1e-6)
        assert np.array(t['features']).shape == (6, c.num_atoms, 512)
        lattice, _, _ = expand_numpy(c,t['state'])
        basis = c.coordinate_basis
        qdot = np.array(t['prediction'])[c.lattice_dof:]
        candidate = np.array(t['candidate_cartesian'])
        legal = np.einsum('ijq,q->ij',basis,qdot)@lattice.T
        assert np.allclose(legal,t['legal_cartesian'],atol=2e-5)
        normal_error = np.einsum('iaq,ba,ib->q',basis,lattice,candidate-legal)
        errors.append(float(np.max(np.abs(normal_error),initial=0)))
        assert errors[-1] < 1e-4
        for name in ('fixed_generation','full_generation'):
            g=e[name]
            chart=compiled_chart(Descriptor.from_dict(g.get('descriptor',e['descriptor'])))
            states=np.array(g['trajectory'])
            assert states.shape == (65,chart.dof) and np.isfinite(states).all()
            checks=[]
            for z in states:
                l,f,_=expand_numpy(chart,z)
                checks.append(action_statistics(chart,f,l))
            maxima={k:max(r[k] for r in checks) for k in ('operation_residual_angstrom','tangent_action_residual','metric_relative_residual')}
            assert maxima['operation_residual_angstrom']<1e-7
            assert maxima['tangent_action_residual']<1e-7
            assert maxima['metric_relative_residual']<1e-10
            statistics.append({'example':e['id'],'mode':name,'states_checked':65,**maxima})
    video=OUT/'UnifiedGroupGen_walkthrough_zh.mp4'
    reader=imageio_ffmpeg.read_frames(str(video))
    metadata=next(reader)
    reader.close()
    assert tuple(metadata['size']) == (manifest['width'],manifest['height'])
    assert abs(metadata['fps']-manifest['fps'])<1e-6
    assert abs(metadata['duration']-manifest['duration_seconds'])<.1
    # Decode the full export, not just the source frames, and count decoded frames.
    result=subprocess.run([ffmpeg,'-v','error','-i',str(video),'-map','0:v:0','-f','null','-',
                           '-progress','pipe:1','-nostats'],capture_output=True,check=True)
    progress=result.stdout.decode('utf-8')
    decoded=[int(s.split('=')[1]) for s in progress.splitlines() if s.startswith('frame=')][-1]
    assert decoded == manifest['frames']
    # Decode an actual output frame for human visual QA.
    subprocess.run([ffmpeg,'-y','-loglevel','error','-ss','147','-i',str(video),'-frames:v','1',
                    str(OUT/'keyframes/decoded_projection.png')],check=True)
    save_json(OUT/'data/verification.json',{'passed':True,'decoded_video_frames':decoded,
              'encoded_video_metadata':metadata,
              'trajectory_states_checked':260,'trajectories':statistics,
              'maximum_projection_normal_equation_error':max(errors),
              'training_targets_and_feature_shapes_checked':True,
              'optimizer_updates':trace['gradient_probe']['optimizer_updates'],
              'video_sha256':manifest['video_sha256'],
              'scope':'Numerical playback and complete video decode; no chemical or physical certification.'})
    print('Verification passed:',decoded,'decoded frames; 260 legal trajectory states.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','trace','render','verify'],default='all')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/imported_server/checkpoints/joint_best.pt')
    parser.add_argument('--history',type=Path,default=ROOT.parent/'tmp/UnifiedGroupGen_remote_audit_20261017/evidence/artifacts/runs/joint_space_layer_large_seed42/model.history.jsonl')
    parser.add_argument('--fps',type=int,default=12)
    args=parser.parse_args()
    if args.fps<1:parser.error('--fps must be positive')
    (OUT/'outputs').mkdir(parents=True,exist_ok=True)
    if args.stage in ('all','trace'): collect(args.checkpoint,args.history)
    if args.stage in ('all','render'):
        import imageio_ffmpeg
        render(json.loads((OUT/'data/model_trace.json').read_text(encoding='utf-8')),args.fps,imageio_ffmpeg.get_ffmpeg_exe())
    if args.stage in ('all','verify'):
        import imageio_ffmpeg
        verify(imageio_ffmpeg.get_ffmpeg_exe())


if __name__=='__main__':
    main()
