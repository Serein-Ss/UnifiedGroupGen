"""Shared joint-flow heads and selectable crystal graph backbones."""
from functools import lru_cache
import math
import torch
from torch import nn
from unifiedgroupgen.conditional.properties import PropertyEncoder
from unifiedgroupgen.group_constraints.flow_state import actions as legal_actions, metric_tangent, coordinate_tangent
from unifiedgroupgen.pl_modules.debug import debugging, observe, stage

@lru_cache(maxsize=512)
def _operation_features(spec):
    return tuple((*op.rotation[0], *op.rotation[1], *op.rotation[2], *op.translation)
                 for op in spec.operations)


class JointFlow(nn.Module):
    flow_version='joint_flow_v2'
    action_kinds=('birth','split','delete','element','replace','swap','freeze','fail')
    def _template_features(self, template, state):
        basis, offset = template.arrays()
        values = state.metric.new_tensor(offset)
        periodic = values.new_tensor(state.spec.periodic)
        sin, cos = (2*math.pi*values).sin(), (2*math.pi*values).cos()
        height = values*(1-periodic)
        summary = torch.cat((values.new_tensor([template.multiplicity/self.max_atoms, template.dof/3]),
                             sin.mean(0), cos.mean(0), height.mean(0),
                             values.new_tensor([float(template.dof == 0)])))
        basis = torch.nn.functional.pad(state.metric.new_tensor(basis), (0,3-template.dof))
        sites = torch.cat((basis.reshape(template.multiplicity,9),sin,cos,height),dim=-1)
        return summary, self.template_site_encoder(sites).mean(0)

    def _discrete_features(self, state, context):
        """Build decision features from templates and elements without live geometry."""
        if not state.letters:
            return context, context.new_zeros((0, context.numel()))
        templates = state.compiled.templates
        sites = torch.stack([self._template_features(w, state)[1] for w in templates])
        species = torch.tensor(state.elements, device=context.device)
        orbit_features = self.element_embedding(species) + sites
        multiplicity = context.new_tensor([w.multiplicity for w in templates])
        pooled = (orbit_features * multiplicity[:, None]).sum(0) / state.num_atoms
        return context + pooled, context + orbit_features

    def _action_logits(self, state, actions, global_token, pools):
        if not actions:
            return global_token.new_zeros(0)
        template_features, template_sites = {}, {}
        for template in state.spec.templates:
            template_features[template.letter], template_sites[template.letter] = self._template_features(template, state)
        contexts, templates, sites, edits, kinds, elements = [], [], [], [], [], []
        for action in actions:
            old = state.spec.template(state.letters[action.orbit]) if action.orbit >= 0 else None
            new = state.spec.template(action.letter) if action.letter else None
            context = pools[action.orbit] if action.orbit >= 0 else global_token
            if action.partner >= 0:
                context = (context+pools[action.partner])/2
            contexts.append(context)
            templates.append(template_features[action.letter] if new is not None else state.metric.new_zeros(12))
            sites.append(template_sites[action.letter] if new is not None else torch.zeros_like(global_token))
            old_m = old.multiplicity if old else 0
            new_m = new.multiplicity if new else 0
            delta = new_m-old_m if action.kind == 'replace' else (new_m if action.kind in ('birth', 'split') else -old_m if action.kind == 'delete' else 0)
            edits.append([delta/self.max_atoms, old_m/self.max_atoms, new_m/self.max_atoms,
                          old.dof/3 if old else 0, new.dof/3 if new else 0, float(action.kind in ('element', 'replace', 'swap'))])
            kinds.append(self.action_kinds.index(action.kind)); elements.append(action.element)
        indices = torch.tensor(kinds, device=global_token.device)
        species = torch.tensor(elements, device=global_token.device)
        hidden = (global_token+torch.stack(contexts)+self.action_embedding(indices)+self.element_embedding(species)
                  +self.template_encoder(torch.stack(templates))+torch.stack(sites)+self.edit_encoder(global_token.new_tensor(edits)))
        return self.action_head(hidden).squeeze(-1)

    def __init__(self,config):
        super().__init__()
        a=config['model'];h=a.get('hidden',128)
        self.allowed_elements=tuple(sorted(set(config['allowed_elements'])))
        if not self.allowed_elements or any(not 1<=z<=118 for z in self.allowed_elements):
            raise ValueError('A nonempty TRAIN element vocabulary is required')
        self.max_atoms=config.get('max_atoms',192);self.max_orbits=config.get('max_orbits',52)
        self.backend=a.get('backbone','equiformer_v2')
        self.orbit_ablation=config.get('orbit_ablation','joint')
        if self.orbit_ablation not in ('joint','fixed_orbits','no_geometry_feedback'):
            raise ValueError('Unknown orbit ablation')
        if 'orbit_ablation' in config and self.backend != 'equiformer_v2':
            raise ValueError('Orbit comparisons require the EquiformerV2 backbone')
        self.checkpoint_layers=a.get('checkpoint_layers',False)
        self.event_budget=config.get('event_budget',96);self.birth_scale=config.get('birth_scale',.2)
        self.element_embedding=nn.Embedding(119,h)
        self.global_encoder=nn.Sequential(nn.Linear(18,h),nn.SiLU(),nn.Linear(h,h))
        self.operation_encoder=nn.Sequential(nn.Linear(12,h),nn.SiLU(),nn.Linear(h,h))
        self.chemical_encoder=nn.Sequential(nn.Linear(357,h),nn.SiLU(),nn.Linear(h,h))
        self.property_encoder=PropertyEncoder(config.get('properties',{}),h,config.get('cond_dropout',0.))
        self.action_embedding=nn.Embedding(8,h)
        self.template_encoder=nn.Sequential(nn.Linear(12,h),nn.SiLU(),nn.Linear(h,h))
        self.template_site_encoder=nn.Sequential(nn.Linear(18,h),nn.SiLU(),nn.Linear(h,h))
        self.edit_encoder=nn.Sequential(nn.Linear(6,h),nn.SiLU(),nn.Linear(h,h))
        self.action_head=nn.Sequential(nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,1))
        self.metric_head=nn.Linear(h,6);self.count_head=nn.Linear(h,1)
        self.element_head=nn.Linear(h,119)
        if self.backend=='equiformer_v2':
            from .periodic import EquiformerFeatures
            self.backbone=EquiformerFeatures(hidden=h,layers=a.get('layers',4),
                channels=a.get('channels',128),lmax=a.get('lmax',4),mmax=a.get('mmax',2),
                cutoff=a.get('cutoff',6.),max_neighbors=a.get('max_neighbors',64),
                max_images=a.get('max_images',16),use_gate_act=a.get('use_gate_act',True))
            self.backbone.checkpoint_layers=self.checkpoint_layers
            self.scalar_readout=nn.Linear(a.get('channels',128),h)
        elif self.backend in ('cspnet', 'equiformer_v3', 'esen', 'mace'):
            if self.backend == 'cspnet':
                from unifiedgroupgen.pl_modules.backbones.cspnet import CSPNetFeatures as Backbone
            elif self.backend == 'equiformer_v3':
                from unifiedgroupgen.pl_modules.backbones.equiformer_v3 import EquiformerV3Features as Backbone
            elif self.backend == 'esen':
                from unifiedgroupgen.pl_modules.backbones.esen import ESENFeatures as Backbone
            else:
                from unifiedgroupgen.pl_modules.backbones.mace import MACEFeatures as Backbone
            self.backbone = Backbone({**a, 'allowed_elements': self.allowed_elements})
            self.scalar_readout = nn.Linear(self.backbone.channels, h)
        elif self.backend=='joint_atom_graph':
            from unifiedgroupgen.pl_modules.layers import AtomGraphBlock
            self.coordinate_encoder=nn.Sequential(nn.Linear(15,h),nn.SiLU(),nn.Linear(h,h))
            self.blocks=nn.ModuleList(AtomGraphBlock(h,a.get('heads',4),24) for _ in range(a.get('layers',4)))
            self.coordinate_head=nn.Linear(h,3)
        else:raise ValueError('Unknown joint_flow_v2 backbone')
        self.last_packed_graphs=0

    def _context(self,states,times,properties):
        features=[];operations=[];chem=[]
        for s,t in zip(states,times):
            k=torch.nn.functional.pad(s.metric,(0,6-len(s.metric)))
            phase=k.new_tensor([t,math.sin(math.pi*t),math.cos(math.pi*t),math.sin(2*math.pi*t),math.cos(2*math.pi*t)])
            features.append(torch.cat((k,k.new_tensor(s.spec.periodic),phase,
                k.new_tensor([math.log1p(s.num_atoms),(s.budget-s.step)/s.budget,s.spec.dimension/3,float(s.frozen)]))))
            op=k.new_tensor(_operation_features(s.spec))
            operations.append(self.operation_encoder(op).mean(0))
            chem.append(s.chemistry.features(s,self.allowed_elements))
        features, operations = torch.stack(features), torch.stack(operations)
        chemistry = self.chemical_encoder(torch.stack(chem))
        base = self.global_encoder(features) + operations + chemistry
        property_context = self.property_encoder.forward_batch(properties, base)
        decisions = None
        if self.orbit_ablation == 'no_geometry_feedback':
            neutral = torch.cat((torch.zeros_like(features[:, :6]), features[:, 6:]), dim=1)
            decisions = self.global_encoder(neutral) + operations + chemistry + property_context
        return base + property_context, decisions

    def _dense(self,states,times,properties,context):
        # Reuse the nonmagnetic feature definition; all attention is truly batched.
        nodes=[];edges=[]
        for s,t,p,c in zip(states,times,properties,context):
            x=s.decoded(wrap=False)['coordinates'];periodic=x.new_tensor(s.spec.periodic)
            phase=2*math.pi*x*periodic
            geo=torch.cat((phase.sin(),phase.cos(),x*(1-periodic),torch.zeros_like(x),periodic.expand_as(x)),-1)
            atom=self.element_embedding(s.decoded()['atomic_numbers'])+self.coordinate_encoder(geo)
            node=torch.cat((c[None],atom))
            e=x.new_zeros((s.num_atoms+1,s.num_atoms+1,24))
            if s.num_atoms:
                delta=x[None]-x[:,None];angle=2*math.pi*delta*periodic
                lifted=delta-periodic*delta.round()
                distance=(lifted@s.decoded()['lattice'].T).norm(dim=-1,keepdim=True)
                same=(s.decoded()['orbit_index'][:,None]==s.decoded()['orbit_index'][None]).to(x.dtype)[...,None]
                e[1:,1:,:11]=torch.cat((angle.sin(),angle.cos(),delta*(1-periodic),distance,same),-1)
            if s.num_atoms:
                n=s.num_atoms
                permutation=torch.tensor(s.compiled.permutations(),device=x.device,dtype=torch.long)
                source=torch.arange(n,device=x.device).expand_as(permutation)
                index=(source*n+permutation).reshape(-1)
                op=torch.stack([torch.cat((x.new_tensor(g.rotation).flatten(),x.new_tensor(g.translation))) for g in s.spec.operations])
                values=torch.cat((op[:,None].expand(-1,n,-1),x.new_ones((len(op),n,1))),-1).reshape(-1,13)
                rel=x.new_zeros((n*n,13)).index_add(0,index,values);counts=rel[:,12:]
                rel=torch.cat((rel[:,:12]/counts.clamp_min(1),counts/len(op)),-1)
                e[1:,1:,11:]=rel.reshape(n,n,13)
            nodes.append(node);edges.append(e)
        width=max(len(n) for n in nodes);batch=len(states);h=context.shape[-1]
        z=context.new_zeros((batch,width,h));e=context.new_zeros((batch,width,width,24))
        mask=torch.zeros((batch,width),device=context.device,dtype=torch.bool)
        for i,(n,edge) in enumerate(zip(nodes,edges)):
            z[i,:len(n)]=n;e[i,:len(n),:len(n)]=edge;mask[i,:len(n)]=True
        for block in self.blocks:
            q,k,v=block.qkv(block.norm1(z)).reshape(batch,width,3,block.heads,block.width).unbind(2)
            q,k,v=(a.transpose(1,2) for a in (q,k,v))
            bias=block.edge(e).permute(0,3,1,2)
            scores=(q@k.transpose(-1,-2)/math.sqrt(block.width)+bias).masked_fill(~mask[:,None,None,:],-torch.inf)
            update=(scores.softmax(-1)@v).transpose(1,2).reshape_as(z)
            z=z+block.output(update);z=z+block.feedforward(block.norm2(z))
        return [z[i,:s.num_atoms+1] for i,s in enumerate(states)], [self.coordinate_head(z[i,1:s.num_atoms+1]) for i,s in enumerate(states)]

    def forward_batch(self,states,times,properties=None,actions=None):
        if not states:return []
        if len(states)!=len(times):raise ValueError('state/time batch size mismatch')
        if any(s.moments is not None or s.spec.magnetic or s.spec.spin_generators or any(op.spin is not None or op.time_reversal for op in s.spec.operations) for s in states):
            raise ValueError('joint_flow_v2 accepts nonmagnetic states only')
        properties=properties or [{} for _ in states]
        with stage('model.context'):
            context, decisions = self._context(states,times,properties)
        if self.backend != 'joint_atom_graph':
            vector,scalar=self.backbone.forward_states(states,context)
            scalar=self.scalar_readout(scalar)
            nodes=[];vectors=[];offset=0
            for s,c in zip(states,context):
                atom=scalar[offset:offset+s.num_atoms]
                pooled=atom.mean(0) if s.num_atoms else torch.zeros_like(c)
                nodes.append(torch.cat(((c+pooled)[None],atom+c)))
                vectors.append(vector[offset:offset+s.num_atoms]);offset+=s.num_atoms
        else:
            nodes,vectors=self._dense(states,times,properties,context)
        self.last_packed_graphs=len(states)
        result=[]
        for i,(s,n,w) in enumerate(zip(states,nodes,vectors)):
            with stage('orbit.legal_actions'):
                support=legal_actions(s,self.allowed_elements,self.max_atoms,self.max_orbits) if actions is None or actions[i] is None else tuple(actions[i])
            if decisions is None:
                token = n[0]
                pools = (torch.stack([n[1:][section].mean(0) for section in s.compiled.slices])
                         if s.letters else n.new_zeros((0,n.shape[-1])))
            else:
                token, pools = self._discrete_features(s, decisions[i])
            with stage('orbit.action_logits'):
                logit=self._action_logits(s,support,token,pools)
            mu=(1+max(0,s.budget-s.step-1)*torch.sigmoid(self.count_head(token).squeeze())
                if support or (not s.frozen and s.status=='active') else token.sum()*0)
            if s.frozen or s.status!='active':mu=mu*0
            element=self.element_head(pools)
            allowed=set(s.chemistry.domain(self.allowed_elements))
            forbidden=torch.tensor([z not in allowed for z in range(119)],device=n.device)
            element=element.masked_fill(forbidden[None],-torch.inf)
            if self.orbit_ablation == 'fixed_orbits' and not s.frozen:
                metric_velocity = torch.zeros_like(s.metric)
                coordinate_velocity = torch.zeros_like(s.coordinates)
            else:
                origin=torch.cat([s.metric.new_tensor(b) for _,b,_ in s.compiled.blocks]) if s.letters else s.coordinates
                metric_velocity = metric_tangent(s,self.metric_head(n[0])[:len(s.metric)]-s.metric)
                coordinate_velocity = coordinate_tangent(
                    s,w,cartesian=self.backend not in ('joint_atom_graph', 'cspnet'),reference=s.coordinates-origin)
            if debugging():
                observe('model_outputs', sample=i, legal_actions=len(support), action_logits=logit,
                        element_logits=element, metric_velocity=metric_velocity, coordinate_velocity=coordinate_velocity)
            result.append(dict(metric_velocity_scaled=metric_velocity,coordinate_velocity_scaled=coordinate_velocity,
                element_logits=element,remaining=mu,actions=support,action_logits=logit))
        return result

    def forward(self,state,time,properties=None,actions=None):
        return self.forward_batch([state],[time],[properties or {}],None if actions is None else [actions])[0]
