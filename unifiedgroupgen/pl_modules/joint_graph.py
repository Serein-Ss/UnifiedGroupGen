"""One complete-atom graph backbone for anchors and joint structured edit rates.

This canonical-frame backbone delegates exact group constraints to the decoder.
It does not claim independent continuous SO(3) x SO(3) frame equivariance.
"""
import math
import torch
from torch import nn
from unifiedgroupgen.conditional.properties import PropertyEncoder
from unifiedgroupgen.group_constraints.state import legal_actions


class AtomGraphBlock(nn.Module):
    def __init__(self, hidden, heads, edge_dim=33):
        super().__init__()
        self.heads, self.width = heads, hidden//heads
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.qkv, self.output = nn.Linear(hidden, 3*hidden), nn.Linear(hidden, hidden)
        self.edge = nn.Sequential(nn.Linear(edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, heads))
        self.feedforward = nn.Sequential(nn.Linear(hidden, 4*hidden), nn.SiLU(), nn.Linear(4*hidden, hidden))

    def forward(self, nodes, edges):
        count = len(nodes)
        q, k, v = self.qkv(self.norm1(nodes)).reshape(count, 3, self.heads, self.width).unbind(1)
        q, k, v = (x.transpose(0, 1) for x in (q, k, v))
        bias = self.edge(edges).permute(2, 0, 1)
        weights = torch.softmax(q @ k.transpose(-1, -2)/math.sqrt(self.width)+bias, dim=-1)
        update = (weights @ v).transpose(0, 1).reshape_as(nodes)
        nodes = nodes+self.output(update)
        return nodes+self.feedforward(self.norm2(nodes))


class JointAtomGraph(nn.Module):
    action_kinds = ('birth', 'split', 'delete', 'element', 'replace', 'swap', 'end', 'fail')

    def __init__(self, allowed_elements, hidden=128, layers=4, heads=4, max_atoms=192,
                 max_orbits=52, properties=None, cond_dropout=.1, constraint_layers=False, task_controls=False,
                 feature_schema="nonmagnetic_v1"):
        super().__init__()
        if hidden % heads or layers < 1:
            raise ValueError('Hidden size must divide heads and at least one layer is required')
        if feature_schema not in ('nonmagnetic_v1', 'legacy_v1'):
            raise ValueError('Unknown joint feature schema')
        self.feature_schema = feature_schema
        self.legacy_features = feature_schema == 'legacy_v1'
        self.edge_dim = 33 if self.legacy_features else 24
        self.allowed_elements = tuple(sorted(set(int(z) for z in allowed_elements)))
        if not self.allowed_elements or any(not 1 <= z <= 118 for z in self.allowed_elements):
            raise ValueError('Invalid element vocabulary')
        self.max_atoms, self.max_orbits = max_atoms, max_orbits
        self.element_embedding = nn.Embedding(119, hidden)
        self.coordinate_encoder = nn.Sequential(nn.Linear(15, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.moment_encoder = (nn.Sequential(nn.Linear(7, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                               if self.legacy_features else None)
        self.global_encoder = nn.Sequential(nn.Linear(18 if self.legacy_features else 17, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.operation_encoder = nn.Sequential(nn.Linear(22 if self.legacy_features else 12, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.property_encoder = PropertyEncoder(properties or {}, hidden, cond_dropout)
        self.blocks = nn.ModuleList([AtomGraphBlock(hidden, heads, self.edge_dim) for _ in range(layers)])
        self.constraint_layers = nn.ModuleList()
        if constraint_layers:
            from unifiedgroupgen.pl_modules.constraint_layers import OperationConstraintLayer
            self.constraint_layers.extend(OperationConstraintLayer(hidden) for _ in range(layers))
        self.chemical_encoder = (nn.Sequential(nn.Linear(357, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                                 if task_controls else None)
        self.final_norm = nn.LayerNorm(hidden)
        self.metric_head = nn.Linear(hidden, 6)
        self.coordinate_head = nn.Linear(hidden, 3)
        self.moment_head = nn.Linear(hidden, 3) if self.legacy_features else None
        self.count_head = nn.Linear(hidden, 1)
        self.action_embedding = nn.Embedding(len(self.action_kinds), hidden)
        self.template_encoder = nn.Sequential(nn.Linear(12, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.template_site_encoder = nn.Sequential(nn.Linear(18, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.edit_encoder = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.action_head = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def _features(self, state, time, properties):
        if not self.legacy_features and (state.moments is not None or state.spec.magnetic
                or state.spec.spin_generators
                or any(op.spin is not None or op.time_reversal for op in state.spec.operations)):
            raise ValueError('The nonmagnetic feature schema does not accept moment or spin states')
        decoded = state.decoded(wrap=False)
        x, lattice = decoded['coordinates'], decoded['lattice']
        periodic = x.new_tensor(state.spec.periodic)
        phase = 2*math.pi*x*periodic
        residual = state.coordinates-x
        coordinate = torch.cat((phase.sin(), phase.cos(), x*(1-periodic), residual,
                                periodic.expand_as(x)), dim=-1)
        atom = self.element_embedding(decoded['atomic_numbers'])+self.coordinate_encoder(coordinate)
        if self.legacy_features:
            moment = x.new_zeros((state.num_atoms, 7))
            if state.moments is not None:
                moment = torch.cat((decoded['moments'], state.moments-decoded['moments'],
                                    x.new_ones((state.num_atoms, 1))), dim=-1)
            atom = atom+self.moment_encoder(moment)
        metric = torch.nn.functional.pad(state.metric, (0, 6-len(state.metric)))
        t = torch.as_tensor(time, device=x.device, dtype=x.dtype)
        clock = torch.stack((t, (math.pi*t).sin(), (math.pi*t).cos(), (2*math.pi*t).sin(), (2*math.pi*t).cos()))
        metadata = [math.log1p(state.num_atoms), (state.budget-state.step)/state.budget]
        if self.legacy_features:
            metadata.append(float(state.moments is not None))
        meta = x.new_tensor([*metadata, state.spec.dimension/3])
        global_features = torch.cat((metric, periodic, clock, meta))
        operations = []
        for op in state.spec.operations:
            spatial = torch.cat((x.new_tensor(op.rotation).reshape(-1), x.new_tensor(op.translation)))
            if self.legacy_features:
                if state.spec.magnetic or state.moments is not None or op.spin is not None:
                    from unifiedgroupgen.group_constraints.fields import spin_representation
                    spin = spin_representation(op, lattice).reshape(-1)
                else:
                    spin = x.new_zeros(9)
                spatial = torch.cat((spatial, spin, x.new_tensor([float(op.time_reversal)])))
            operations.append(spatial)
        if self.legacy_features:
            for generator in state.spec.spin_generators:
                operations.append(torch.cat((x.new_zeros(12), x.new_tensor(generator).reshape(-1), x.new_tensor([2.]))))
        group = self.operation_encoder(torch.stack(operations)).mean(0)
        global_token = self.global_encoder(global_features)+group+self.property_encoder(properties or {}, x)
        if self.chemical_encoder is not None:
            global_token = global_token+self.chemical_encoder(state.chemistry.features(state, self.allowed_elements))
        elif state.chemistry.mode != 'free':
            raise ValueError('This checkpoint was not trained with chemical task controls')
        nodes = torch.cat((global_token[None], atom))
        edges = x.new_zeros((state.num_atoms+1, state.num_atoms+1, self.edge_dim))
        if state.num_atoms:
            delta = x[None]-x[:, None]
            angle = 2*math.pi*delta*periodic
            lifted = delta-periodic*delta.round()
            distances = (lifted @ lattice.T).norm(dim=-1, keepdim=True)
            same = (decoded['orbit_index'][:, None] == decoded['orbit_index'][None]).to(x.dtype)[..., None]
            edges[1:, 1:, :11] = torch.cat((angle.sin(), angle.cos(), delta*(1-periodic), distances, same), dim=-1)
            # Explicit operation-labelled relations on the complete atom graph.
            # Spatial and internal matrices occupy distinct representation slots.
            n = state.num_atoms
            permutation = torch.tensor(state.compiled.permutations(), device=x.device, dtype=torch.long)
            source = torch.arange(n, device=x.device).expand_as(permutation)
            indices = (source*n+permutation).reshape(-1)
            relation_dim = 21 if self.legacy_features else 12
            finite = torch.stack(operations[:len(state.spec.operations)])[:, :relation_dim]
            transported = finite[:, None].expand(-1,n,-1)
            values = torch.cat((transported,x.new_ones((len(finite),n,1))),dim=-1).reshape(-1,relation_dim+1)
            relation = x.new_zeros((n*n,relation_dim+1)).index_add(0,indices,values)
            counts = relation[:,relation_dim:]
            relation = torch.cat((relation[:,:relation_dim]/counts.clamp_min(1),counts/len(finite)),dim=-1)
            edges[1:,1:,11:] = relation.reshape(n,n,relation_dim+1)
        return nodes, edges

    def _action_logits(self, state, actions, nodes):
        if not actions:
            return nodes.new_zeros(0)
        global_token = nodes[0]
        pools = [nodes[1:][section].mean(0) for section in state.compiled.slices]
        template_features, template_sites = {}, {}
        for w in state.spec.templates:
            a, b = w.arrays()
            values = state.metric.new_tensor(b)
            periodic = values.new_tensor(state.spec.periodic)
            template_features[w.letter] = torch.cat((values.new_tensor([w.multiplicity/self.max_atoms, w.dof/3]),
                (2*math.pi*values).sin().mean(0), (2*math.pi*values).cos().mean(0),
                (values*(1-periodic)).mean(0), values.new_tensor([float(w.dof == 0)])))
            basis = torch.nn.functional.pad(state.metric.new_tensor(a),(0,3-w.dof)).reshape(w.multiplicity,9)
            site = torch.cat((basis,(2*math.pi*values).sin(),(2*math.pi*values).cos(),values*(1-periodic)),dim=-1)
            template_sites[w.letter] = self.template_site_encoder(site).mean(0)
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
        indices = torch.tensor(kinds, device=nodes.device)
        species = torch.tensor(elements, device=nodes.device)
        hidden = (global_token+torch.stack(contexts)+self.action_embedding(indices)+self.element_embedding(species)
                  +self.template_encoder(torch.stack(templates))+torch.stack(sites)+self.edit_encoder(nodes.new_tensor(edits)))
        return self.action_head(hidden).squeeze(-1)

    def forward(self, state, time, properties=None, actions=None):
        if state.metric.device != next(self.parameters()).device:
            raise ValueError('Move the complete joint state to the model device')
        nodes, edges = self._features(state, time, properties)
        fields = None
        for index, block in enumerate(self.blocks):
            nodes = block(nodes, edges)
            if self.constraint_layers:
                nodes, fields = self.constraint_layers[index](nodes, state)
        nodes = self.final_norm(nodes)
        actions = legal_actions(state, self.allowed_elements, self.max_atoms, self.max_orbits) if actions is None else tuple(actions)
        logits = self._action_logits(state, actions, nodes)
        # Every active teacher state has at least END remaining. This lower
        # bound is compatible with the marginal target and prevents a vanishing
        # clock from stalling forever; the finite budget prevents explosion.
        mu = (1+(state.budget-state.step-1)*torch.sigmoid(self.count_head(nodes[0]).squeeze(-1))
              if state.status == 'active' and state.step < state.budget else nodes[0].sum()*0)
        result = dict(metric_anchor=self.metric_head(nodes[0])[:len(state.metric)],
                    coordinate_anchor=self.coordinate_head(nodes[1:]),
                    moment_anchor=None if self.moment_head is None or state.moments is None else self.moment_head(nodes[1:]),
                    remaining=mu, actions=actions, action_logits=logits)
        if fields is not None:
            result['constraint_fields'] = fields
        return result
