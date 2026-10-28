"""Teacher-forced token diagnostics, following legacy_eval/components on the model device."""
import torch
from torch.nn import functional as F


def descriptor_batch(model, descriptors, properties):
    sequences=[model.vocab.encode(d) for d in descriptors]
    device=model.embedding.weight.device
    width=max(len(s)-1 for s in sequences)
    ids=torch.full((len(sequences),width),model.vocab.index['PAD'],dtype=torch.long,device=device)
    targets=torch.full_like(ids,-100)
    legal=torch.ones(len(sequences),width,len(model.vocab.tokens),dtype=torch.bool,device=device)
    for i,(sequence,descriptor) in enumerate(zip(sequences,descriptors)):
        n=len(sequence)-1
        ids[i,:n]=torch.tensor(sequence[:-1],device=device)
        targets[i,:n]=torch.tensor(sequence[1:],device=device)
        legal[i,:n]=torch.stack([model.legality.allowed(sequence[:j],None,descriptor.kind) for j in range(1,len(sequence))]).to(device)
    h=model.embedding(ids)+model.position(torch.arange(width,device=device))[None]
    h=h+model.condition.forward_batch(properties,h)[:,None]
    causal=torch.ones(width,width,dtype=torch.bool,device=device).triu(1)
    logits=model.head(model.transformer(h,mask=causal,src_key_padding_mask=ids.eq(model.vocab.index['PAD'])))
    logits=logits.masked_fill(~legal,-torch.inf)
    losses=F.cross_entropy(logits.transpose(1,2),targets,reduction='none')
    predictions=logits.argmax(-1)
    result=[]
    for i,sequence in enumerate(sequences):
        n=len(sequence)-1
        parts={p:{'nll_sum':0.,'tokens':0,'correct':0} for p in ('G','W','Z','STOP')}
        for j,token in enumerate(sequence[1:]):
            part=parts[model.vocab.tokens[token].split(':')[0]]
            part['nll_sum']+=float(losses[i,j]);part['tokens']+=1;part['correct']+=int(predictions[i,j]==token)
        result.append({'descriptor_loss':float(losses[i,:n].mean()),
            'teacher_forced_sequence_correct':bool(predictions[i,:n].eq(targets[i,:n]).all()),'parts':parts})
    return result
