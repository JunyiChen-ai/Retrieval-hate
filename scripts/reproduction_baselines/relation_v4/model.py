"""Expert-relation gate with an exact static-rank-fusion fallback."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_rank(scores, valid):
    """Per-video percentile ranks; ties receive their deterministic mid-rank."""
    out=torch.zeros_like(scores)
    for b in range(scores.shape[0]):
        n=int(valid[b].sum())
        if n == 0: continue
        x=scores[b,:n]
        # Pairwise definition gives identical scores identical ranks.
        less=(x[:,None,:] < x[None,:,:]).sum(1).to(x.dtype)
        equal=(x[:,None,:] == x[None,:,:]).sum(1).to(x.dtype)
        out[b,:n]=(less+.5*(equal-1))/max(1,n-1)
    return out


class ExpertRelationGate(nn.Module):
    """Use cross-expert temporal relations to correct static rank fusion.

    The dynamic branch never sees raw video/audio/text features.  Its complete
    input is expert ranks, consensus/disagreement, and locally transported
    expert ranks.  A zero-initialized scalar makes construction exactly equal
    to the externally supplied validation-selected static fusion.
    """
    def __init__(self,n_experts,static_weights,hidden=64,window=12,temperature=.2):
        super().__init__(); self.n_experts=int(n_experts); self.window=int(window); self.temperature=float(temperature)
        weight=torch.as_tensor(static_weights,dtype=torch.float32)
        if weight.shape!=(self.n_experts,) or bool((weight<0).any()) or float(weight.sum())<=0: raise ValueError("invalid static weights")
        self.register_buffer("static_weights",weight/weight.sum())
        # Evidence is open: no named semantic class, only prediction geometry.
        relation_width=3*self.n_experts*self.n_experts
        self.temporal=nn.Sequential(nn.Conv1d(self.n_experts+relation_width+2,hidden,3,padding=1),nn.GELU(),nn.Conv1d(hidden,hidden,3,padding=1),nn.GELU())
        self.gate=nn.Linear(hidden,self.n_experts)
        self.correction_scale=nn.Parameter(torch.zeros(()))

    def transport(self,ranks,valid):
        """Align each target expert to every source expert within a local window."""
        b,t,e=ranks.shape; index=torch.arange(t,device=ranks.device); legal=(index[:,None]-index[None,:]).abs()<=self.window; legal=legal[None,None]&valid[:,None,:,None]&valid[:,None,None,:]
        # Similar local changes, not absolute confidence calibration, drive alignment.
        change=F.pad(ranks[:,1:]-ranks[:,:-1],(0,0,1,0)).permute(0,2,1)
        # [batch, target expert, source expert, target time, source time].
        pair=-(change[:,:,None,:,None]-change[:,None,:,None,:]).square()/self.temperature
        pair=pair.masked_fill(~legal[:,None],-torch.inf); att=torch.softmax(pair,-1); att=torch.nan_to_num(att,nan=0.0)*legal[:,None]
        aligned=torch.einsum("bqets,bse->btqe",att,ranks)
        lag=(index[None,:]-index[:,None]).to(ranks.dtype)
        expected_lag=torch.einsum("bqets,ts->btqe",att,lag)
        return aligned,expected_lag,att

    def forward(self,scores,valid):
        ranks=masked_rank(scores,valid); aligned,lag,transport=self.transport(ranks,valid)
        consensus=(ranks*self.static_weights).sum(-1,keepdim=True)
        disagreement=torch.sqrt(((ranks-consensus).square()*self.static_weights).sum(-1,keepdim=True)+1e-8)
        target=ranks[:,:,:,None].expand_as(aligned)
        relation=torch.cat([aligned,target-aligned,lag/max(1,self.window)],-1).reshape(ranks.shape[0],ranks.shape[1],-1)
        evidence=torch.cat([ranks,relation,consensus,disagreement],-1)*valid[...,None]
        hidden=self.temporal(evidence.transpose(1,2)).transpose(1,2)*valid[...,None]
        dynamic_weight=torch.softmax(self.gate(hidden),-1)
        dynamic=(dynamic_weight*ranks).sum(-1)
        static=consensus.squeeze(-1)
        correction=torch.tanh(self.correction_scale)*(dynamic-static)
        final=(static+correction)*valid
        return {"frame_score":final,"static_score":static*valid,"rank_correction":correction*valid,"expert_gate":dynamic_weight,"expert_ranks":ranks,"consensus":consensus.squeeze(-1)*valid,"disagreement":disagreement.squeeze(-1)*valid,"aligned_ranks":aligned,"expected_lag":lag,"transport":transport}


class AnalyticExpertRelationGate(nn.Module):
    """Parameter-free relation gate; only ``beta`` is selected on validation.

    Experts receive more weight where their rank is supported by temporally
    aligned ranks from the other experts. ``beta=0`` is exactly static fusion,
    providing the required fail-safe without fitting any labels.
    """
    def __init__(self,n_experts,static_weights,beta=0.,gamma=0.,window=12,temperature=.2,
                 inputs_are_calibrated=False):
        super().__init__()
        self.core=ExpertRelationGate(n_experts,static_weights,hidden=8,
                                     window=window,temperature=temperature)
        self.beta=float(beta)
        self.gamma=float(gamma)
        self.inputs_are_calibrated=bool(inputs_are_calibrated)

    def forward(self,scores,valid):
        ranks=(scores*valid[...,None] if self.inputs_are_calibrated
               else masked_rank(scores,valid))
        aligned,lag,transport=self.core.transport(ranks,valid)
        weight=self.core.static_weights
        # aligned[b,t,q,e] is source expert e aligned to target expert q.
        agreement=-(ranks[:,:,:,None]-aligned).abs()
        support=(agreement*weight[None,None,None,:]).sum(-1)
        static=(ranks*weight).sum(-1)*valid
        if self.beta == 0.0:
            gate=weight[None,None,:].expand_as(ranks)
            final=static
        else:
            logits=torch.log(weight.clamp_min(1e-8))[None,None,:]+self.beta*support
            gate=torch.softmax(logits,-1)
            final=(gate*ranks).sum(-1)*valid
        # Propagate complementary evidence after local cross-expert alignment.
        transported=(aligned*weight[None,None,:,None]
                     *weight[None,None,None,:]).sum((-1,-2))*valid
        final=final+self.gamma*(transported-final)*valid
        return {"frame_score":final,"static_score":static,
                "rank_correction":final-static,"expert_gate":gate,
                "expert_ranks":ranks,"support":support*valid[...,None],
                "transported_score":transported,"aligned_ranks":aligned,"expected_lag":lag,
                "transport":transport}
