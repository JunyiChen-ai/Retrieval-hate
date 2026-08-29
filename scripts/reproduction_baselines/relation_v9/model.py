"""Minimal dependence-aware prior/locator factorization."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def dependence_weights(sequences, threshold=.98):
    """Cluster near-duplicate train experts and give every cluster equal mass."""
    x=np.asarray(sequences,float)
    if x.ndim != 2: raise ValueError("expected pooled train frames [N,E]")
    e=x.shape[1]; corr=np.nan_to_num(np.corrcoef(x,rowvar=False),nan=0.)
    parent=list(range(e))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    for i in range(e):
        for j in range(i):
            if corr[i,j]>=threshold: union(i,j)
    clusters={}
    for i in range(e): clusters.setdefault(find(i),[]).append(i)
    weight=np.zeros(e,float)
    for members in clusters.values(): weight[members]=1./len(clusters)/len(members)
    return weight.astype(np.float32),[members for _,members in sorted(clusters.items())],corr


def masked_mean(x,mask):
    w=mask.to(x.dtype)
    while w.ndim<x.ndim:w=w.unsqueeze(-1)
    return (x*w).sum(1,keepdim=True)/w.sum(1,keepdim=True).clamp_min(1)


def shift_sequence(x,lag):
    """Shift without circular wrap; positive lag delays the sequence."""
    y=torch.zeros_like(x)
    if lag>=0:
        if lag<len(x): y[lag:]=x[:len(x)-lag]
    elif -lag<len(x): y[:lag]=x[-lag:]
    return y


def estimate_lag(reference, source, window):
    """Deterministic change-correlation lag, used only for relation geometry."""
    dr=F.pad(reference[1:]-reference[:-1],(1,0)); ds=F.pad(source[1:]-source[:-1],(1,0))
    candidates=[]
    for lag in range(-window,window+1):
        aligned=shift_sequence(ds,lag); denom=(dr.norm()*aligned.norm()).clamp_min(1e-8)
        candidates.append((float((dr*aligned).sum()/denom),lag))
    return max(candidates,key=lambda z:(z[0],-abs(z[1])))[1]


class DependenceAwareRelation(nn.Module):
    def __init__(self,n_experts,clusters,hidden=32,window=8):
        super().__init__();self.n_experts=n_experts;self.clusters=[list(x) for x in clusters];self.n_clusters=len(clusters);self.window=window
        flat=sorted(x for group in self.clusters for x in group)
        if flat!=list(range(n_experts)) or not self.clusters:raise ValueError("clusters must partition all experts")
        self.register_buffer("weights",torch.full((self.n_clusters,),1./self.n_clusters))
        self.token=nn.Sequential(nn.Linear(6,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.attn=nn.MultiheadAttention(hidden,1,batch_first=True)
        self.prior_head=nn.Linear(hidden,1)
        self.locator=nn.Sequential(nn.Conv1d(2*self.n_clusters,hidden,3,padding=1),nn.GELU(),nn.Conv1d(hidden,1,3,padding=1))
        self.prior_scale=nn.Parameter(torch.zeros(()));self.locator_scale=nn.Parameter(torch.zeros(()))

    def distribution_tokens(self,x,valid):
        rows=[]
        for b in range(len(x)):
            z=x[b,valid[b]]
            rows.append(torch.stack([z.mean(0),z.std(0,unbiased=False),z.quantile(.1,0),z.quantile(.5,0),z.quantile(.9,0),z.topk(max(1,len(z)//8),dim=0).values.mean(0)],-1))
        return torch.stack(rows)

    def aggregate_clusters(self,x):
        # Exact duplicates are collapsed before any learned operation.
        return torch.stack([x[...,members].mean(-1) for members in self.clusters],-1)

    def aligned_centered(self,centered,valid,weights):
        result=torch.zeros_like(centered);lags=torch.zeros(len(centered),self.n_clusters,dtype=torch.long,device=centered.device)
        for b in range(len(centered)):
            n=int(valid[b].sum()); z=centered[b,:n]; reference=(z*weights).sum(-1)
            for e in range(self.n_clusters):
                lag=estimate_lag(reference,z[:,e],self.window);lags[b,e]=lag;result[b,:n,e]=shift_sequence(z[:,e],lag)
        return result,lags

    def forward(self,x,valid,expert_keep=None,cluster_noise=None):
        x=self.aggregate_clusters(x)
        if cluster_noise is not None:
            if cluster_noise.shape!=x.shape:raise ValueError("noise must be sampled at cluster level [B,T,C]")
            x=(x+cluster_noise).clamp(0,1)
        if expert_keep is None: expert_keep=torch.ones_like(self.weights)
        if expert_keep.shape!=(self.n_clusters,):raise ValueError("dropout must be cluster-level")
        # An accidentally all-dropped draw is a no-corruption fallback, not a
        # zero-evidence pseudo-example.
        if not bool((expert_keep>0).any()):expert_keep=torch.ones_like(expert_keep)
        w=self.weights*expert_keep;w=w/w.sum().clamp_min(1e-8)
        consensus=(x*w).sum(-1)*valid;base_prior=masked_mean(consensus,valid).squeeze(1);base_locator=(consensus-base_prior[:,None])*valid
        tokens=self.token(self.distribution_tokens(x,valid));padding=(expert_keep<=0)[None].expand(len(x),-1);related,_=self.attn(tokens,tokens,tokens,key_padding_mask=padding)
        prior_delta=(self.prior_head(related).squeeze(-1)*w).sum(-1)
        active=expert_keep[None,None,:];x_active=x*active
        centered=(x_active-masked_mean(x_active,valid))*valid[...,None]*active;aligned,lags=self.aligned_centered(centered,valid,w)
        locator_delta=self.locator(torch.cat([centered,aligned],-1).transpose(1,2)).squeeze(1)*valid
        locator_delta=(locator_delta-masked_mean(locator_delta,valid))*valid
        base_score=(base_prior[:,None]+base_locator).clamp(1e-5,1-1e-5);base_logit=torch.logit(base_score)
        prior_gain=torch.tanh(self.prior_scale);locator_gain=torch.tanh(self.locator_scale)
        prior_logit=masked_mean(base_logit,valid).squeeze(1)+prior_gain*prior_delta
        base_logit_locator=(base_logit-masked_mean(base_logit,valid))*valid
        locator_logit=base_logit_locator+locator_gain*locator_delta
        locator_logit=(locator_logit-masked_mean(locator_logit,valid))*valid
        # This expression gives an exact, not merely approximate, identity at
        # zero initialization while preserving gradients into both gains.
        frame_logit=(base_logit+prior_gain*prior_delta[:,None]+locator_gain*locator_delta)*valid
        return {"frame_logit":frame_logit,"prior_logit":prior_logit,"locator_logit":locator_logit,"base_logit":base_logit,"base_prior":base_prior,"base_locator":base_locator,"prior_delta":prior_delta,"locator_delta":locator_delta,"lags":lags}
