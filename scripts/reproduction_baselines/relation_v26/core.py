import hashlib,json,math,random,copy
import numpy as np,torch
from torch import nn
DESIGN_SHA='203960db2e45cd9f4e25eb61f9864ca6ba204fd184ffed1fd415b94cacccc859';MIGRATION_SHA='044358e1cf09c4a7d086eee4bbbe5bbb483c4f77a5f538b6d3717bd09f39b0f8';ARCH='v26_finite_rf_dilated_v1';OFF=(-16,-8,-4,-2,2,4,8,16);SEEDS=(234,2025,3407);DILATIONS=(1,2,4,8)
def ch(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def tensor_ch(x):return ch([z.detach().cpu().tolist() if torch.is_tensor(z) else z for z in x])
def availability_counts(ms):return [int(torch.as_tensor(x).sum()) for x in ms]
def intervention_count(ms):return int(torch.stack([torch.as_tensor(x,dtype=torch.bool) for x in ms],1).any(1).sum())
def fold(v):return int(hashlib.sha256(v.encode()).hexdigest()[:8],16)%5
class Decoder(nn.Module):
 def __init__(self,d):super().__init__();self.p=nn.Linear(d,128);self.net=nn.Sequential(nn.Linear(258,256),nn.GELU(),nn.Linear(256,d))
 def forward(self,x,m,t):
  # fixed gather excludes [t-1,t,t+1] structurally
  side=[]
  for oo in (OFF[:4],OFF[4:]):
   ix=[t+o for o in oo if 0<=t+o<len(x) and bool(m[t+o])];z=torch.stack([self.p(x[i]) for i in ix]).mean(0) if ix else torch.zeros(128,dtype=x.dtype,device=x.device);side += [z,torch.tensor([len(ix)/4],dtype=x.dtype,device=x.device)]
  return self.net(torch.cat(side))
class RFBlock(nn.Module):
 def __init__(self,dilation):
  super().__init__();self.dw=nn.Conv1d(256,256,3,padding=dilation,dilation=dilation,groups=256,bias=False);self.w1=nn.Linear(256,512,bias=True);self.w2=nn.Linear(512,256,bias=True);self.ln=nn.LayerNorm(256,eps=1e-5,elementwise_affine=True)
 def forward(self,h,q):
  u=self.dw(h.transpose(1,2)).transpose(1,2);v=h+u;return q[...,None]*self.ln(v+self.w2(torch.nn.functional.gelu(self.w1(v))))
class CTW(nn.Module):
 def __init__(self,dims=(512,128,768),model_seed=234):
  super().__init__();self.proj=nn.ModuleDict({n:nn.Linear(d,128,bias=True) for n,d in zip(('visual','audio','text'),dims)});self.input_projection=nn.Linear(387,256,bias=True);self.blocks=nn.ModuleList([RFBlock(d) for d in DILATIONS]);self.contribution_head=nn.Linear(256,1,bias=True);self._canonical_init(model_seed)
 def _canonical_init(self,seed):
  gen=torch.Generator(device='cpu');gen.manual_seed(seed)
  ordered=[self.proj[n] for n in ('visual','audio','text')]+[self.input_projection]
  for b in self.blocks:ordered.extend((b.dw,b.w1,b.w2,b.ln))
  with torch.no_grad():
   for m in ordered:
    if isinstance(m,(nn.Linear,nn.Conv1d)):
     nn.init.xavier_uniform_(m.weight,gain=1.,generator=gen)
     if m.bias is not None:nn.init.zeros_(m.bias)
    else:nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
   nn.init.zeros_(self.contribution_head.weight);nn.init.zeros_(self.contribution_head.bias)
 def _pos(self,p,dtype,device):
  p=torch.as_tensor(p,device=device,dtype=dtype)[...,None];k=torch.arange(0,256,2,device=device,dtype=dtype);w=torch.exp(-math.log(10000.)*k/256);z=torch.zeros(*p.shape[:-1],256,device=device,dtype=dtype);z[...,0::2]=torch.sin(p*w);z[...,1::2]=torch.cos(p*w);return z
 def contributions_batch(self,xs,ms,positions=None):
  # Inputs may be [T,D] or [B,T,D].
  squeeze=xs[0].ndim==2
  if squeeze:xs=[x[None] for x in xs];ms=[m[None] for m in ms]
  q=torch.stack(ms,2).any(2).to(xs[0].dtype);z=[]
  for n,x,m in zip(('visual','audio','text'),xs,ms):z.append(torch.nn.functional.gelu(self.proj[n](x))*m[...,None])
  if positions is None:positions=torch.arange(xs[0].shape[1],device=xs[0].device)[None].expand(xs[0].shape[0],-1)
  h=q[...,None]*(self.input_projection(torch.cat(z+[torch.stack(ms,2).to(xs[0].dtype)],2))+self._pos(positions,xs[0].dtype,xs[0].device))
  for b in self.blocks:h=b(h,q)
  a=q*self.contribution_head(h).squeeze(-1);return (a.squeeze(0),q.squeeze(0)) if squeeze else (a,q)
 def residual_batch(self,xs,ms):
  a,q=self.contributions_batch(xs,ms);squeeze=a.ndim==1
  if squeeze:a=a[None];q=q[None]
  te=q.sum(1)
  if torch.any(te<=0):raise RuntimeError('all-missing video')
  r=a.sum(1)/te;return r.squeeze(0) if squeeze else r
 def residual(self,xs,ms):return self.residual_batch(xs,ms)
 def forward(self,xs,ms,g):return g+self.residual(xs,ms)
 def effects_slow(self,xs,ms,bs,g):
  base=self(xs,ms,g);T=len(xs[0]);te=torch.stack(ms,1).any(1).sum().to(base.dtype);out=[]
  for t in range(T):
   cf=[x.clone() for x in xs]
   for f in range(3):
    if bool(ms[f][t]):cf[f][t]=bs[f][t].detach()
   out.append(te*(base-self(cf,ms,g)))
  return torch.stack(out)
 def effects(self,xs,ms,bs,g,chunk=64):
  # Exact finite-RF local cones, chunked in fixed ascending target order.
  base_a,q=self.contributions_batch(xs,ms);T=len(xs[0]);te=q.sum().to(base_a.dtype);base_r=base_a.sum()/te;out=[];radius=15;context=30;width=61
  for lo in range(0,T,chunk):
   ts=list(range(lo,min(T,lo+chunk)));B=len(ts);xx=[x.new_zeros(B,width,x.shape[1]) for x in xs];mm=[torch.zeros(B,width,dtype=torch.bool,device=x.device) for x in xs];positions=torch.zeros(B,width,dtype=torch.long,device=xs[0].device)
   for j,t in enumerate(ts):
    gl=max(0,t-context);gh=min(T,t+context+1);jl=gl-(t-context);jh=jl+(gh-gl);positions[j]=torch.arange(t-context,t+context+1,device=positions.device)
    for f in range(3):
     xx[f][j,jl:jh]=xs[f][gl:gh];mm[f][j,jl:jh]=ms[f][gl:gh]
     if bool(ms[f][t]):xx[f][j,context]=bs[f][t].detach()
   new_a,_=self.contributions_batch(xx,mm,positions);delta=[]
   for j,t in enumerate(ts):
    gl=max(0,t-radius);gh=min(T,t+radius+1);jl=gl-(t-context);jh=jl+(gh-gl);local=base_a[gl:gh].sum()-new_a[j,jl:jh].sum();cf=base_a.detach().clone();cf[gl:gh]=new_a[j,jl:jh].detach();numeric=te*((g.detach()+base_r.detach())-(g.detach()+cf.sum()/te));delta.append(local+(numeric-local).detach())
   out.append(torch.stack(delta))
  return torch.cat(out)
class Probe(CTW):
 def __init__(self,dims=(512,128,768),model_seed=26027):super().__init__(dims,model_seed)
 def forward(self,xs,ms,g=None):return self.residual(xs,ms)
def fractional_lme(e,rho=.2,tau=1.):
 e=torch.sort(e,descending=True).values;n=e.numel();mass=rho*n;k=int(math.floor(mass));w=mass-k;parts=[]
 if k:parts.append(e[:k])
 if w>1e-12:parts.append(e[k:k+1]+math.log(w))
 z=torch.cat(parts);return tau*(torch.logsumexp(z/tau,0)-math.log(mass))
def ctw_loss(model,row,replacement):
 xs,ms,g,y=row['X'],row['masks'],row['G'],row['y'];logit=model(xs,ms,g);e=torch.clamp(model.effects(xs,ms,replacement,g),-12,12);loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,y)
 if float(y)==0.:loss=loss+torch.nn.functional.huber_loss(e,torch.zeros_like(e),delta=1.,reduction='mean')
 else:loss=loss+.25*torch.nn.functional.softplus(-fractional_lme(e))
 return loss,logit,e
def permutation(rows,seed):
 groups={}
 for r in rows:groups.setdefault(r['T'],[]).append(r)
 out=[];man=[]
 for T,rr in groups.items():
  rr=sorted(rr,key=lambda z:z['id']);n=len(rr);shift=0 if n==1 else 1+seed%(n-1)
  for i,r in enumerate(rr):
   d=rr[(i+shift)%n];out.append({**r,'X':d['X'],'masks':d['masks'],'oof_b':d['oof_b']});man.append({'recipient':r['id'],'donor':d['id'],'donor_fold':fold(d['id']),'T':T,'nonself':r['id']!=d['id'],'raw_sha':tensor_ch(d['X']),'mask_sha':tensor_ch(d['masks']),'b_sha':tensor_ch(d['oof_b']),'availability':availability_counts(d['masks']),'intervention_coverage':intervention_count(d['masks'])})
 return out,man
def bootstrap_indices(n,B,seed,labels=None):
 rng=np.random.default_rng(seed);out=[]
 for _ in range(B):
  for attempt in range(100):
   ix=rng.integers(0,n,n)
   if labels is None or len(set(np.asarray(labels)[ix]))==2:break
  else:raise RuntimeError('bootstrap class redraw exhausted')
  out.append(ix)
 return out
def three_bootstraps(y,npos,nmixed,B=2000):return {'all32':bootstrap_indices(len(y),B,26031,y),'positive':bootstrap_indices(npos,B,26032),'mixed':bootstrap_indices(nmixed,B,26033)}
