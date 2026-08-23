from __future__ import annotations
import torch
from torch import nn

class StateClassifier(nn.Module):
    def __init__(self,z_dim,action_dim=256,dropout=.1):
        super().__init__();self.z=nn.Sequential(nn.Linear(z_dim,256),nn.LayerNorm(256),nn.GELU())
        layer=nn.TransformerEncoderLayer(256,4,512,dropout,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(layer,2);self.cls=nn.Parameter(torch.zeros(1,1,256))
        self.head=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,1))
    def forward(self,z,tokens,mask=None):
        b=z.shape[0];cls=self.cls.expand(b,-1,-1);seq=torch.cat([cls,tokens],1)
        if mask is not None:
            if mask.shape != tokens.shape[:2]:
                raise ValueError("token mask must be [batch,n_tokens]")
            cls_mask=torch.zeros((b,1),dtype=torch.bool,device=mask.device)
            mask=torch.cat([cls_mask,mask.to(dtype=torch.bool)],dim=1)
        state=self.encoder(seq,src_key_padding_mask=mask)[:,0]
        return self.head(torch.cat([self.z(z),state],-1)).squeeze(-1)

class ActionTokenizer(nn.Module):
    def __init__(self,n_windows=30,d_model=256,max_order=60,dropout=.1):
        super().__init__();self.ocr=nn.Linear(768,d_model);self.dense=nn.Linear(1024,d_model)
        self.kind=nn.Embedding(2,d_model);self.order=nn.Embedding(max_order,d_model)
        pos=torch.arange(n_windows)[:,None];div=torch.exp(torch.arange(0,d_model,2)*(-torch.log(torch.tensor(10000.0))/d_model))
        pe=torch.zeros(n_windows,d_model);pe[:,0::2]=torch.sin(pos*div);pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("window_pe",pe);self.empty=nn.Parameter(torch.zeros(2,d_model))
        self.norm=nn.LayerNorm(d_model);self.act=nn.GELU();self.drop=nn.Dropout(dropout)
    def forward(self,outcome,kind,window,is_empty=None,acquired_order=None):
        if kind.ndim!=1 or window.ndim!=1: raise ValueError("kind/window must be vectors")
        projected=torch.stack([self.ocr(x[:768]) if int(k)==0 else self.dense(x[:1024])
                               for x,k in zip(outcome,kind)])
        acquired_order=torch.zeros_like(window) if acquired_order is None else acquired_order
        token=projected+self.kind(kind)+self.window_pe[window]+self.order(acquired_order)
        if is_empty is not None:
            empty=self.empty[kind]+self.kind(kind)+self.window_pe[window]+self.order(acquired_order)
            token=torch.where(is_empty[:,None],empty,token)
        return self.drop(self.act(self.norm(token)))

class SharedSpecialistPolicy(nn.Module):
    """Shared trunk with registered joint/OCR/dense specialist heads."""
    def __init__(self,input_dim,arms=("B2","B3","B4","B5","B6","B7","B8","B9","B10","B11","B12")):
        super().__init__();self.arms=tuple(arms)
        self.shared=nn.Sequential(nn.Linear(input_dim,512),nn.GELU(),nn.LayerNorm(512))
        self.heads=nn.ModuleDict({f"{a}_{s}":nn.Linear(512,1) for a in self.arms for s in ("ocr","dense","joint")})
    def forward(self,x,arm,specialist):
        key=f"{arm}_{specialist}"
        if key not in self.heads: raise KeyError("unregistered policy head: "+key)
        return self.heads[key](self.shared(x)).squeeze(-1)

TRAINING_GRID={"learning_rate":(1e-4,3e-4),"weight_decay":(0.0,1e-4),"dropout":(0.1,0.3),
               "utility_loss_w":(0.5,1.0),"ranking_loss_w":(0.0,0.2),"batch_size":32,
               "optimizer":"AdamW","grad_clip":1.0,"max_epochs":60,"mixed_precision":False}
ARM_CONTRACT={
 "B2":{"stochastic_draws":20,"probability_averaging":False},
 "B3":{"kind":"uniform"},"B4":{"kind":"salience","window_asr":False},
 "B5":{"kind":"learned_stopping"},"B6":{"routes":"B3_feasible_package"},
 "B7":{"kind":"cost_aware_stopping"},"B8":{"kind":"myopic_classifier"},
 "B9":{"kind":"additive_singleton","set_interactions":False},
 "B10":{"kind":"set_conditional_utility","set_interactions":True},
 "B11":{"kind":"fixed_singleton_label_free"},"B12":{"kind":"registered_policy_ensemble"}}

class UtilityPolicy(nn.Module):
    def __init__(self,input_dim,dropout=.1):
        super().__init__();self.net=nn.Sequential(nn.Linear(input_dim,1024),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(1024,512),nn.GELU(),nn.Linear(512,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,x):return self.net(x).squeeze(-1)

class AdditiveSingletonPolicy(nn.Module):
    """B9 cannot inspect the purchased set; B10 uses UtilityPolicy for that."""
    def __init__(self,cheap_dim,action_dim=256):
        super().__init__();self.net=nn.Sequential(nn.Linear(cheap_dim+action_dim+2,512),nn.GELU(),nn.Linear(512,1))
    def forward(self,cheap,action_token,remaining_budget,estimated_cost,purchased_tokens=None):
        if purchased_tokens is not None:raise RuntimeError("HALT_B9_SET_INTERACTION")
        x=torch.cat([cheap,action_token,remaining_budget[:,None],estimated_cost[:,None]],dim=-1)
        return self.net(x).squeeze(-1)

def fit_synthetic_interaction(seed=1):
    """Behavioral F9: B10 MLP fits XOR-like set/action utility; additive B9 cannot."""
    torch.manual_seed(seed);x=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]);y=torch.tensor([0.,1.,1.,0.])
    model=nn.Sequential(nn.Linear(2,8),nn.Tanh(),nn.Linear(8,1));opt=torch.optim.Adam(model.parameters(),lr=.05)
    for _ in range(300):
        loss=nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(),y);opt.zero_grad();loss.backward();opt.step()
    acc=((model(x).squeeze()>0)==(y>0.5)).float().mean().item()
    # Least-squares additive model with intercept cannot classify all XOR points.
    A=torch.cat([torch.ones(4,1),x],1);coef=torch.linalg.lstsq(A,y).solution;add=((A@coef)>=.5)==(y>.5)
    return acc,float(add.float().mean())

def fixed_epoch_refit(train_step,epochs,outer_query_callback=None):
    if outer_query_callback is not None:raise RuntimeError("HALT_OUTER_QUERY_CALLBACK")
    for epoch in range(int(epochs)):train_step(epoch)
    return int(epochs)
