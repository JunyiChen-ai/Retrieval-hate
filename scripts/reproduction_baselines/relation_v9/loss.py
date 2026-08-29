"""Weak-label objective and expert corruption consistency."""
import torch
import torch.nn.functional as F


def topk_bag(logits,valid,divisor=8):
    rows=[]
    for i in range(len(logits)):
        z=logits[i,valid[i]];rows.append(z.topk(max(1,len(z)//divisor)).values.mean())
    return torch.stack(rows)


def weak_loss(clean,corrupt,valid,video_label,smooth=.01,consistency=.1):
    prior=F.binary_cross_entropy_with_logits(clean["prior_logit"],video_label)
    bag=F.binary_cross_entropy_with_logits(topk_bag(clean["frame_logit"],valid),video_label)
    negative=(torch.sigmoid(clean["frame_logit"])*valid*(1-video_label[:,None])).sum()/valid.sum().clamp_min(1)
    temporal=[]
    for i in range(len(valid)):
        z=clean["locator_logit"][i,valid[i]];temporal.append((z[1:]-z[:-1]).abs().mean() if len(z)>1 else z.new_zeros(()))
    consistency_loss=F.mse_loss(clean["prior_logit"],corrupt["prior_logit"])+F.mse_loss(clean["locator_logit"][valid],corrupt["locator_logit"][valid])
    total=prior+bag+.1*negative+smooth*torch.stack(temporal).mean()+consistency*consistency_loss
    return total,{"prior_bce":prior,"bag_bce":bag,"negative":negative,"smooth":torch.stack(temporal).mean(),"consistency":consistency_loss}
