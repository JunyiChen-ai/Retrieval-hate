import numpy as np
from relation_v10.copula import dependence_clusters, cluster_weights


def clusters(values,threshold=.999):
    x=np.asarray(values,float);groups,corr=dependence_clusters(x,threshold)
    # Correlation is undefined for constants; exact equality remains enough to
    # guarantee replication invariance for constant duplicate experts.
    parent=list(range(len(groups)))
    merged=[];used=set()
    for i,g in enumerate(groups):
        if i in used:continue
        union=list(g);used.add(i)
        for j,h in enumerate(groups[i+1:],i+1):
            if j not in used and np.array_equal(x[:,g[0]],x[:,h[0]]):union+=h;used.add(j)
        merged.append(sorted(union))
    return merged,corr


def fit(values):
    groups,corr=clusters(values);x=np.asarray(values,float);classes=[]
    for group in groups:
        unique=[]
        for member in group:
            for eq in unique:
                if np.array_equal(x[:,member],x[:,eq[0]]):eq.append(member);break
            else:unique.append([member])
        classes.append(unique)
    representatives=np.stack([x[:,eq].mean(1) for unique in classes for eq in unique],1)
    collapsed=[];offset=0
    for unique in classes:collapsed.append(list(range(offset,offset+len(unique))));offset+=len(unique)
    weights,quality,_=cluster_weights(representatives,collapsed,True)
    mass=np.asarray([weights[g].sum() for g in collapsed]);mass/=mass.sum()
    return {"clusters":groups,"replication_classes":classes,"cluster_mass":mass,"quality":quality,"correlation":corr}


def huber_barycenter(values,state,delta=.15,iterations=12):
    x=np.asarray(values,float);classes=state.get("replication_classes",[[[i] for i in g] for g in state["clusters"]]);centers=np.stack([np.median(np.stack([x[:,eq].mean(1) for eq in unique],1),axis=1) for unique in classes],1)
    mass=np.asarray(state["cluster_mass"]);location=np.sum(centers*mass,axis=1)
    for _ in range(iterations):
        residual=centers-location[:,None];robust=np.minimum(1.,delta/np.maximum(np.abs(residual),1e-12))
        weight=robust*mass;location=np.sum(weight*centers,axis=1)/np.maximum(weight.sum(1),1e-12)
    return location
