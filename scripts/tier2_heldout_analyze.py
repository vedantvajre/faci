
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def _acts(analyzer, texts, layer, batch_size=16):
    out=[]
    for i in range(0, len(texts), batch_size):
        a = analyzer.model.get_embedding(texts[i:i+batch_size], layer=layer, position=analyzer.position)
        out.append(a.detach().float().cpu())
    return torch.cat(out,0).numpy()

def _cv_auc(Xp, Xn, rng, n_splits=5, select=False, layers=None):
    single = not isinstance(Xp, dict)
    if single: Xp={0:Xp}; Xn={0:Xn}; layers=[0]
    layers = layers or sorted(Xp.keys())
    ip, iN = rng.permutation(len(Xp[layers[0]])), rng.permutation(len(Xn[layers[0]]))
    fp, fn = np.array_split(ip, n_splits), np.array_split(iN, n_splits)
    scores, labels = [], []
    for k in range(n_splits):
        tr_p=np.concatenate([fp[j] for j in range(n_splits) if j!=k]); tr_n=np.concatenate([fn[j] for j in range(n_splits) if j!=k])
        best_l, best_a = layers[0], -1
        for l in (layers if select else layers[:1]):
            d=Xp[l][tr_p].mean(0)-Xn[l][tr_n].mean(0); n=np.linalg.norm(d)
            if n==0: continue
            d/=n
            if select:
                sc=np.r_[Xp[l][tr_p]@d, Xn[l][tr_n]@d]; y=np.r_[np.ones(len(tr_p)),np.zeros(len(tr_n))]
                a=roc_auc_score(y,sc)
                if a>best_a: best_a,best_l=a,l
            else: best_l=l
        d=Xp[best_l][tr_p].mean(0)-Xn[best_l][tr_n].mean(0); d/=np.linalg.norm(d)+1e-12
        scores+=list(Xp[best_l][fp[k]]@d)+list(Xn[best_l][fn[k]]@d)
        labels+=[1]*len(fp[k])+[0]*len(fn[k])
    return float(roc_auc_score(labels,scores))

def heldout_tier1(analyzer, pos_texts, neg_texts, seed=0, n_perm=200):
    pos_texts=list(dict.fromkeys(pos_texts)); neg_texts=list(dict.fromkeys(neg_texts))
    rng=np.random.default_rng(seed)
    try: L=analyzer.model.num_layers
    except Exception: L=len(analyzer.model._get_layer_modules())
    layers=list(range(L))
    Xp={l:_acts(analyzer,list(pos_texts),l) for l in layers}; Xn={l:_acts(analyzer,list(neg_texts),l) for l in layers}
    auc=_cv_auc(Xp,Xn,rng,select=True,layers=layers)
    Ap=np.stack([Xp[l] for l in layers],0); An=np.stack([Xn[l] for l in layers],0)
    pool=np.concatenate([Ap,An],1); npos=Ap.shape[1]; null=[]
    for _ in range(n_perm):
        idx=rng.permutation(pool.shape[1])
        null.append(_cv_auc({l:pool[i][idx[:npos]] for i,l in enumerate(layers)},
                            {l:pool[i][idx[npos:]] for i,l in enumerate(layers)}, rng, select=True, layers=layers))
    return {'tier1_separation':auc,'tier1_perm_null_mean':float(np.mean(null)),
            'tier1_perm_null_p95':float(np.percentile(null,95)),
            'tier1_protocol':'5-fold cross-fitted held-out AUC; train-fold layer selection; 200-permutation control'}

def heldout_tier2(analyzer, pos_texts, neg_texts, seed=0, paired=False, n_splits=5):
    # groups: variants of one underlying item stay together. paired tasks (qa/factual):
    # neg_i is derived from pos_i's context/prompt, so both share group key pos_texts[i].
    rng=np.random.default_rng(seed+1)
    if paired:
        keys=list(dict.fromkeys(pos_texts))
        kid={k:i for i,k in enumerate(keys)}
        gp=[kid[t] for t in pos_texts]; gn=[kid[pos_texts[i]] for i in range(len(neg_texts))]
    else:
        pos_texts=list(dict.fromkeys(pos_texts)); neg_texts=list(dict.fromkeys(neg_texts))
        gp=list(range(len(pos_texts))); gn=[len(pos_texts)+i for i in range(len(neg_texts))]
    texts=list(pos_texts)+list(neg_texts)
    y=np.r_[np.ones(len(pos_texts)),np.zeros(len(neg_texts))]
    groups=np.array(gp+gn)
    try: L=analyzer.model.num_layers
    except Exception: L=len(analyzer.model._get_layer_modules())
    layers=list(range(L+1))  # probe also sees embedding layer, as before
    X={l:_acts(analyzer,texts,l) for l in layers}
    ug=rng.permutation(np.unique(groups)); folds=np.array_split(ug,n_splits)
    accs=[]; chosen=[]
    for k in range(n_splits):
        te=np.isin(groups,folds[k]); tr=~te
        if y[te].min()==y[te].max() or y[tr].min()==y[tr].max(): continue
        best_l,best_a=layers[0],-1
        for l in layers:
            try: a=cross_val_score(LogisticRegression(max_iter=1000),X[l][tr],y[tr],cv=3).mean()
            except Exception: a=0.0
            if a>best_a: best_a,best_l=a,l
        clf=LogisticRegression(max_iter=1000).fit(X[best_l][tr],y[tr])
        accs.append(clf.score(X[best_l][te],y[te])); chosen.append(best_l)
    import collections
    return {'tier2_best_accuracy': float(np.mean(accs)) if accs else float('nan'),
            'tier2_best_layer': int(collections.Counter(chosen).most_common(1)[0][0]) if chosen else -1,
            'protocol':'nested grouped 5-fold: layer selected by inner 3-fold CV on train folds; grouped by underlying item for paired tasks'}
