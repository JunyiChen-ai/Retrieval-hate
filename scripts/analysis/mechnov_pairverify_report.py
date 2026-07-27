#!/usr/bin/env python
"""Reporting tool for the MECHNOV pair-verify pregate. Re-reads the OUT jsons and
prints every number that appears in the record, at 4 dp. No computation of new
treatment quantities: it only formats what mechnov_pairverify.py wrote."""
import json
import os
import sys

REPO = "/data/jehc223/RGCL"
DS = ["hatemm", "zh", "en"]
NAME = {"hatemm": "HateMM", "zh": "MHC-ZH", "en": "MHC-EN"}


def load(d):
    p = os.path.join(REPO, f"scripts/analysis/mechnov_pairverify_{d}_OUT.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    for d in DS:
        O = load(d)
        if O is None:
            print(f"### {NAME[d]}: MISSING")
            continue
        m = O["meta"]
        print("=" * 100)
        print(f"### {NAME[d]}  n_train={m['n_train_items']} pos-rate={m['pos_rate']} "
              f"model={m['model']}  sha={m['script_sha256'][:12]}")
        for space, R in O["spaces"].items():
            c1 = R["control1_5foldmean"]
            po = R["pooled"]
            me = R["control3_mechanism"]
            star = " *** PRIMARY" if space == "fused" else ""
            print(f"\n-- space={space}{star}")
            print(f"   C1 pair-AUC  cos_full={c1['auc_cosine_fullspace']:.4f} "
                  f"cos_pca={c1['auc_cosine_pcaspace']:.4f} | "
                  f"mlp={c1['auc_mlp']:.4f} (D={c1['d_auc_mlp_vs_cos_full']:+.4f} "
                  f"signs {c1['foldsigns_dauc_mlp']}) | "
                  f"logistic={c1['auc_logistic']:.4f} (D={c1['d_auc_logistic_vs_cos_full']:+.4f} "
                  f"signs {c1['foldsigns_dauc_logistic']})")
            print(f"      per-fold dAUC mlp={c1['folddeltas_dauc_mlp']} "
                  f"logistic={c1['folddeltas_dauc_logistic']}")
            print(f"      eval pairs/fold={int(c1['n_eval_pairs'])} "
                  f"same-class rate={c1['same_class_rate']:.4f} | "
                  f"pair-pred posrate mlp={c1['posrate_pairpred_mlp']:.4f} "
                  f"log={c1['posrate_pairpred_logistic']:.4f}")
            print(f"   C2 end-to-end (pooled LOO over all {me['n_items']} train items)")
            print(f"      deployed vote      acc={po['acc_deployed']:.4f} "
                  f"mF1={po['mF1_deployed']:.4f} posrate={po['posrate_deployed']:.4f}")
            print(f"      cos-shape ctrl 2b  acc={po['acc_cos_shape']:.4f} "
                  f"mF1={po['mF1_cos_shape']:.4f} posrate={po['posrate_cos_shape']:.4f} "
                  f"(D vs dep {po['acc_cos_shape'] - po['acc_deployed']:+.4f})")
            for mo in ("mlp", "logistic"):
                for ag in ("max", "mean3"):
                    k = f"{mo}_{ag}"
                    tag = " <PRIMARY>" if ag == "max" else ""
                    print(f"      {k:16s} acc={po['acc_'+k]:.4f} mF1={po['mF1_'+k]:.4f} "
                          f"posrate={po['posrate_'+k]:.4f} "
                          f"Ddep={po['dacc_'+k+'_vs_deployed']:+.4f} "
                          f"D2b={po['dacc_'+k+'_vs_cos_shape']:+.4f} "
                          f"signs={po['foldsigns_'+k]} "
                          f"folds={po['folddeltas_'+k]}{tag}")
            print(f"   C3 mechanism: deployed wrong={me['n_deployed_wrong']} "
                  f"pathology pop (same-class analogue rank<=5)={me['n_pathology_pop']} "
                  f"median rank all={me['median_sc_rank_all']} "
                  f"wrong={me['median_sc_rank_deployed_wrong']}")
            for mo in ("mlp", "logistic"):
                for ag in ("max", "mean3"):
                    k = f"{mo}_{ag}"
                    print(f"      {k:16s} fixed={me[k+'_fixed']} broke={me[k+'_broke']} "
                          f"net={me[k+'_net']:+d} exch={me[k+'_exchange_rate']} "
                          f"patho_fixed={me[k+'_pathology_fixed']}"
                          f"/{me['n_pathology_pop']} "
                          f"({me[k+'_pathology_frac_fixed']})")
            pf = R["per_fold"][0]
            print(f"   fit info: pca_dim={pf['pca_dim']} evr={pf['pca_explained_var']:.4f} "
                  f"pairs_total={pf['n_pairs_total']} pairs_fitted={pf['n_pairs_fitted']} "
                  f"fit same-class rate={pf['fit_same_class_rate']:.4f} "
                  f"secs/fold~{pf['secs']}")


if __name__ == "__main__":
    sys.exit(main())
