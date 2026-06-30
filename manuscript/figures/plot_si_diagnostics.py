#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (
    CHARCOAL,
    ERRORBAR_KW,
    FAINT_GRID,
    LINEAR_COLOR,
    MODEL_COLORS,
    NONLINEAR_COLOR,
    save_figure,
    setup_style,
    style_axes,
)

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent.parent
EXP = ROOT / "experiments_davis2024"
MECH = EXP / "mechanism_results"
OUT = THIS / "_si_diagnostics"
MODEL_LABELS = {"exp7a":"MT-FT", "exp7c":"ST-FT", "exp7d":"ST-TFS"}

def load(rel):
    return json.loads((MECH/rel).read_text())

def main():
    m1=load('mechanism_m1_results.json')['aggregated']
    m2=load('mechanism_m2_results.json')['aggregated']
    m3=load('mechanism_m3_results.json')['aggregated']
    m3b=load('mechanism_m3b_nonlinear_probe.json')['results']['aggregated']
    m5=load('mechanism_m5a_results.json')['aggregated']
    setup_style()
    fig,axes=plt.subplots(2,2,figsize=(8.2,6.4))
    # rank preservation
    ax=axes[0,0]
    perts=[('scrambled_random','Random'),('sorted_line','Line'),('swapped_bsite','B swap'),('template_dap4','Template')]
    x=np.arange(len(perts)); width=0.24
    for j,exp in enumerate(['exp7a','exp7c']):
        vals=[m1[exp][k]['spearman'] for k,_ in perts]
        ax.bar(x+(j-0.5)*width, vals, width, color=MODEL_COLORS[exp], label=MODEL_LABELS[exp])
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in perts],rotation=20,ha='right',fontsize=8)
    ax.set_ylabel('Spearman rho',fontsize=9); ax.set_title('a  Rank preservation under geometry changes',loc='left',fontsize=10,fontweight='bold')
    ax.axhline(0,color=CHARCOAL,lw=0.5); ax.set_ylim(-0.2,1.05); ax.legend(fontsize=8,frameon=False)
    # scaling
    ax=axes[0,1]
    scales=[0.70,0.80,0.90,1.00,1.10,1.20,1.30]
    for exp in ['exp7a','exp7c']:
        ys=[]
        for s in scales:
            vals=[]
            for mat,d in m2[exp]['per_material'].items():
                pred=d[f'{s:.2f}']
                base=d['1.00']
                vals.append((pred-base)/base*100.0)
            ys.append(np.mean(vals))
        ax.plot(scales,ys,marker='o',ms=3,lw=1,color=MODEL_COLORS[exp],label=MODEL_LABELS[exp])
    ax.axhline(0,color=CHARCOAL,lw=0.5); ax.axvline(1.0,color='0.85',lw=0.6)
    ax.set_xlabel('Uniform coordinate scale',fontsize=9); ax.set_ylabel('Mean Δprediction (%)',fontsize=9)
    ax.set_title('b  Uniform-scaling sensitivity',loc='left',fontsize=10,fontweight='bold'); ax.legend(fontsize=8,frameon=False)
    # probes
    ax=axes[1,0]
    targets=['Vdet','density','OB','frac_N','frac_O','n_atoms']
    labels=[r'$V_{det}$','Density','OB','N frac.','O frac.','Atoms']
    x=np.arange(len(targets)); width=0.24
    vals_lin=[m3['exp7a']['probe_results'][t]['r2_embedding'] for t in targets]
    vals_krr=[m3b['exp7a']['probe_results'][t]['r2_krr'] for t in targets]
    ax.bar(x-width/2,vals_lin,width,color=LINEAR_COLOR,label='Linear')
    ax.bar(x+width/2,vals_krr,width,color=NONLINEAR_COLOR,label='Nonlinear')
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=25,ha='right',fontsize=8)
    ax.set_ylim(-0.2,1.05); ax.set_ylabel(r'$R^2$ (CV)',fontsize=9)
    ax.set_title('c  Chemical information in the baseline embedding',loc='left',fontsize=10,fontweight='bold')
    ax.axhline(0,color=CHARCOAL,lw=0.5); ax.legend(fontsize=8,frameon=False,loc='lower left',ncols=2)
    # stability
    ax=axes[1,1]
    exps=['exp7a','exp7c','exp7d']
    vals=[m5[e]['mean'] for e in exps]
    err=[m5[e]['std'] for e in exps]
    ax.bar(np.arange(len(exps)), vals, yerr=err, color=[MODEL_COLORS[e] for e in exps], error_kw=ERRORBAR_KW, capsize=ERRORBAR_KW["capsize"])
    ax.set_xticks(np.arange(len(exps))); ax.set_xticklabels([MODEL_LABELS[e].replace(' ','\n') for e in exps],fontsize=8)
    ax.set_ylim(0.94,1.0); ax.set_ylabel('Mean cross-fold cosine',fontsize=9)
    ax.set_title('d  Descriptor stability across checkpoints',loc='left',fontsize=10,fontweight='bold')
    for a in axes.ravel():
        style_axes(a, grid=True)
    fig.tight_layout()
    save_figure(fig, OUT)
    print(OUT)
if __name__=='__main__': main()
