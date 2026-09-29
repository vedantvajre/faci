from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description="Plot the paper's token-site figure from a completed notebook run.")
parser.add_argument('--input', type=Path, required=True,
                    help='Path to analysis/layer_site_summary.csv from the exported run')
parser.add_argument('--output', type=Path, default=Path('reproduced/fig_token_sites.pdf'),
                    help='Output PDF path; a PNG preview is written beside it')
args = parser.parse_args()
if args.output.suffix.lower() != '.pdf':
    parser.error('--output must end in .pdf')
args.output.parent.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(args.input)
plt.rcParams.update({'font.family':'serif','font.serif':['Nimbus Roman','Liberation Serif','DejaVu Serif'],
                     'font.size':8.5,'axes.titlesize':10,'axes.labelsize':9,
                     'xtick.labelsize':8,'ytick.labelsize':8,
                     'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(6.221,2.68),sharey=True)
sites=[('cls','Read-out token','#31688e','o'),
       ('edited','Edited hypothesis token','#cc6235','^'),
       ('control','Unchanged content token','#54875b','s')]
for ax,key,title in zip(axs,['bert_nli','roberta_nli'],['BERT (NLI)','RoBERTa (NLI)']):
    ax.axhline(0,color='#999999',linewidth=.65,zorder=0)
    ax.axhline(1,color='#aaaaaa',linestyle=':',linewidth=.7,zorder=0)
    ax.axvspan(10.65,11.35,color='#e5e5e5',zorder=0)
    ax.axvline(5.5,color='#999999',linestyle='--',linewidth=.7,zorder=0)
    for site,label,color,marker in sites:
        s=data[(data.model==key)&(data.site==site)].sort_values('block')
        assert s.block.tolist()==list(range(12))
        ax.plot(s.block,s.median_recovery,label=label,color=color,marker=marker,
                markersize=3.0,linewidth=1.35,markeredgewidth=.35,zorder=3)
        ax.fill_between(s.block,s.pointwise_ci95_low,s.pointwise_ci95_high,
                        color=color,alpha=.15,zorder=1,linewidth=0)
    ax.set(title=title,xlabel=r'Transformer block $\ell$',xlim=(-.35,11.35))
    ax.set_xticks([0,2,5,8,11])
    ax.tick_params(length=3,width=.7)
axs[0].set_ylabel('Median signed recovery')
axs[0].set_ylim(min(-.045,float(data.pointwise_ci95_low.min())-.015),
                 max(1.045,float(data.pointwise_ci95_high.max())+.015))
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.018),ncol=3,
           frameon=False,fontsize=8,handlelength=1.8,columnspacing=1.25,handletextpad=.45)
fig.subplots_adjust(left=.083,right=.995,top=.89,bottom=.27,wspace=.115)
fig.savefig(args.output)
fig.savefig(args.output.with_suffix('.png'),dpi=200)
print(f"Saved {args.output} and {args.output.with_suffix('.png')}")
