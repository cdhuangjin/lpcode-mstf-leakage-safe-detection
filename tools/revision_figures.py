"""Five manuscript arguments drawn deterministically from saved evidence only.

Contract: 183 mm, white background, editable PDF/SVG, 600 dpi PNG/TIFF.
Figures 1/2 define representation and evaluation; Figure 3 separates controlled
from system contrasts; Figure 4 contrasts refitting with descriptive reliance;
Figure 5 bounds the claim using every registered transformation and domain.
No model fitting, cherry-picking, smoothing, or synthetic experimental data.
"""
from pathlib import Path
import sys, json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from revision_evidence import ROOT, OUT
sys.path.insert(0,str(ROOT/'repro/2502.17749/v1'))
from lpcode_v1.release_figure_qa import require_matplotlib_panel_alignment

DEST=ROOT/'results/07_manuscript/revised_figures'
TEAL='#087F8C'; GRAY='#687589'; BLUE='#3269A8'; INK='#192E43'; LIGHT='#EAF3F5'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
    'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'text.color':INK,'axes.labelcolor':INK})
E=json.loads((OUT/'evidence.json').read_text()); M=json.loads((ROOT/'results/05_mechanism_analysis/feature_importance/mechanism_summary.json').read_text())
def save(fig,n):
    base=DEST/f'figure{n}'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(base)+'.alignment.json',overlay_svg=str(base)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(base)+'.pdf',facecolor='white')
    fig.savefig(str(base)+'.svg',facecolor='white')
    fig.savefig(str(base)+'.png',dpi=600,facecolor='white')
    fig.savefig(str(base)+'.tiff',dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
def box(ax,x,y,w,h,label,fill=LIGHT,color=INK,size=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.018',linewidth=.7,edgecolor='#B4C9D4',facecolor=fill))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=color,linespacing=1.5)
def arrow(ax,a,b): ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':GRAY,'lw':1,'shrinkA':3,'shrinkB':3})
def panel(ax,label,title):
    ax.set_title(title,loc='left',pad=13,fontweight='bold')
    ax.annotate(label,xy=(0,1),xycoords='axes fraction',xytext=(-20,13),textcoords='offset points',fontweight='bold',fontsize=10)
    ax.grid(axis='x',color='#E7EDF1',lw=.5);ax.set_axisbelow(True)
def interval(ax,x,y,ci,color,marker='o'):
    ax.errorbar(x,y,xerr=np.array([[x-100*ci['low']],[100*ci['high']-x]]),fmt=marker,color=color,markersize=5,capsize=2,lw=1.2)
def run():
    DEST.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(183/25.4,3.25));fig.subplots_adjust(left=.015,right=.985,bottom=.03,top=.95);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    box(ax,.015,.72,.20,.18,'Human source h');box(ax,.015,.36,.20,.18,'Candidate c')
    box(ax,.27,.72,.23,.18,'Shared extractor\n28-D endpoint Fh');box(ax,.27,.36,.23,.18,'Shared extractor\n28-D endpoint Fc')
    arrow(ax,(.215,.81),(.27,.81));arrow(ax,(.215,.45),(.27,.45))
    box(ax,.565,.36,.41,.54,'Endpoint blocks: Fh ; Fc\n\nSigned change: ΔF = Fc − Fh\n\nRelative change: ΔF / (|Fh| + ε)',size=9)
    arrow(ax,(.50,.81),(.565,.81));arrow(ax,(.50,.45),(.565,.45))
    box(ax,.035,.035,.28,.17,'112-D MSTF\n[Fh ; Fc ; ΔF ; R]',fill='#DDEEF0')
    box(ax,.39,.035,.22,.17,'Fixed XGBoost');box(ax,.69,.035,.28,.17,'Matching paraphrase\n/ non-matching pair')
    arrow(ax,(.77,.36),(.77,.26));arrow(ax,(.77,.26),(.17,.26));arrow(ax,(.17,.26),(.17,.205));arrow(ax,(.315,.12),(.39,.12));arrow(ax,(.61,.12),(.69,.12))
    save(fig,1)
    fig,ax=plt.subplots(figsize=(183/25.4,3.0));fig.subplots_adjust(left=.015,right=.985,bottom=.035,top=.97);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    box(ax,.025,.74,.26,.2,'Positive provenance bank\nBoth human origins tracked')
    box(ax,.37,.74,.26,.2,'Exact-content components\nAssign whole components')
    box(ax,.715,.74,.26,.2,'Partition-local pairs\nBalanced cross-component negatives',size=8)
    arrow(ax,(.285,.84),(.37,.84));arrow(ax,(.63,.84),(.715,.84))
    box(ax,.19,.39,.62,.19,'Verify dual-endpoint + exact-code isolation\nReuse the same train/test pair hashes',fill='#DDEEF0')
    arrow(ax,(.845,.74),(.845,.65));arrow(ax,(.845,.65),(.50,.65));arrow(ax,(.50,.65),(.50,.58))
    for x,title,subtitle in [(.02,'Gate A','Strict clean\nA1 vs A0'),(.27,'Gate B','Held-out generator\nFull MSTF vs original'),(.52,'Gate C','Deterministic edits\nFull MSTF vs original'),(.77,'Gate D','Held-out language\nThree-seed descriptive')]:
        box(ax,x,.015,.21,.21,title+'\n'+subtitle,size=8);arrow(ax,(x+.105,.34),(x+.105,.225))
    ax.plot([.125,.875],[.34,.34],color=GRAY,lw=.8);arrow(ax,(.50,.39),(.50,.34));save(fig,2)
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,3.8));fig.subplots_adjust(left=.085,right=.975,bottom=.24,top=.86,hspace=1.15,wspace=.25)
    titles=['Strict clean','Held-out generator','Combined transformation','Held-out language']
    for i,(ax,r) in enumerate(zip(axs.flat,E['main'])):
        panel(ax,'abcd'[i],titles[i]);ax.grid(False);x=r['delta']*100;interval(ax,x,0,r['ci'],TEAL)
        ax.set(xlim=(-.5,13),ylim=(-.65,.65),yticks=[],xticks=[0,4,8,12]);ax.axvline(0,lw=.6,color=GRAY)
        ax.text(x,.35,f"+{x:.3f} pp",ha='center',fontsize=9,fontweight='bold',color=TEAL)
        ax.text(.02,-.48,'A1 − A0 · fixed XGBoost' if i==0 else 'Full MSTF − original MLP'+(' · descriptive' if i==3 else ''),transform=ax.transAxes,fontsize=7.5)
        if i>=2:ax.set_xlabel('Paired F1 difference (pp)',labelpad=21)
    save(fig,3)
    fig,axs=plt.subplots(1,2,figsize=(183/25.4,3.55));fig.subplots_adjust(left=.12,right=.975,bottom=.19,top=.82,wspace=.7)
    ax=axs[0];panel(ax,'a','Controlled block contributions')
    labels=['C1: A1 − A0','C2: A2 − A0','C3: A4 − A1','C4: A5 − A4','C5: A5 − A0']
    for offset,env,color,marker in [(.12,'clean',TEAL,'o'),(-.12,'unseen',BLUE,'s')]:
        for i,name in enumerate(['C1','C2','C3','C4','C5']):
            r=E['contrasts'][env]['overall'][name];interval(ax,r['mean_delta_f1']*100,i+offset,r['ci_95'],color,marker)
    ax.set(yticks=range(5),yticklabels=labels,ylim=(4.6,-.6),xlim=(-.4,6.5),xlabel='Paired F1 difference (pp)');ax.axvline(0,lw=.7,color=GRAY)
    ax.plot([],[],color=TEAL,marker='o',ls='',label='Clean');ax.plot([],[],color=BLUE,marker='s',ls='',label='Held-out generator');ax.legend(loc='upper left',bbox_to_anchor=(-.15,1.28),ncol=2,fontsize=7)
    ax=axs[1];panel(ax,'b','Leading group depends on context');ax.grid(False)
    envs=['clean','unseen_llm','combined_attack','cross_language'];labs=['Clean','Unseen LLM','Combined','Held-out language']
    for i,(env,label) in enumerate(zip(envs,labs)):
        r=max(M['rankings'][env]['group_rows'],key=lambda r:r['permutation_mean'])
        ax.barh(i,r['permutation_mean']*100,height=.24,color=TEAL if i<3 else BLUE)
        ax.text(.08,i+.30,'Relative · structural' if i<3 else 'Signed · original style',fontsize=7,va='center')
    ax.set(yticks=range(4),yticklabels=labs,ylim=(3.65,-.55),xlim=(0,2.8),xlabel='Mean permutation F1 decrease (pp)')
    ax.text(0,-.22,'Descriptive ranking; not causal attribution',transform=ax.transAxes,fontsize=7)
    save(fig,4)
    fig,axs=plt.subplots(1,2,figsize=(183/25.4,3.75));fig.subplots_adjust(left=.17,right=.97,bottom=.18,top=.82,wspace=.7)
    ax=axs[0];panel(ax,'a','Sensitivity to deterministic edits')
    cond=['comment_removal','identifier_rename','format_normalization','comment_injection','combined'];labs=['Comment removal','Identifier rename','Formatting','Comment injection','Combined']
    for offset,m,col,marker in [(-.12,'lpcode_original',GRAY,'s'),(.12,'mstf',TEAL,'o')]:
        for i,key in enumerate(cond):
            r=E['c']['clean_to_attack_drops']['by_condition'][key][m];interval(ax,r['macro_language_mean_drop_f1']*100,i+offset,r['ci_95'],col,marker)
    ax.set(yticks=range(5),yticklabels=labs,ylim=(4.6,-.6),xlim=(-.7,7.7),xlabel='Clean-to-edit F1 drop (pp)');ax.axvline(0,color=GRAY,lw=.6)
    ax.plot([],[],color=GRAY,marker='s',ls='',label='Original MLP');ax.plot([],[],color=TEAL,marker='o',ls='',label='Full MSTF');ax.legend(loc='upper left',bbox_to_anchor=(-.25,1.28),ncol=2,fontsize=7)
    ax=axs[1];panel(ax,'b','Transfer across evaluated domains')
    pairs=[('GPT-3.5',E['b']['paired_mstf_minus_lpcode']['by_holdout']['gpt3.5']),('Gemini-Pro',E['b']['paired_mstf_minus_lpcode']['by_holdout']['gemini-pro']),('WizardCoder',E['b']['paired_mstf_minus_lpcode']['by_holdout']['wizardcoder:33b-v1.1']),('DeepSeek',E['b']['paired_mstf_minus_lpcode']['by_holdout']['deepseek-coder:33b-instruct'])]+[(l,E['d']['paired_mstf_minus_lpcode']['by_heldout_language'][k]) for k,l in [('c','C'),('cpp','C++'),('java','Java'),('py','Python')]]
    for i,(label,r) in enumerate(pairs): interval(ax,100*r.get('mean_delta_f1',r.get('macro_language_mean_delta_f1')),i,r['ci_95'],TEAL if i<4 else BLUE,'o' if i<4 else 's')
    ax.set(yticks=range(8),yticklabels=[x[0] for x in pairs],ylim=(7.6,-.6),xlim=(0,10),xticks=[0,5,10],xlabel='Full MSTF − original MLP (pp)');ax.axhline(3.5,color='#CFD9E1',lw=.7)
    save(fig,5)
    print('Exported five figures in PDF/SVG/PNG/TIFF.')
if __name__=='__main__':run()
