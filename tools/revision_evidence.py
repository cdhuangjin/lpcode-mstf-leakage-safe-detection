"""Read-only evidence extraction for the revised manuscript. No model fitting."""
from pathlib import Path
import json, csv, hashlib, statistics, ast
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/07_manuscript/revised_sources'
def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8'))
def ledger(p): return [json.loads(x) for x in (ROOT/p).read_text().splitlines() if x]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(a): return statistics.mean(a)
def table(head,rows): return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def ci(d): return f"[{100*d['low']:.3f}, {100*d['high']:.3f}]"
def run():
    OUT.mkdir(exist_ok=True,parents=True)
    reg=read('results/06_paper_assets/frozen_result_registry.json')
    paths=[Path(f['path']) for b in reg['bundles'].values() for f in b['files'].values()]
    originals=[ROOT/'results/07_manuscript/paper.md',ROOT/'results/07_manuscript/LPcode_MSTF_Manuscript_Full.docx',ROOT/'WileyDesign/Optimal-Design-layout/LPcode_MSTF_Wiley.tex',ROOT/'WileyDesign/Optimal-Design-layout/build_refined_verify/LPcode_MSTF_Wiley.pdf']
    paths+=originals+[ROOT/'results/05_mechanism_analysis/folds.jsonl',ROOT/'results/negative_pair_robustness/raw_results.json']
    snap=OUT/'original_and_evidence_sha256.json'
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in paths}
    if snap.exists(): assert read(str(snap.relative_to(ROOT)))==hashes,'Source changed during manuscript revision'
    else: snap.write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    for b in reg['bundles'].values():
        for f in b['files'].values(): assert digest(Path(f['path']))==f['sha256']
    a=ledger('results/05_mechanism_analysis/folds.jsonl')
    contrasts=read('results/05_mechanism_analysis/ablation_summary.json')['environments']
    b=read('results/02_unseen_llm/summary.json'); c=read('results/03_style_attack/summary.json'); d=read('repro/2502.17749/v1/results/04_cross_language/summary.json')
    av={e:{m:mean([r['f1'] for r in a if r['environment']==e and r['method']==m]) for m in ['A0','A1','A2','A3','A4','A5']} for e in ['clean','unseen']}
    bm={m:mean([v[m]['f1_mean'] for v in b['macro_language_summaries'].values()]) for m in ['mstf','lpcode_original']}
    dm={m:mean([v[m]['f1_mean'] for v in d['cell_summaries'].values()]) for m in ['mstf','lpcode_original']}
    main=[dict(gate='A: strict clean',baseline=av['clean']['A0'],method=av['clean']['A1'],comparison='A1 vs A0; fixed XGBoost',delta=contrasts['clean']['overall']['C1']['mean_delta_f1'],ci=contrasts['clean']['overall']['C1']['ci_95']),
          dict(gate='B: held-out generator',baseline=bm['lpcode_original'],method=bm['mstf'],comparison='Full MSTF vs LPcode original (MLP)',delta=b['paired_mstf_minus_lpcode']['overall']['macro_holdout_language_mean_delta_f1'],ci=b['paired_mstf_minus_lpcode']['overall']['ci_95']),
          dict(gate='C: combined transformation',baseline=c['macro_language_summaries']['combined']['lpcode_original']['f1_mean'],method=c['macro_language_summaries']['combined']['mstf']['f1_mean'],comparison='Full MSTF vs LPcode original (MLP)',delta=c['paired_mstf_minus_lpcode']['by_condition']['combined']['macro_language_mean_delta_f1'],ci=c['paired_mstf_minus_lpcode']['by_condition']['combined']['ci_95']),
          dict(gate='D: held-out language',baseline=dm['lpcode_original'],method=dm['mstf'],comparison='Full MSTF vs LPcode original (MLP)',delta=d['paired_mstf_minus_lpcode']['overall_equal_language_mean_delta_f1'],ci=d['paired_mstf_minus_lpcode']['ci_95'])]
    tokens={}
    for i,r in enumerate(main):
        letter='ABCD'[i]
        for k in ['baseline','method']:tokens[letter+'_'+k]=f"{r[k]:.4f}"
        tokens[letter+'_delta']=f"{100*r['delta']:.3f}"; tokens[letter+'_ci']=ci(r['ci'])
        assert abs(r['method']-r['baseline']-r['delta'])<1e-12
    tokens['TABLE_MAIN']=table(['Gate / comparison','Baseline F1','Method F1','Difference (pp)','95% interval (pp)'],[[r['gate']+'; '+r['comparison'],f"{r['baseline']:.4f}",f"{r['method']:.4f}",f"{r['delta']*100:+.3f}",ci(r['ci'])] for r in main])
    variants=[('A0','Original 10','Endpoints',20),('A1','Original 10','Endpoints + signed difference',30),('A2','Enhanced 28','Endpoints',56),('A3','Enhanced 28','Signed difference only',28),('A4','Enhanced 28','Endpoints + signed difference',84),('A5','Enhanced 28','Endpoints + signed + relative',112)]
    tokens['TABLE_ABLATION']=table(['Variant','Features','Representation','Dimensions','Clean F1','Held-out F1'],[[m,f,r,n,f"{av['clean'][m]:.4f}",f"{av['unseen'][m]:.4f}"] for m,f,r,n in variants])
    cr=[]
    for name in ['C1','C2','C3','C4','C5']:
        l=contrasts['clean']['overall'][name];u=contrasts['unseen']['overall'][name]
        cr.append([name,l['left']+' minus '+l['right'],f"{100*l['mean_delta_f1']:+.3f}",ci(l['ci_95']),f"{100*u['mean_delta_f1']:+.3f}",ci(u['ci_95'])])
        for e in ['clean','unseen']:tokens[name+'_'+e]=f"{100*contrasts[e]['overall'][name]['mean_delta_f1']:+.3f}"
    tokens['TABLE_CONTRASTS']=table(['Contrast','Definition','Clean (pp)','95% interval','Held-out (pp)','95% interval'],cr)
    names={'gpt3.5':'GPT-3.5','gemini-pro':'Gemini-Pro','wizardcoder:33b-v1.1':'WizardCoder-33B','deepseek-coder:33b-instruct':'DeepSeek-Coder-33B'}
    gen=[]
    for key,label in names.items():
        vals=b['macro_language_summaries'][key];dif=b['paired_mstf_minus_lpcode']['by_holdout'][key]
        gen.append([label,f"{vals['lpcode_original']['f1_mean']:.4f}",f"{vals['mstf']['f1_mean']:.4f}",f"{dif['macro_language_mean_delta_f1']*100:+.3f}",ci(dif['ci_95'])])
    tokens['TABLE_GENERATOR']=table(['Held-out generator','LPcode original','Full MSTF','Difference (pp)','95% interval (pp)'],gen)
    attack=[]
    for key in ['clean','comment_removal','identifier_rename','format_normalization','comment_injection','combined']:
        vals=c['macro_language_summaries'][key];dif=c['paired_mstf_minus_lpcode']['by_condition'][key]
        drops=[0,0] if key=='clean' else [100*c['clean_to_attack_drops']['by_condition'][key][m]['macro_language_mean_drop_f1'] for m in ['lpcode_original','mstf']]
        attack.append([key.replace('_',' ').capitalize(),f"{vals['lpcode_original']['f1_mean']:.4f}",f"{vals['mstf']['f1_mean']:.4f}",f"{dif['macro_language_mean_delta_f1']*100:+.3f}",f'{drops[0]:.3f} / {drops[1]:.3f}'])
    tokens['TABLE_ATTACK']=table(['Transformation','LPcode original','Full MSTF','Difference (pp)','Drops: original / MSTF (pp)'],attack)
    lang=[]
    for key,label in [('c','C'),('cpp','C++'),('java','Java'),('py','Python')]:
        vals=d['cell_summaries'][key];dif=d['paired_mstf_minus_lpcode']['by_heldout_language'][key]
        lang.append([label,*[f"{vals[m]['f1_mean']:.4f} ± {vals[m]['f1_std']:.4f}" for m in ['lpcode_original','mstf']],f"{dif['mean_delta_f1']*100:+.3f}"])
    tokens['TABLE_LANGUAGE']=table(['Held-out language','LPcode original: mean ± SD','MSTF: mean ± SD','Difference (pp)'],lang)
    neg=list(csv.DictReader((ROOT/'results/negative_pair_robustness/summary.csv').open()))
    tokens['TABLE_NEGATIVE']=table(['Negatives','A0 F1','A1 F1','Difference (pp)','95% interval (pp)'],[[r['negative_pairing'],f"{float(r['baseline_f1_mean']):.4f}",f"{float(r['mstf_f1_mean']):.4f}",f"{100*float(r['delta_f1_mean']):+.3f}",f"[{100*float(r['ci_95_low']):.3f}, {100*float(r['ci_95_high']):.3f}]"] for r in neg])
    drops=c['clean_to_attack_drops']['by_condition']['combined']
    tokens['DROP_ORIGINAL']=f"{100*drops['lpcode_original']['macro_language_mean_drop_f1']:.3f}"
    tokens['DROP_MSTF']=f"{100*drops['mstf']['macro_language_mean_drop_f1']:.3f}"
    tokens['DROP_REDUCTION']=f"{100*(1-drops['mstf']['macro_language_mean_drop_f1']/drops['lpcode_original']['macro_language_mean_drop_f1']):.2f}"
    evidence=dict(main=main,variants=av,contrasts=contrasts,b=b,c=c,d=d,negative=neg,tokens=tokens)
    (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print('Evidence extracted; source hashes unchanged; tokens:',len(tokens))
    return evidence
if __name__=='__main__':run()
