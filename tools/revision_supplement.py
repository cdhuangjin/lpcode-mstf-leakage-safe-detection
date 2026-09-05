"""Supplementary tables generated from existing ledgers and audited summaries."""
import csv,json,subprocess,re
from revision_evidence import ROOT,OUT,table
P=ROOT/'results/07_manuscript'
def csvtable(path,cols):
    rows=list(csv.DictReader((ROOT/path).open(encoding='utf-8-sig')))
    return table(cols,[[f'{float(r[c]):.4f}' if c not in ['negative_pairing','language','condition','method','environment','group'] and r[c] and c!='n_fold_records' else r[c] for c in cols] for r in rows])
def run():
    e=json.loads((OUT/'evidence.json').read_text());reg=json.loads((ROOT/'results/06_paper_assets/frozen_result_registry.json').read_text())
    s='# Supplementary information\n\n## Multi-view coding-style transitions for leakage-safe paired detection of LLM-paraphrased code\n\nJin Huang, Qiao Li and Qisen Gao\n\nAll tables derive from saved evidence. No model fitting or new formal experiment was performed during this revision. F1 is positive-class F1; equal-weight means across experimental cells are not class-macro F1.\n\n# S1 Controlled contrast intervals\n\n'+e['tokens']['TABLE_CONTRASTS']+'\n\nC1: signed differences with original features; C2: endpoint expansion; C3: endpoint expansion with signed differences; C4: relative block; C5: full representation versus original endpoints. Intervals retain folds inside three seed clusters. A positive interval for the extremely small unseen C4 is not a claim of practical importance.\n\n# S2 Source registry\n\n'
    s+=table(['Gate','Protocol','Manifest SHA-256'],[[g,b['protocol_version'],b['files']['manifest.json']['sha256']] for g,b in reg['bundles'].items()])
    s+='\n\nFull file digests are supplied in the revision provenance snapshot and frozen registry. SHA-256 values identify files, not evidence of methodological sufficiency by themselves.\n\n# S3 Per-generator and per-language results\n\n'+e['tokens']['TABLE_GENERATOR']+'\n\n'+e['tokens']['TABLE_LANGUAGE']
    s+='\n\n# S4 Negative-pair sensitivity\n\n'+e['tokens']['TABLE_NEGATIVE']+'\n\nThese are A1–A0 comparisons with original ten features, not full MSTF. Current, random and hard refer to the registered negative construction only. Positives and isolation constraints remain fixed.\n\n# S5 Transformation uncertainty\n\n'
    rows=[]
    for cond,v in e['c']['clean_to_attack_drops']['by_condition'].items():
        if cond=='clean':continue
        for m in ['lpcode_original','mstf']:
            r=v[m];ci=r['ci_95'];rows.append([cond.replace('_',' '),m.replace('_',' '),f"{r['macro_language_mean_drop_f1']*100:.3f}",f"[{100*ci['low']:.3f}, {100*ci['high']:.3f}]"])
    s+=table(['Condition','Method','Drop (pp)','95% seed-cluster CI (pp)'],rows)
    s+='\n\n# S6 Interpretation and access\n\nAll five quantitative transformations are retained; combined applies removal, renaming and formatting, without injection. Parsing checks do not prove semantic equivalence. Per-language transformed scores, change fractions and parser-regression statistics remain in the accompanying attack decomposition CSV. Full grouped importance and ranking stability are available in the local mechanism CSVs. These materials are included in the publicly accessible research deposit.\n\nCode, pinned dependencies, frozen records, figure source data and reproduction commands are publicly deposited at <https://github.com/cdhuangjin/lpcode-mstf-leakage-safe-detection/tree/1b6a5b9f7f274b22a718b53219581d5f57a30792>. Anonymous access was verified on 5 September 2026. Raw LPcode data are obtained from its original pinned repository, with checksums supplied in the deposit. Open-reuse licensing remains subject to author confirmation; no archival DOI or independent formal retraining is claimed.\n'
    (P/'LPcode_MSTF_Supplementary.md').write_text(s,encoding='utf-8')
    tex=subprocess.run(['pandoc','-f','markdown','-t','latex','-s','-V','geometry:margin=20mm','-V','fontsize:10pt','-V','colorlinks:true'],input=s,text=True,encoding='utf-8',stdout=subprocess.PIPE,check=True).stdout
    tex=tex.replace(r'\begin{document}',r'\usepackage{xurl,seqsplit}\begin{document}')
    tex=re.sub(r'\b[a-f0-9]{64}\b',lambda m:r'\seqsplit{'+m[0]+'}',tex)
    (P/'LPcode_MSTF_Supplementary.tex').write_text(tex,encoding='utf-8')
    # Public-facing graphical TOC text is a draft, not a submitted artefact.
    g='# Graphical table-of-contents draft\n\n'+(P/'paper_revised.md').read_text(encoding='utf-8').splitlines()[0][2:]+'\n\nJin Huang*, Qiao Li, Qisen Gao\n\n![Paired transition representation](revised_figures/figure1.png)\n\nGiven a human source and a candidate, MSTF combines endpoint style with explicit changes. Leakage-safe evaluation isolates both origins and exact code content. Fixed-classifier ablations support richer endpoints and signed differences, whereas the relative block adds little. Generator, deterministic-transformation and language tests define the evaluated generalisation boundary.\n'
    (P/'GRAPHICAL_TOC_DRAFT.md').write_text(g,encoding='utf-8')
if __name__=='__main__':run()
