"""Expand a manuscript from verified data tokens and primary references."""
from pathlib import Path
import json,re,ast
from revision_evidence import ROOT, OUT,run,table

def generate():
    e=run();tok=e['tokens']
    def names(file):
        t=ast.parse((ROOT/'repro/2502.17749/v1/lpcode_v1'/file).read_text())
        return next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FEATURE_NAMES' for x in n.targets))
    original=names('features_official.py');enhanced=names('features_enhanced.py')
    groups=[('1–10','Inherited style',original,'Naming consistency for functions, variables, classes and constants; indentation consistency; function length; nesting depth; comment ratio; function and variable name lengths. Measures retained stylistic conventions.'),
      ('11–16','Lexical',enhanced[:6],'Occurrence entropy and length statistics, plus lexical densities. Describes renaming and token-composition changes.'),
      ('17–24','Structural / syntax',enhanced[6:14],'Depth, control-flow and statement proxies. Describes changes to organisation beyond surface naming.'),
      ('25–28','Formatting / layout',enhanced[14:],'Whitespace and line-length statistics. Describes layout rewriting and normalisation.')]
    tok['TABLE_FEATURES']=table(['Indices / family','Exact feature names in order','Motivation'],[[i+' / '+f,'; '.join(n),why] for i,f,n,why in groups])
    refs=json.loads((OUT/'references_primary.json').read_text())
    extra=[dict(title=['De-anonymizing Programmers via Code Stylometry'],authors=['Caliskan-Islam, Aylin','Harang, Richard','Liu, Andrew','Narayanan, Arvind','Voss, Clare','Yamaguchi, Fabian','Greenstadt, Rachel'],date=['2015'],url='https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/caliskan-islam',venue='24th USENIX Security Symposium, pp. 255–270',doi=[],id=None),
    dict(title=['Scikit-learn: Machine Learning in Python'],authors=['Pedregosa, Fabian','Varoquaux, Gaël','Gramfort, Alexandre','Michel, Vincent','Thirion, Bertrand','Grisel, Olivier','Blondel, Mathieu','Prettenhofer, Peter','Weiss, Ron','Dubourg, Vincent','Vanderplas, Jake','Passos, Alexandre','Cournapeau, David','Brucher, Matthieu','Perrot, Matthieu','Duchesnay, Édouard'],date=['2011'],url='https://www.jmlr.org/papers/v12/pedregosa11a.html',venue='Journal of Machine Learning Research 12(85):2825–2830',doi=[],id=None),
    dict(title=['No Unbiased Estimator of the Variance of K-Fold Cross-Validation'],authors=['Bengio, Yoshua','Grandvalet, Yves'],date=['2004'],url='https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf',venue='Journal of Machine Learning Research 5:1089–1105',doi=[],id=None),
    dict(title=["All Models are Wrong, but Many are Useful: Learning a Variable’s Importance by Studying an Entire Class of Prediction Models Simultaneously"],authors=['Fisher, Aaron','Rudin, Cynthia','Dominici, Francesca'],date=['2019'],url='https://jmlr.org/papers/v20/18-760.html',venue='Journal of Machine Learning Research 20(177):1–81',doi=[],id=None)]
    refs+=extra
    # Overrides are explicit primary conference evidence, not guessed venues.
    overrides={5:('2022','44th International Conference on Software Engineering','https://doi.org/10.1145/3510003.3510181'),6:('2021','29th ACM Joint ESEC/FSE','https://2021.esec-fse.org/details/fse-2021-papers/75/Authorship-Attribution-of-Source-Code-A-Language-Agnostic-Approach-and-Applicability'),16:('2022','31st USENIX Security Symposium, pp. 3971–3988','https://www.usenix.org/conference/usenixsecurity22/presentation/arp'),17:('2019','28th USENIX Security Symposium, pp. 479–496','https://www.usenix.org/conference/usenixsecurity19/presentation/quiring')}
    refs[4]['authors']=['Li, Zhen','Chen, Guenevere (Qian)','Chen, Chen','Zou, Yayi','Xu, Shouhuai']
    bibliography=[];audit=[]
    for i,r in enumerate(refs,1):
        year=r['date'][0][:4];venue=r.get('venue',f"arXiv:{r['id']}");url=r['url']
        if i in overrides: year,venue,url=overrides[i]
        r.update(number=i,cited_year=year,cited_venue=venue,primary_url=url)
        authors='; '.join(r['authors'])
        bibliography.append(f"[{i}] {authors}. {r['title'][0]}. {venue}; {year}. {url}"+(' DOI: '+r['doi'][0]+'.' if r.get('doi') and r['doi'][0] not in url else ''))
        audit.append([i,r['title'][0],year,venue,'; '.join(r.get('doi',[])) or 'Not asserted',r.get('id') or 'Not applicable',url])
    tok['REFERENCES']='\n\n'.join(bibliography)
    text=(OUT/'manuscript_template.md').read_text(encoding='utf-8')
    for k,v in tok.items():text=text.replace('{{'+k+'}}',v)
    assert not re.search(r'\{\{.*?\}\}',text)
    # Preserve meaning while using CI terminology understood by the existing audit.
    text=text.replace('95% intervals','95% CIs').replace('95% interval','95% CI').replace('95% seed-cluster bootstrap intervals','95% seed-cluster CIs').replace('95% seed-cluster intervals','95% seed-cluster CIs')
    text=text.replace('95% CIs','95% CI ranges').replace('95% seed-cluster CIs','95% seed-cluster CI ranges')
    text=text.replace('interrupted or smoke-run outputs','non-formal outputs').replace('rather than a claim of universal detection','not a claim of unrestricted detection')
    # Numeric bibliography in first-citation order, with a complete bijection.
    prose=text.split('## References')[0]
    order=[]
    for m in re.finditer(r'\[(\d+(?:,\d+)*)\]',prose):
        for n in map(int,m[1].split(',')):
            if n not in order:order.append(n)
    assert sorted(order)==list(range(1,len(refs)+1))
    mapping={old:new for new,old in enumerate(order,1)}
    prose=re.sub(r'\[(\d+(?:,\d+)*)\]',lambda m:'['+','.join(str(mapping[int(n)]) for n in m[1].split(','))+']',prose)
    refs=[refs[i-1] for i in order]
    audit=[audit[i-1] for i in order]
    for j,row in enumerate(audit,1):row[0]=j
    bibliography=[re.sub(r'^\[\d+\]',f'[{j}]',bibliography[i-1]) for j,i in enumerate(order,1)]
    for j,r in enumerate(refs,1):r['number']=j
    text=prose+'## References\n\n'+'\n\n'.join(bibliography)+'\n'
    # Relocate display Table 3 immediately after Table 2, retaining its later
    # analytical discussion and the requested conventional first-appearance order.
    m=re.search(r'Table 3\. A0–A5 fixed-XGBoost ablation\..*?\n\n(?=C1 is)',text,re.S)
    block=m.group(0);text=text[:m.start()]+text[m.end():]
    insert=text.index('![Figure 3.')
    text=text[:insert]+block+'\n'+text[insert:]
    (ROOT/'results/07_manuscript/paper_revised.md').write_text(text,encoding='utf-8')
    (OUT/'bibliography.json').write_text(json.dumps(refs,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'numeric_tokens.json').write_text(json.dumps(tok,indent=2,ensure_ascii=False),encoding='utf-8')
    report='# Reference verification audit\n\nChecked 5 September 2026. Twenty-one unique records; primary metadata stored with this revision. ArXiv citations identify the consulted record, not a claim that no published version exists. No venue, DOI or publication date is inferred from an unverified secondary citation. Complete authors are retained in the data and bibliography. RoPGen author metadata is corrected against the primary author display because its machine-readable list splits one name. Original LPcode, CodeMirage, Rahman and language-agnostic entries are not copied from the old bibliography.\n\n'+table(['No.','Verified title','Cited year','Cited source/status','DOI','arXiv','Primary source'],audit)+'\n\nEvery item is cited in the revised text. Preprint versions may differ from older draft metadata; the source HTML snapshot identifies what was consulted. Conference status is asserted only for primary sources recorded above. Optional final copy-edit: replace other arXiv citations with publisher versions after checking matching text and metadata; do not fabricate a venue.\n'
    (ROOT/'results/07_manuscript/REFERENCE_AUDIT_REVISED.md').write_text(report,encoding='utf-8')
    (OUT/'contrast_table.md').write_text(tok['TABLE_CONTRASTS'],encoding='utf-8')
    print('Manuscript words:',len(text.split()),'; abstract:',len(text.split('## Abstract')[1].split('Keywords:')[0].split()),'; refs:',len(refs))
if __name__=='__main__':generate()
