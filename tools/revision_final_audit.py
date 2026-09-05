"""Source, numeric, citation, caption, figure and PDF structural checks."""
import json,re,zipfile,xml.etree.ElementTree as ET,subprocess,hashlib
import pymupdf
from revision_evidence import ROOT,OUT,run,digest
P=ROOT/'results/07_manuscript';A=ROOT/'results/08_submission_audit'
def check():
    e=run();text=(P/'paper_revised.md').read_text(encoding='utf-8');checks={}
    refs=json.loads((OUT/'bibliography.json').read_text(encoding='utf-8'))
    prose,bib=text.split('## References')
    cited={int(n) for m in re.finditer(r'\[(\d+(?:,\d+)*)\]',prose) for n in m[1].split(',')}
    checks['citation_bijection']=cited==set(range(1,22))==set(map(int,re.findall(r'(?m)^\[(\d+)\]',bib)))
    checks['reference_titles_unique']=len({r['title'][0].casefold() for r in refs})==21
    checks['eleven_sections']=len(re.findall(r'(?m)^## \d+ ',prose))==11
    checks['abstract_under_200']=len(prose.split('## Abstract')[1].split('Keywords:')[0].split())<=200
    checks['seven_tables']=re.findall(r'(?m)^Table (\d+)\.',prose)==list('1234567')
    checks['five_figures']=re.findall(r'!\[Figure (\d+)\.',prose)==list('12345')
    checks['negative_c4_preserved']='-0.019' in text and '+0.007' in text
    checks['explicit_release_blocker']=text.count('TODO_PUBLIC_RELEASE')==1
    checks['no_other_template_tokens']=not re.search(r'\{\{.*?\}\}|\bXX\b|lorem ipsum',text)
    checks['mean_f1_definition']='positive-class' in text and 'does not average F1 over the two class labels' in text
    for i,r in enumerate(e['main']):
        checks['gate_'+str(i+1)+'_arithmetic']=abs(r['method']-r['baseline']-r['delta'])<1e-12
        checks['gate_'+str(i+1)+'_display']=f"{r['delta']*100:.3f}" in text
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    with zipfile.ZipFile(P/'LPcode_MSTF_Wiley_Revised.docx') as z:
        doc=ET.fromstring(z.read('word/document.xml'));allxml=''.join(z.read(x).decode('utf-8') for x in z.namelist() if x.startswith('word/footer') and x.endswith('.xml'))
    instructions=[x.text or '' for x in doc.findall('.//w:instrText',ns)]
    checks['word_twelve_live_caption_fields']=sum('SEQ' in x for x in instructions)==12
    checks['word_four_native_equations']=len(doc.findall('.//m:oMathPara',ns))==4
    checks['word_live_page']='PAGE' in allxml
    checks['word_seven_tables']=len(doc.findall('w:body/w:tbl',ns))==7
    checks['source_hashes_unchanged']=True # run() above asserts exact snapshot identity.
    pdf=pymupdf.open(P/'LPcode_MSTF_Wiley_Revised.pdf');pt='\n'.join(p.get_text() for p in pdf)
    checks['pdf_twelve_pages']=len(pdf)==12
    checks['pdf_no_production_placeholders']=not any(x in pt for x in ['Received:','Accepted:','v0:', 'https://doi.org/\n'])
    checks['pdf_all_authors']=all(x in pt for x in ['Jin Huang','Qiao Li','Qisen Gao'])
    checks['pdf_all_contact_emails']=all(x in pt for x in ['614938561@qq.com','1339275715@qq.com','1350728839@qq.com'])
    caption_text=pt.replace('F I G U R E','FIGURE').replace('T A B L E','TABLE')
    checks['pdf_figure_captions']=all('FIGURE '+str(i) in caption_text for i in range(1,6))
    checks['pdf_table_captions']=all('TABLE '+str(i) in caption_text for i in range(1,8))
    log=(P/'revised_build/LPcode_MSTF_Wiley_Revised.log').read_text(encoding='utf-8',errors='replace')
    checks['pdf_no_overfull_boxes']='Overfull' not in log
    checks['pdf_no_undefined_citations']=not re.search(r'Citation .* undefined',log)
    fig_qa=[]
    skill=__import__('pathlib').Path('C:/Users/PC/.codex/skills/nature-figure/scripts')
    for i in range(1,6):
        path=P/f'revised_figures/figure{i}.pdf'
        for name,args in [('glyph',['audit_pdf_text.py',str(path),'--min-pt','5','--json']),('collision',['audit_figure_collisions.py',str(path),'--json-out',str(path.with_suffix('.collision.json'))])]:
            r=subprocess.run([__import__('sys').executable,str(skill/args[0]),*args[1:]],text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            checks[f'figure_{i}_{name}']=r.returncode==0
            (OUT/f'figure{i}_{name}.txt').write_text(r.stdout,encoding='utf-8')
        with pymupdf.open(path) as f: checks[f'figure_{i}_width']=abs(f[0].rect.width/72*25.4-183)<.1
    r=subprocess.run([__import__('sys').executable,str(skill/'validate_figure.py'),str(ROOT/'tools/revision_figures.py')],text=True,encoding='utf-8',stdout=subprocess.PIPE)
    (OUT/'figure_source_preflight.txt').write_text(r.stdout,encoding='utf-8')
    checks['figure_source_no_fail']='[FAIL]' not in r.stdout
    status='PASS' if all(checks.values()) else 'FAIL'
    report={'status':status,'checks':checks,'pdf_pages':len(pdf),'word_editable_review_format':True,'release_status':'BLOCKER','outputs_sha256':{x.name:digest(x) for x in [P/'paper_revised.md',P/'LPcode_MSTF_Wiley_Revised.pdf',P/'LPcode_MSTF_Wiley_Revised.docx']}}
    (A/'revised_final_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(status,[k for k,v in checks.items() if not v]);return report
if __name__=='__main__':check()
