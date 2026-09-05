"""Polish the Pandoc Word assembly with OfficeCLI styles and live fields."""
from pathlib import Path
import json,subprocess,zipfile,re,xml.etree.ElementTree as ET
from revision_evidence import ROOT,OUT
P=ROOT/'results/07_manuscript/LPcode_MSTF_Wiley_Revised.docx'
def cli(args):
    r=subprocess.run(['officecli',*args],encoding='utf-8',text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(r.stdout)
    return r.stdout
def run():
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(P) as z: root=ET.fromstring(z.read('word/document.xml'))
    ps=root.find('w:body',ns).findall('w:p',ns)
    commands=[dict(command='set',path='/',props={'defaultFont':'Times New Roman','pageWidth':'11906','pageHeight':'16838','marginTop':'1134','marginBottom':'1134','marginLeft':'1134','marginRight':'1134'})]
    commands.append(dict(command='add',parent='/styles',type='style',props={'id':'TableNormal','name':'Normal Table','type':'table'}))
    for style,size in [('Normal','11pt'),('BodyText','11pt'),('FirstParagraph','11pt'),('Heading1','14pt'),('Heading2','12pt'),('Heading3','12pt'),('Title','20pt'),('Caption','9pt'),('ImageCaption','9pt')]:
        commands.append(dict(command='set',path='/styles/'+style,props={'font':'Times New Roman','size':size,'color':'000000','lineSpacing':'1.15x','spaceAfter':'6pt'}))
    commands.append(dict(command='set',path='/body/p[1]',props={'style':'Title','align':'center'}))
    i=0
    for p in ps:
        if p.find('{http://schemas.openxmlformats.org/officeDocument/2006/math}oMathPara') is not None:continue
        i+=1
        text=''.join(p.itertext());text=''.join(p.findall('.//w:t',ns)[j].text or '' for j in range(len(p.findall('.//w:t',ns))))
        path=f'/body/p[{i}]'
        if 2<=i<=6:commands.append(dict(command='set',path=path,props={'align':'center','size':'10pt','spaceAfter':'4pt'}))
        cap=re.match(r'^(Figure|Table) (\d+)\. (.*)',text)
        if cap:
            kind,num,caption=cap.groups()
            commands.extend([dict(command='set',path=path,props={'text':kind+' ','style':'Caption','keepWithNext':'true' if kind=='Table' else 'false','size':'9pt'}),dict(command='add',parent=path,type='field',props={'fieldType':'seq','identifier':kind}),dict(command='add',parent=path,type='run',props={'text':'. '+caption,'size':'9pt'})])
        if re.match(r'^\[\d+\] ',text):commands.append(dict(command='set',path=path,props={'indent':'360','hangingIndent':'360','size':'9pt','spaceAfter':'5pt'}))
        if text in ['Funding','Conflict of interest','Data availability','Code availability']:commands.append(dict(command='set',path=path,props={'style':'Heading2'}))
    commands+= [dict(command='add',parent='/',type='header',props={'text':'MSTF · Leakage-safe paired provenance','align':'right','size':'9pt'}),dict(command='add',parent='/',type='footer',props={'field':'page','align':'center','size':'9pt'}),dict(command='set',path='/',props={'recalcFields':'seq'}),dict(command='set',path='/settings',props={'updateFields':'true'})]
    batch=OUT/'word_format_commands.json';batch.write_text(json.dumps(commands,ensure_ascii=False,indent=2),encoding='utf-8')
    result=cli(['batch',str(P),'--input',str(batch),'--stop-on-error'])
    (OUT/'word_format.log').write_text(result,encoding='utf-8')
    cli(['close',str(P)])
    # Pandoc leaves footnotePr after pgMar; OfficeCLI's root margin setter retains
    # that order. Repair only this schema ordering, preserving every element.
    with zipfile.ZipFile(P) as z: updated=ET.fromstring(z.read('word/document.xml'))
    section=updated.find('w:body/w:sectPr',ns)
    for name in ['endnotePr','footnotePr']:
        el=section.find('w:'+name,ns)
        if el is not None:
            section.remove(el)
            idx=sum(x.tag.endswith('Reference') for x in section)
            section.insert(idx,el)
    cli(['raw-set',str(P),'/document','--xpath','/w:document/w:body/w:sectPr','--action','replace','--xml',ET.tostring(section,encoding='unicode')])
    cli(['close',str(P)])
    for check,args in [('word_validate',['validate']),('word_outline',['view','outline']),('word_issues',['view','issues']),('word_html',['view','html'])]:
        actual=[args[0],str(P),*args[1:]]
        result=cli(actual);(OUT/(check+'.txt')).write_text(result,encoding='utf-8');print(check,result[:160])
if __name__=='__main__':run()
