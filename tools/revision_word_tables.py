"""Keep short manuscript tables together and preserve native caption fields."""
import json,zipfile,xml.etree.ElementTree as ET
from revision_word import cli,P,OUT
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(P) as z:root=ET.fromstring(z.read('word/document.xml'))
commands=[]
for ti,t in enumerate(root.findall('w:body/w:tbl',ns),1):
    rows=t.findall('w:tr',ns)
    for ri,row in enumerate(rows,1):
        commands.append(dict(command='set',path=f'/body/tbl[{ti}]/tr[{ri}]',props={'cantSplit':'true'}))
        for ci,cell in enumerate(row.findall('w:tc',ns),1):
            for pi,p in enumerate(cell.findall('w:p',ns),1):
                commands.append(dict(command='set',path=f'/body/tbl[{ti}]/tr[{ri}]/tc[{ci}]/p[{pi}]',props={'keepWithNext':'true' if ri<len(rows) else 'false','size':'9pt','spaceAfter':'2pt','lineSpacing':'1.15x'}))
path=OUT/'word_table_commands.json';path.write_text(json.dumps(commands),encoding='utf-8')
print(cli(['batch',str(P),'--input',str(path),'--stop-on-error'])[-150:]);cli(['close',str(P)])
