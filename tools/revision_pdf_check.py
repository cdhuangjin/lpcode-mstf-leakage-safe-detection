"""Render the revised PDF and record page-level geometry without changing it."""
from pathlib import Path
import json,shutil
import pymupdf
from PIL import Image,ImageOps,ImageDraw
from revision_evidence import ROOT,OUT
PAPER=ROOT/'results/07_manuscript'
def run():
    source=PAPER/'revised_build/LPcode_MSTF_Wiley_Revised.pdf'
    target=PAPER/'LPcode_MSTF_Wiley_Revised.pdf';shutil.copy2(source,target)
    doc=pymupdf.open(target);pages=[];thumbs=[]
    preview=PAPER/'revised_preview';preview.mkdir(exist_ok=True)
    for i,p in enumerate(doc):
        text=p.get_text();spans=[s for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        outside=[s['text'] for s in spans if s['bbox'][0]<0 or s['bbox'][1]<0 or s['bbox'][2]>p.rect.width+.5 or s['bbox'][3]>p.rect.height+.5]
        p.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(preview/f'page_{i+1:02}.png')
        pages.append({'page':i+1,'words':len(text.split()),'outside_page':outside,'text':text})
        im=Image.open(preview/f'page_{i+1:02}.png').convert('RGB');im.thumbnail((310,435));tile=Image.new('RGB',(330,465),'#e6ebee');tile.paste(im,((330-im.width)//2,10));ImageDraw.Draw(tile).text((15,447),str(i+1),fill='black');thumbs.append(tile)
    grid=Image.new('RGB',(330*3,465*((len(thumbs)+2)//3)),'white')
    for i,im in enumerate(thumbs):grid.paste(im,((i%3)*330,(i//3)*465))
    grid.save(preview/'contact_sheet.png')
    (OUT/'pdf_pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Pages',len(doc),'outside',sum(len(p['outside_page']) for p in pages));print([(p['page'],p['words']) for p in pages])
if __name__=='__main__':run()
