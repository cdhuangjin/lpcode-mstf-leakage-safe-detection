"""Refresh read-only review renders and supplementary deliverable."""
import json,shutil
import pymupdf
from PIL import Image,ImageDraw
from revision_evidence import ROOT,OUT
from revision_word import cli,P
base=P.parent;preview=base/'revised_preview'
shutil.copy2(base/'revised_build/LPcode_MSTF_Supplementary.pdf',base/'LPcode_MSTF_Supplementary.pdf')
report={}
for name,path in [('supplement',base/'LPcode_MSTF_Supplementary.pdf'),('word',preview/'word_review.pdf')]:
    doc=pymupdf.open(path);tiles=[];outside=[]
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5));pix.save(preview/f'{name}_{i+1:02}.png')
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((310,435))
        tile=Image.new('RGB',(330,465),'#e6ebee');tile.paste(im,((330-im.width)//2,10));ImageDraw.Draw(tile).text((15,447),str(i+1),fill='black');tiles.append(tile)
        for b in page.get_text('blocks'):
            if b[0]<0 or b[1]<0 or b[2]>page.rect.width+.5 or b[3]>page.rect.height+.5:outside.append([i+1,b[:4]])
    grid=Image.new('RGB',(330*3,465*((len(tiles)+2)//3)),'white')
    for i,tile in enumerate(tiles):grid.paste(tile,((i%3)*330,(i//3)*465))
    grid.save(preview/f'{name}_contact_sheet.png');report[name]={'pages':len(doc),'outside':outside}
for name,args in [('word_validate',['validate']),('word_issues',['view','issues']),('word_html',['view','html'])]:
    (OUT/(name+'.txt')).write_text(cli([args[0],str(P),*args[1:]]),encoding='utf-8')
(OUT/'extra_previews.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)
