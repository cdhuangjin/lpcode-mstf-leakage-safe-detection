"""Build a separate Wiley author manuscript, retaining the supplied USG class."""
from pathlib import Path
import subprocess,re,json,sys
from revision_evidence import ROOT,OUT

PAPER=ROOT/'results/07_manuscript'
TEX=ROOT/'WileyDesign/Optimal-Design-layout/LPcode_MSTF_Wiley_Revised.tex'
def pandoc(text):
    return subprocess.run(['pandoc','-f','markdown+raw_tex','-t','latex','--wrap=none'],input=text,text=True,encoding='utf-8',stdout=subprocess.PIPE,check=True).stdout.strip()
def build():
    text=(PAPER/'paper_revised.md').read_text(encoding='utf-8')
    title=text.splitlines()[0][2:];abstract=text.split('## Abstract\n\n')[1].split('\n\nKeywords:')[0]
    body=text.split('## 1 Introduction')[1].split('## References')[0];body='## 1 Introduction'+body
    # Resolve bracketed citations through the verified bibliography.
    body=re.sub(r'\[(\d+(?:,\d+)*)\]',lambda m:r'\cite{'+','.join('r'+n for n in m[1].split(','))+'}',body)
    figures=[]
    def fig(m):
        n=m[1];caption=m[2]
        tex='\n\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=\\linewidth]{../../results/07_manuscript/revised_figures/figure'+n+'.pdf}\n\\caption{'+pandoc(caption)+'}\\label{fig:'+n+'}\n\\end{figure}\n'
        figures.append(tex);return '\n```{=latex}\n'+tex+'```\n'
    body=re.sub(r'!\[Figure (\d+)\. (.*?)\]\(revised_figures/figure\d+\.png\)',fig,body,flags=re.S)
    tables=[]
    def tbl(m):
        n,cap,raw=m[1],m[2],m[3];rows=[line.strip().strip('|').split('|') for line in raw.strip().splitlines()];rows=[rows[0]]+rows[2:];cols=len(rows[0])
        spec=('p{.15\\linewidth}X p{.30\\linewidth}' if n=='1' else ('p{.36\\linewidth}'+'X'*(cols-1) if n=='2' else 'X'*cols))
        lines=['\\begin{table}[!htbp]','\\caption{'+pandoc(cap)+'}\\label{tab:'+n+'}','\\centering\\small','\\setlength{\\tabcolsep}{4pt}','\\renewcommand{\\arraystretch}{1.18}','\\begin{tabularx}{\\linewidth}{'+spec+'}','\\toprule']
        for i,row in enumerate(rows):
            cells=[]
            for cell in row:
                v=pandoc(cell.strip())
                # Feature identifiers remain exact, but permit a line break after each underscore.
                v=v.replace(r'\_',r'\_\allowbreak{}')
                cells.append(v)
            lines.append(' & '.join(cells)+r' \\')
            if i==0:lines.append(r'\midrule')
        lines+=['\\bottomrule','\\end{tabularx}','\\end{table}']
        tables.append('\n'.join(lines));return '\n```{=latex}\n'+'\n'.join(lines)+'\n```\n'
    body=re.sub(r'(?m)^Table (\d+)\. ([^\n]*)\n\n((?:\|[^\n]*\n)+)',tbl,body)
    # Strip hand-written section numbers; USG supplies numbering.
    body=re.sub(r'(?m)^## \d+ ', '# ',body);body=re.sub(r'(?m)^### \d+\.\d+ ', '## ',body)
    body=body.replace('## Declarations','# Declarations').replace('### Funding','## Funding').replace('### Conflict of interest','## Conflict of interest').replace('### Data availability','## Data availability').replace('### Code availability','## Code availability')
    texbody=pandoc(body)
    refs=json.loads((OUT/'bibliography.json').read_text(encoding='utf-8'))
    bibliography=['\\begin{thebibliography}{99}']
    for r in refs:
        a='; '.join(r['authors']);entry=f"{a}. {r['title'][0]}. {r['cited_venue']}; {r['cited_year']}."
        bibliography.append('\\bibitem{r'+str(r['number'])+'} '+pandoc(entry)+r' \url{'+r['primary_url']+'}')
    bibliography.append('\\end{thebibliography}')
    original=(TEX.parent/'LPcode_MSTF_Wiley.tex').read_text(encoding='utf-8')
    pre=original.split('\\abstract[ABSTRACT]')[0]
    pre=pre.replace(r'\volume{0}',r'\volume{}').replace(r'\copyyear{2026}',r'\copyyear{}').replace('TECHNOLOGICAL ARTICLE','EMPIRICAL STUDY')
    pre=pre.replace(r'\documentclass[ASNA]',r'\documentclass[ASNA,twocolumn]')
    texbody=texbody.replace(r'\begin{figure}[!htbp]',r'\begin{figure*}[!t]').replace(r'\end{figure}',r'\end{figure*}')
    texbody=texbody.replace(r'\begin{table}[!htbp]',r'\begin{table*}[!t]').replace(r'\end{table}',r'\end{table*}')
    texbody=texbody.replace(r'\section{Declarations}',r'\section*{Declarations}')
    pre=re.sub(r'\\title\{.*?\}',lambda m:r'\title{'+pandoc(title)+'}',pre)
    pre=re.sub(r'\\keywords\{.*?\}',lambda m:r'\keywords{code paraphrasing | paired provenance | code stylometry | leakage-safe evaluation | distribution shift | interpretable representation}',pre)
    pre=pre.replace(r'\corres{Jin Huang (\email{614938561@qq.com})}',r'\corres{Jin Huang (\email{614938561@qq.com}). Co-author emails: Qiao Li (\email{1339275715@qq.com}); Qisen Gao (\email{1350728839@qq.com}).}')
    pre=pre.replace(r'\usepackage{anyfontsize}','')
    pre=pre.replace(r'\usepackage{amsmath}',r'\usepackage{amsmath,amssymb,tabularx,array,xurl}'+ '\n'+r'\newcommand{\tightlist}{}')
    pre+=r'''
% Author-version only: omit production dates, DOI, volume and fake page range.
\makeatletter
\gdef\@dummy@received{}\gdef\@dummy@revised{}\gdef\@dummy@accepted{}
\gdef\@history@dates{}\gdef\@DOI@text{}
\def\oddfoot@titlepage@info{\hbox to\textwidth{\footnotesize Author manuscript\hfill\thepage}}
\def\evenfoot@titlepage@info{\oddfoot@titlepage@info}
\makeatother
\setlength{\emergencystretch}{2em}
'''
    pre+='\\abstract[ABSTRACT]{'+pandoc(abstract)+'}\n\\begin{document}\n\\maketitle\n\\pagestyle{plain}\n\\thispagestyle{plain}\n\\raggedbottom\n'
    TEX.write_text(pre+texbody+'\n'+ '\n'.join(bibliography)+'\n\\end{document}\n',encoding='utf-8')
    if '--tex-only' in sys.argv:return
    # Pandoc preserves editable equations, actual tables and hierarchy in Word.
    docmd=text.replace('## Abstract','# Abstract').replace('## References','# References').replace('## Declarations','# Declarations')
    docmd=re.sub(r'(?m)^## (\d+ )',r'# \1',docmd);docmd=re.sub(r'(?m)^### (\d+\.\d+ )',r'## \1',docmd)
    docmd=re.sub(r'\\tag\{(\d+)\}',lambda m:r'\quad ('+m[1]+')',docmd)
    docmd=docmd.replace('](revised_figures/',']('+str(PAPER/'revised_figures').replace('\\','/')+'/')
    subprocess.run(['pandoc','-f','markdown','-t','docx','-o',str(PAPER/'LPcode_MSTF_Wiley_Revised.docx')],input=docmd,text=True,encoding='utf-8',check=True)
    print(TEX);print('Tables:',len(tables),'Figures:',len(figures))
if __name__=='__main__':build()
