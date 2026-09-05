"""Fetch primary-source bibliographic metadata; never infer a publication venue."""
from pathlib import Path
import concurrent.futures, json, urllib.request, re, html
from html.parser import HTMLParser
class Metadata(HTMLParser):
    def __init__(self): super().__init__(); self.meta={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='meta': self.meta.setdefault(a.get('name'),[]).append(a.get('content',''))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/07_manuscript/revised_sources'
IDS = ['2502.17749','2506.11059','2409.01382','2412.14611','2202.06043',
       '2001.11593','2301.11305','2310.05130','2303.11156','2002.08155',
       '2009.08366','2109.00859','2203.03850','1909.09436','1603.02754',
       '2010.09470','1905.12386']

def fetch(aid):
    url='https://arxiv.org/abs/'+aid
    data=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research bibliographic verification'}),timeout=45).read()
    page=data.decode('utf-8'); parser=Metadata(); parser.feed(page)
    def meta(name): return parser.meta.get(name,[])
    def content(pattern):
        m=re.search(pattern,page,re.S)
        return html.unescape(re.sub('<[^>]*>',' ',m.group(1))).strip() if m else None
    out={'id':aid,'url':url,'title':meta('citation_title'), 'authors':meta('citation_author'),
         'date':meta('citation_date'),'doi':meta('citation_doi'),'journal':meta('citation_journal_title'),
         'journal_ref':content(r'<td class="tablecell jref">(.*?)</td>'),
         'abstract':content(r'<blockquote class="abstract[^\"]*">(.*?)</blockquote>'),
         'verification':'arXiv primary metadata; journal status only where explicitly recorded', 'checked':'2026-09-05'}
    (OUT/(aid+'.html')).write_bytes(data)
    return out

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(fetch,IDS))
    (OUT/'references_primary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in rows: print(r['id'],r['title'],r['authors'],r['date'],r['journal_ref'],r['doi'])
