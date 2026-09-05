"""Small public figure geometry audit; not the private Nature Figure validator.

The historical function name is an API adapter. Output explicitly identifies
the narrower axes-geometry scope; publication collision reports are archived.
"""
from pathlib import Path
import json
import math

def require_matplotlib_panel_alignment(fig, *, json_out, overlay_svg, strict=True, **kwargs):
    boxes=[list(ax.get_position().bounds) for ax in fig.axes]
    errors=[i for i,(x,y,w,h) in enumerate(boxes) if not all(math.isfinite(v) for v in (x,y,w,h)) or min(x,y)<0 or min(w,h)<=0 or x+w>1.001 or y+h>1.001]
    result={'validator':'public-axes-geometry-v1','scope':'axes finite, positive and inside canvas; not text collision or equal-gutter certification','axes':boxes,'errors':errors,'status':'FAIL' if errors else 'PASS'}
    Path(json_out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    rects=''.join(f'<rect x="{x*1000}" y="{(1-y-h)*1000}" width="{w*1000}" height="{h*1000}" fill="none" stroke="blue"/>' for x,y,w,h in boxes)
    Path(overlay_svg).write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">'+rects+'</svg>',encoding='utf-8')
    if strict and errors:raise ValueError(result)
    return result
