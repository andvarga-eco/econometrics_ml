import json, sys, os
sys.dont_write_bytecode = True
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.')
from bank import TOPICS, Q
from collections import Counter
ids=[q['id'] for q in Q]; assert len(ids)==len(set(ids))
tids={t['id'] for t in TOPICS}
for q in Q:
    assert q['t'] in tids and q['lvl'] in (1,2,3)
    if q['type']=='mc': assert 0<=q['ans']<len(q['opts'])
print(len(Q), Counter((q['t'],q['lvl']) for q in Q))
import re
fix=lambda x: re.sub(r'\$[^$]*\$', lambda m: m.group(0).replace('<',r'\lt ').replace('>',r'\gt '), x) if isinstance(x,str) else ([fix(i) for i in x] if isinstance(x,list) else x)
for q in Q:
    for k in ('q','exp','opts'):
        if k in q: q[k]=fix(q[k])
clean=[{k:v for k,v in q.items() if v is not None} for q in Q]
h=open('template.html').read()
h=h.replace('__TOPICS__',json.dumps(TOPICS,ensure_ascii=False)).replace('__BANK__',json.dumps(clean,ensure_ascii=False).replace('</','<\\/'))
open('examen_adaptativo.html','w').write(h)
