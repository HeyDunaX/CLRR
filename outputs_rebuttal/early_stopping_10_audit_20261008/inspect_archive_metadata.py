import io,json,zipfile,concurrent.futures
from pathlib import Path
from urllib.parse import quote
import requests
out=Path('outputs_rebuttal/early_stopping_10_audit_20261008');v=json.loads((out/'public_inventory.json').read_text());rev=v['sha']
class RangeReader(io.RawIOBase):
 def __init__(self,url,size): self.url=url;self.size=size;self.pos=0;self.blocks={};self.bytes=0;self.session=requests.Session()
 def readable(self): return True
 def seekable(self): return True
 def tell(self): return self.pos
 def seek(self,n,w=0):
  self.pos=n if w==0 else self.pos+n if w==1 else self.size+n
  return self.pos
 def read(self,n=-1):
  if n<0: n=self.size-self.pos
  if n>1048576: raise ValueError('Refuse large member read')
  end=min(self.size,self.pos+n);parts=[]
  while self.pos<end:
   b=self.pos//262144;start=b*262144;last=min(self.size-1,start+262143)
   if b not in self.blocks:
    with self.session.get(self.url,headers={'Range':f'bytes={start}-{last}'},stream=True,timeout=25) as r:
     r.raise_for_status()
     if r.status_code!=206 or r.headers.get('Content-Range','').split('/')[0]!=f'bytes {start}-{last}': raise ValueError('Range mismatch; refuse full download')
     data=r.raw.read(last-start+2)
     if len(data)!=last-start+1: raise ValueError('Range size mismatch')
    self.blocks[b]=data;self.bytes+=len(data)
   count=min(end-self.pos,len(self.blocks[b])-(self.pos-start));parts.append(self.blocks[b][self.pos-start:self.pos-start+count]);self.pos+=count
  return b''.join(parts)
def inspect(e):
 p=e['rfilename'];parts=p.split('/');a={'path':p,'backbone':parts[2],'method':parts[3],'metadata':{}}
 try:
  reader=RangeReader(f'https://huggingface.co/FiveC/amis-rewire-checkpoints/resolve/{rev}/{quote(p)}',e['size'])
  with zipfile.ZipFile(reader) as z:
   a['members']=[{'name':x.filename,'bytes':x.file_size} for x in z.infolist()]
   for x in z.infolist():
    if Path(x.filename).name in ('trainer_state.json','metrics.json','run_manifest.json','training_receipt.json','COMPLETE.json') and x.file_size<1048576:
     a['metadata'][x.filename]=json.loads(z.read(x))
  a['status']='INSPECTED';a['bytes_transferred']=reader.bytes
 except Exception as err: a['status']='UNINSPECTED';a['error']=type(err).__name__+': '+str(err)
 print(a['backbone'],a['method'],a['status'],[(k,m.get('epoch'),m.get('completed_epochs'),m.get('global_step')) for k,m in a['metadata'].items()],flush=True)
 return a
c=[e for e in v['siblings'] if e['rfilename'].startswith('checkpoints/amis_mandarin/') and e['rfilename'].endswith('/best.zip') and e['rfilename'].split('/')[3] in ('baseline','middle-alignment','strong-lora','lora','bitfit') and ('original-' in e['rfilename'] or '/strong-lora-comparison/' in e['rfilename'])]
print('ARCHIVES',len(c),flush=True);records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for a in pool.map(inspect,c):
  records.append(a);(out/'archive_metadata.json').write_text(json.dumps({'revision':rev,'archives':records},indent=2),encoding='utf-8')
print('TOTAL_METADATA_BYTES',sum(r.get('bytes_transferred',0) for r in records),flush=True)
