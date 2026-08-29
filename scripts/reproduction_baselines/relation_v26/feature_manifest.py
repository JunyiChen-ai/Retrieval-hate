import hashlib,json,math
from pathlib import Path
from artifacts import sha
from core import DESIGN_SHA
DIMS={'visual':512,'audio':128,'text':768};MODELS={'clip':'openai/clip-vit-base-patch16:get_image_features:no_l2','vggish':'torchvggish==0.2:postprocess_false','text_en':'bert-base-uncased:last_hidden_cls','text_zh':'bert-base-chinese:last_hidden_cls'}
def verify(path):
 m=json.load(open(path));req={'schema','design_sha256','corpus','split','ids','records','models','normalization','G_binding','root_sha256','labels_or_gt_read'}
 if set(m)!=req or m['schema']!='v26_features_v1' or m['design_sha256']!=DESIGN_SHA or m['labels_or_gt_read'] is not False or m['models']!=MODELS:raise RuntimeError('feature manifest')
 if len(m['ids'])!=len(set(m['ids'])) or set(m['records'])!=set(m['ids']):raise RuntimeError('coverage')
 if set(m['normalization'])!={'path','sha256','fit_split'} or sha(m['normalization']['path'])!=m['normalization']['sha256'] or m['normalization']['fit_split']!='train':raise RuntimeError('normalization')
 if set(m['G_binding'])!={'domain','source_sha256','raw_root_sha256','raw_seal_sha256','split_manifest_sha256','finalizer_sha256'} or m['G_binding']['domain']!='signed_logit':raise RuntimeError('G binding')
 for v,p in m['records'].items():
  r=json.load(open(p));T=math.ceil(r['duration']);rk={'corpus','split','opaque_id','duration','G','G_domain','seconds','source_hashes'}
  if set(r)!=rk or r['corpus']!='thvl' or r['split']!=m['split'] or r['opaque_id']!=v or r['G_domain']!='signed_logit' or not math.isfinite(r['G']) or len(r['seconds'])!=T:raise RuntimeError('record schema')
  for i,z in enumerate(r['seconds']):
   if z['second']!=i or set(z)!= {'second','visual','audio','text','availability'} or len(z['availability'])!=3 or any(x not in (0,1) for x in z['availability']) or not any(z['availability']):raise RuntimeError('second schema')
   for j,(f,d) in enumerate((('visual',512),('audio',128),('text',768))):
    if (z['availability'][j] and (len(z[f])!=d or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in z[f]))) or (not z['availability'][j] and z[f]!=[]):raise RuntimeError('modality schema')
 root=hashlib.sha256((m['G_binding']['raw_root_sha256']+m['G_binding']['raw_seal_sha256']+m['G_binding']['finalizer_sha256']+m['G_binding']['split_manifest_sha256']+m['normalization']['sha256']+''.join(v+'\t'+sha(m['records'][v])+'\n' for v in sorted(m['ids']))).encode()).hexdigest()
 if root!=m['root_sha256']:raise RuntimeError('root')
 return m
