import json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
sys.path.insert(0,str(Path(__file__).parent))
import video_selection as vs
from artifacts import atomic,sha
from core import DESIGN_SHA

class T(unittest.TestCase):
 def test_signed_selection_epoch_and_prediction_failclosed(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);ids=[f'v{i:02d}' for i in range(32)];labels=d/'labels';atomic(labels,{'schema_version':'v24_video_labels_v1','corpus':'thvl','split':'val','label_semantics':'any_target_video_level','records':[{'video_id':v,'any_target_label':i%2} for i,v in enumerate(ids)]})
   files={a:{} for a in ('real','permuted','negative_mean')}
   for a in files:
    for ep in range(9):
     p=d/f'{a}{ep}';scores={v:{'video_logit':float((i%2)*(ep==2)+(i%3)/100)} for i,v in enumerate(ids)};atomic(p,{'records':scores});files[a][str(ep)]={'path':str(p.resolve()),'sha256':sha(p)}
   pm={'seed':234,'files':files};mp=d/'pm';atomic(mp,pm);key=Path('/home/jehc223/Hate-follow-up/results/steward_private/thvl_bench/v26_steward_key_frozen/ed25519_private.key');out=d/'sel'
   with patch.object(vs,'verify_predictions',return_value=(pm,ids)):
    x=vs.run([str(mp)],str(labels),str(key),str(out));self.assertEqual(x['selected_epoch'],2)
    y=json.load(open(out));y['selected_epoch']=3;att=Ed25519PrivateKey.generate();unsigned={k:v for k,v in y.items() if k!='signature_hex'};y['signature_hex']=att.sign(vs.canon(unsigned)).hex();atomic(d/'forged',y);self.assertRaises(Exception,vs.verify,d/'forged',[str(mp)],str(labels))
    mp.write_bytes(b'fake');self.assertRaises(Exception,vs.verify,out,[str(mp)],str(labels))
if __name__=='__main__':unittest.main()
