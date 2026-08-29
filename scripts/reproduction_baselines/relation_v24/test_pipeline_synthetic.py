#!/usr/bin/env python3
import hashlib,hmac,json,subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class TestPipeline(unittest.TestCase):
 def test_train_select_infer_evaluate_chain(self):
  with tempfile.TemporaryDirectory() as z:
   d=Path(z);producer=d/'producer.py';producer.write_text('frozen producer\n');ps=sha(producer)
   def split(name,n,labels):
    ids=[f'{name}{i}' for i in range(n)];gs=d/f'{name}_global.jsonl';gs.write_text(''.join(json.dumps({'video_id':v,'global_causal_score':0.0})+'\n' for v in ids));gh=sha(gs)
    man=d/f'{name}_ids.json';man.write_text(json.dumps({'corpus':'toy','split':name,'ids':ids,'v23_global_source_sha256':gh,'producer_sha256':ps}))
    bag=d/f'{name}_bags.jsonl';rows=[]
    for i,v in enumerate(ids):
     local=[0.,8.,0.] if labels[i] else [0.,0.,0.];r={'corpus':'toy','split':name,'video_id':v,'global_causal_score':0.0,'families':{'text':[local],'multimodal':[local]},'source_hashes':{'text_scores_sha256':'a'*64,'multimodal_scores_sha256':'b'*64,'v23_global_source_sha256':gh}}
     if name!='test':r['video_label']=labels[i]
     rows.append(r)
    bag.write_text(''.join(json.dumps(r)+'\n' for r in rows));return ids,gs,man,bag
   labels20=[0,1]*10;_,tgs,tm,tb=split('train',20,labels20);vids,vgs,vm,vb=split('val',20,labels20);tids,xgs,xm,xb=split('test',4,[0,1,0,1]);run=d/'run';frozen=d/'frozen.json'
   subprocess.run([sys.executable,str(HERE/'train.py'),'--bags',str(tb),'--id-manifest',str(tm),'--producer',str(producer),'--v23-global-source',str(tgs),'--corpus','toy','--out-dir',str(run)],check=True)
   # The conclusion path requires the new val-join schema and an explicit no-retrain addendum.
   val_join_sha=sha(HERE/'steward_val_join.py');evidence_sha='e'*64
   vm.write_text(json.dumps({'schema_version':'v24_join_manifest_v2','corpus':'toy','split':'val','ids':vids,'v23_global_source_sha256':sha(vgs),'evidence_producer_sha256':evidence_sha,'join_producer_sha256':val_join_sha,'evidence_manifest_sha256':'a'*64,'evidence_config_sha256':'b'*64,'labels_manifest_sha256':'c'*64,'bags_sha256':sha(vb)}))
   tp=json.loads((run/'train_protocol.json').read_text());ad=d/'addendum.json';ad.write_text(json.dumps({'schema_version':'v24_protocol_addendum_v1','status':'NO_RETRAIN_PROVENANCE_MIGRATION','corpus':'toy','train_protocol_sha256':sha(run/'train_protocol.json'),'train_id_manifest_sha256':sha(tm),'train_join_manifest_sha256':'d'*64,'train_join_producer_sha256':tp['producer_sha256'],'train_evidence_manifest_sha256':'f'*64,'train_evidence_config_sha256':'1'*64,'train_evidence_producer_sha256':evidence_sha,'selector_source_sha256':sha(HERE/'selector.py'),'val_join_source_sha256':val_join_sha}))
   subprocess.run([sys.executable,str(HERE/'selector.py'),'--train-dir',str(run),'--protocol-addendum',str(ad),'--val-bags',str(vb),'--val-id-manifest',str(vm),'--v23-global-source',str(vgs),'--corpus','toy','--out',str(frozen)],check=True)
   selected=json.loads(frozen.read_text());self.assertIn(selected['status'],('VIDEO_VAL_PASS_PENDING_TEMPORAL','VIDEO_VAL_FAIL_FALLBACK_EPOCH0'));self.assertIn('selected_real_diagnostics',selected);self.assertIn('effective_gamma_min_0.01',selected['gates']);xmj=json.loads(xm.read_text());xmj['producer_sha256']=evidence_sha;xm.write_text(json.dumps(xmj))
   # A synthetic already-video-passed artifact isolates the signed temporal transition test.
   selected['status']='VIDEO_VAL_PASS_PENDING_TEMPORAL';selected['all_video_gates_pass']=True;pending=d/'pending.json';pending.write_text(json.dumps(selected,sort_keys=True))
   denied=d/'denied';q=subprocess.run([sys.executable,str(HERE/'infer.py'),'--frozen-config',str(pending),'--bags',str(xb),'--id-manifest',str(xm),'--v23-global-source',str(xgs),'--split','test','--out-dir',str(denied)]);self.assertNotEqual(q.returncode,0)
   key=d/'key';key.write_bytes(b'synthetic steward secret');payload={'status':'WITHIN_SHUFFLE_PASS','frozen_config_sha256':sha(pending),'within_macro_ap_gain':.02,'within_macro_roc_gain':.03,'shuffle_pass':True,'steward_id':'synthetic-steward'};canon=json.dumps({k:payload[k] for k in sorted(payload)},sort_keys=True,separators=(',',':')).encode();payload['signature_hmac_sha256']=hmac.new(key.read_bytes(),canon,hashlib.sha256).hexdigest();gate=d/'gate.json';gate.write_text(json.dumps(payload));final=d/'final.json';subprocess.run([sys.executable,str(HERE/'finalize.py'),'--video-frozen',str(pending),'--steward-gate',str(gate),'--steward-key',str(key),'--out',str(final)],check=True);self.assertEqual(json.loads(final.read_text())['status'],'FINAL_PASS')
   pred=d/'pred';subprocess.run([sys.executable,str(HERE/'infer.py'),'--frozen-config',str(final),'--bags',str(xb),'--id-manifest',str(xm),'--v23-global-source',str(xgs),'--split','test','--out-dir',str(pred)],check=True)
   labels=d/'labels.json';labels.write_text(json.dumps({v:i%2 for i,v in enumerate(tids)}));ev=d/'eval.json';subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--pred-dir',str(pred),'--labels',str(labels),'--out',str(ev)],check=True);self.assertIn('video_ap',json.loads(ev.read_text()))
if __name__=='__main__':unittest.main()
