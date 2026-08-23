from __future__ import annotations
import argparse,json
from pathlib import Path
from .actions import read_frame,split_rows,video_path
from .common import ContactLedger,atomic_json
from .visual_assets import model_provenance
def run():
    import decord,torch,transformers
    from transformers import AutoImageProcessor,CLIPVisionModel
    ledger=ContactLedger();row=sorted(split_rows("train",ledger),key=lambda r:str(r["id"]))[0];p=video_path(str(row["id"]));ledger.register(p,"visual_preflight_raw")
    rgb,actual=read_frame(p,0.0)
    if rgb is None:raise RuntimeError("HALT_DECODER_PREFLIGHT")
    mid="openai/clip-vit-large-patch14-336";proc=AutoImageProcessor.from_pretrained(mid,local_files_only=True);model=CLIPVisionModel.from_pretrained(mid,local_files_only=True)
    if model.config.hidden_size!=1024:raise RuntimeError("HALT_VISUAL_DIM")
    return {"schema":"cvoi-visual-preflight/1","decoder":"decord-"+decord.__version__,"torch":torch.__version__,
            "transformers":transformers.__version__,"decoded_shape":list(rgb.shape),"actual_t_s":actual,
            "hidden_size":model.config.hidden_size,"provenance":model_provenance(model,proc),"contact":ledger.snapshot()}
def main():
    a=argparse.ArgumentParser();a.add_argument("--out",type=Path,required=True);x=a.parse_args();r=run();atomic_json(x.out,r);print(json.dumps(r,sort_keys=True))
if __name__=="__main__":main()
