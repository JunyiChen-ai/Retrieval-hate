import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scra_shift_probe import null_calibration, CELLS
out = []
for ds, mt, tag in CELLS:
    r = null_calibration(ds, mt)
    r["tag"] = tag
    print(json.dumps(r), flush=True)
    out.append(r)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "scra_shift_null.json"), "w"), indent=2)
