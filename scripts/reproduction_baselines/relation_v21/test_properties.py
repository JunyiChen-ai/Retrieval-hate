#!/usr/bin/env python3
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from forward_sequential import nearest_ocr,user_text

class TestV21(unittest.TestCase):
 def test_ocr_deterministic_nearest_and_threshold(self):
  r=[{'t_mid':8,'window_k':2,'texts':[{'text':'bad','conf':.49}]},{'t_mid':2,'window_k':1,'texts':[{'text':'HELLO','conf':.9}]}]
  self.assertEqual(nearest_ocr(r,3),'HELLO')
 def test_no_six_category_heuristic(self):
  x=user_text('generic policy','question','speech','ocr')
  for word in ('target','hostility','dehumanization','exclusion','violence','symbol'):
   self.assertNotIn(word,x.lower())
 def test_asr_arm_has_no_ocr_marker(self):
  self.assertNotIn('Visible OCR',user_text('p','q','s'))
if __name__=='__main__':unittest.main()
