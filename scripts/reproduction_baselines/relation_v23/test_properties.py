#!/usr/bin/env python3
import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from preregister import make_windows,speech_in_window,POLICY
class TestV23(unittest.TestCase):
 def test_full_coverage_no_overlap_last(self):
  w=make_windows(65.5);self.assertEqual([(x['start'],x['end']) for x in w],[(0,30),(30,60),(60,65.5)])
  self.assertEqual(w[0]['start'],0);self.assertEqual(w[-1]['end'],65.5)
  for a,b in zip(w,w[1:]):self.assertEqual(a['end'],b['start'])
 def test_exact_multiple_has_no_empty_last(self):self.assertEqual(len(make_windows(60)),2)
 def test_missing_speech_literal(self):self.assertEqual(speech_in_window([],0,30),'[NO SPEECH]')
 def test_asr_cap_prefix(self):self.assertEqual(speech_in_window([{'start':0,'end':1,'text':'x'*4000}],0,30),'x'*3000)
 def test_midpoint_assignment_once(self):
  c=[{'start':29,'end':31,'text':'x'}];self.assertEqual(speech_in_window(c,0,30),'[NO SPEECH]');self.assertEqual(speech_in_window(c,30,60),'x')
 def test_finite_windows(self):
  for x in make_windows(31.2):self.assertTrue(all(math.isfinite(x[k]) for k in ('start','end','center')))
 def test_generic_policy(self):
  for x in ('THVL','Bias','Verbal Abuse'):self.assertNotIn(x,POLICY)
if __name__=='__main__':unittest.main()
