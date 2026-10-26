"""Offline checks only; the fake API never sends a network request."""
import argparse,copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

P=argparse.ArgumentParser()
P.add_argument('--repo',required=True,type=Path)
P.add_argument('--full-log',required=True,type=Path)
A=P.parse_args()
spec=importlib.util.spec_from_file_location('reablate',A.repo/'seqvlm/reablate_scanrefer.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
DATA=json.loads((A.repo/'data/scanrefer_250.json').read_text())

class OfflineChecks(unittest.TestCase):
    def test_default_has_no_api_count_limit(self):
        args=R.args_parser().parse_args(['--full-log','unused.log','--out','unused'])
        self.assertIsNone(args.max_new_api_calls)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=SimpleNamespace(cache_root=str(root/'cache'),max_new_api_calls=None)
            api=R.CachedAPI(lambda *_:{'answer':'ok'},a,dict(model='test'),root)
            api('glm',[{'text':'one'}]);api('glm',[{'text':'two'}])
            self.assertEqual(api.calls,2)
    def test_full_alignment_and_parse_recovery(self):
        p,m,c=R.read_full(A.full_log,DATA)
        self.assertEqual((len(p),len(c),m['n']),(250,250,250))
        self.assertEqual((m['Overall@25'],m['Overall@50']),(.584,.508))
        self.assertEqual(p[DATA[3]['caption']]['raw_query'],DATA[3]['caption'])
    def test_changed_query_rejected(self):
        d=copy.deepcopy(DATA);d[0]['caption']='wrong query'
        with self.assertRaises(ValueError):R.read_full(A.full_log,d)
    def test_case_metrics_full_precision(self):
        rows=[dict(unique=True,iou=.24999,pred_box=[1],use_vlm=True),dict(unique=False,iou=.5,pred_box=[1],use_vlm=True),dict(unique=False,iou=None,pred_box=None,use_vlm=False)]
        m=R.metrics_from_cases(rows)
        self.assertEqual(m['Overall@25'],1/3);self.assertEqual(m['Overall@50'],1/3)
        self.assertEqual(m['Unique@25'],0);self.assertEqual(m['Multiple@25'],.5)
        self.assertEqual(m['exceptions'],1)
    def test_api_cache_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            def fake(model,messages):
                calls.append(messages);return dict(answer='{}',prompt_tokens=2,completion_tokens=1)
            a=SimpleNamespace(cache_root=str(root/'cache'),max_new_api_calls=1)
            api=R.CachedAPI(fake,a,dict(model='test'),root)
            self.assertEqual(api('glm',[dict(text='same')])['answer'],'{}')
            self.assertEqual(api('glm',[dict(text='same')])['answer'],'{}')
            with self.assertRaises(R.BudgetStop):api('glm',[dict(text='changed')])
            self.assertEqual((len(calls),api.calls,api.hits),(1,1,1))
            restarted=R.CachedAPI(fake,a,dict(model='test'),root)
            restarted('glm',[dict(text='same')]);self.assertEqual(restarted.calls,0)
    def test_model_identity_separates_api_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            def fake(*args):calls.append(1);return {'answer':'ok'}
            a=SimpleNamespace(cache_root=str(root/'cache'),max_new_api_calls=2)
            R.CachedAPI(fake,a,dict(model='v1'),root)('glm',[])
            R.CachedAPI(fake,a,dict(model='v2'),root)('glm',[])
            self.assertEqual(len(calls),2)
    def test_variants_dependency_and_no_full(self):
        self.assertNotIn('full',R.VARIANTS)
        for name,c in R.VARIANTS.items():
            if c['AR']:self.assertTrue(c['AQ'] and c['TC'])
            self.assertFalse(c['FG'])
    def test_canvas_dependency_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for directory in ('crop_meta','view_meta','crop/s/1','posed/s','data/mask3d'):(root/directory).mkdir(parents=True)
            (root/'crop_meta/s.json').write_text(json.dumps({'1':[dict(file_name='a.jpg')]}))
            (root/'view_meta/s.json').write_text(json.dumps({'1':[dict(image_name='f')]}))
            (root/'crop/s/1/a.jpg').write_bytes(b'crop1')
            (root/'posed/s/f.jpg').write_bytes(b'frame1')
            (root/'data/mask3d/s.npz').write_bytes(b'props1')
            a=SimpleNamespace(crop_meta=str(root/'crop_meta'),view_meta=str(root/'view_meta'),crop_root=str(root/'crop'),posed_root=str(root/'posed'),cache_root=str(root/'cache'))
            class Selector:
                def _load_json(self,path):return json.loads(Path(path).read_text())
                def build_query_specific_canvas(self,**kw):
                    p=Path(self.canvas_root)/'canvas.jpg';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'canvas');return str(p)
            def wrapped(ar=True):
                p=SimpleNamespace(dynamic_view_selector=Selector())
                R.install_canvas_cache(p,a,dict(TC=True,AR=ar),dict(k=5,relation=2 if ar else 0))
                return p.dynamic_view_selector.build_query_specific_canvas
            kw=dict(scene_id='s',target_proposal_id=1,query='q',parsed_query={'anchors':[1]},anchor_infos=[{'proposal_id':2}],candidate_meta={'proposal_id':1})
            with patch.object(R,'ROOT',root):
                f=wrapped();first=f(**kw);self.assertEqual(first,f(**kw))
                kw2=copy.deepcopy(kw);kw2['anchor_infos']=[{'proposal_id':3}]
                self.assertNotEqual(first,f(**kw2))
                (root/'crop/s/1/a.jpg').write_bytes(b'changed crop bytes')
                self.assertNotEqual(first,f(**kw))
                appearance=wrapped(False);initial=appearance(**kw)
                self.assertEqual(initial,appearance(**kw2))

unittest.main(argv=['test_reablation_offline.py'],verbosity=2)
