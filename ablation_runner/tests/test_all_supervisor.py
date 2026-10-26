import argparse,importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
P=argparse.ArgumentParser()
P.add_argument('--repo',type=Path,default=ROOT/'anchorvra_ablation_review')
P.add_argument('--full-log',type=Path,default=ROOT/'upload/scanrefer_full_20261016_220015(2).log')
A=P.parse_args()
source=ROOT/'start_all_scanrefer.py'
if not source.is_file():source=ROOT.parent/'start_all_scanrefer.py'
spec=importlib.util.spec_from_file_location('supervisor',source)
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)

class SupervisorTests(unittest.TestCase):
    def exercise(self,retry=False,fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);seen=[]
            def fake_run(command,**kwargs):
                name=command[command.index('--variants')+1];seen.append(name)
                self.assertNotEqual(name,'full')
                self.assertNotIn('--max-new-api-calls',command)
                if retry and len(seen)==1:return SimpleNamespace(returncode=75)
                if fail and name=='baseline':return SimpleNamespace(returncode=4)
                output=Path(command[command.index('--out')+1])/name;output.mkdir(parents=True,exist_ok=True)
                metrics=dict(n=250,exceptions=0,**{k:.5 for k in S.METRICS})
                (output/'metrics.json').write_text(json.dumps(metrics))
                return SimpleNamespace(returncode=0)
            a=SimpleNamespace(out=out,repo=A.repo,full_log=A.full_log,model='glm-test',auto_retries=1)
            with patch.object(S.subprocess,'run',side_effect=fake_run),patch.object(S.time,'sleep'):
                result=S.worker(a)
            if fail:
                self.assertEqual(result,1);self.assertFalse((out/'PAPER_READY.json').exists())
                self.assertEqual(len(seen),6)
            else:
                self.assertEqual(result,0);self.assertTrue((out/'PAPER_READY.json').exists())
                self.assertIn('Full AnchorVRA',(out/'ablation_table.tex').read_text())
                self.assertEqual(seen,S.RUN_ORDER if not retry else [S.RUN_ORDER[0],*S.RUN_ORDER])
            self.assertTrue((out/'summary.csv').exists())
    def test_all_groups_and_paper_export(self):self.exercise()
    def test_transient_retry_resumes_then_continues(self):self.exercise(retry=True)
    def test_permanent_group_failure_does_not_fake_complete(self):self.exercise(fail=True)

unittest.main(argv=['test_all_supervisor.py'],verbosity=2)
