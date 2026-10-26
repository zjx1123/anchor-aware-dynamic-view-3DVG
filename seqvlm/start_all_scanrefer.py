#!/usr/bin/env python3
"""One command: background all six ScanRefer ablations, import Full, export tables.

Run from the extracted bundle:
    python start_all_scanrefer.py --repo /path/to/your/repository
Uses the active Python/conda environment and config.yaml -> glm.model.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os,shutil,subprocess,sys,time
from datetime import datetime
from pathlib import Path

NAMES=['baseline','anchor','target','anchor_target','local_full','no_global_same_gate']
RUN_ORDER=['anchor_target','local_full','no_global_same_gate','baseline','anchor','target']
LABELS={'baseline':'Baseline','anchor':'Anchor only','target':'Target only','anchor_target':'Anchor + Target',
        'local_full':'Local reasoning','no_global_same_gate':'No global (same gate)','full_imported':'Full AnchorVRA'}
METRICS=['Overall@25','Overall@50','Unique@25','Unique@50','Multiple@25','Multiple@50']

def atomic(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)

def load_runner(repo):
    spec=importlib.util.spec_from_file_location('ablation_core',repo/'seqvlm/reablate_scanrefer.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def render_results(out,core,full,n):
    rows=[]
    for name in NAMES:
        p=out/name/'run'/name/'metrics.json'
        status=json.loads((out/name/'supervisor_status.json').read_text()) if (out/name/'supervisor_status.json').exists() else {'status':'pending'}
        metrics=json.loads(p.read_text()) if p.exists() else {}
        # Both count and exit status must be valid; stale/partial files do not pass.
        complete=status.get('returncode')==0 and metrics.get('n')==n
        rows.append(dict(config=name,description=LABELS[name],**core.VARIANTS[name],
                         status='complete' if complete else status.get('status','pending'),**metrics))
    rows.append(dict(config='full_imported',description=LABELS['full_imported'],AQ=True,TC=True,AR=True,FG=True,status='reused_full_log',**full))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['config','description','AQ','TC','AR','FG','status','n',*METRICS,'exceptions'],extrasaction='ignore')
        w.writeheader();w.writerows(rows)
    lines=['# ScanRefer automatic component ablation','', 'Metrics are percentages. Full is imported, never rerun.','',
           '| Configuration | Status | N | '+' | '.join(METRICS)+' |','|---|---|---:|'+ '|'.join(['---:']*6)+'|']
    for row in rows:
        values=[f'{100*row[k]:.2f}' if isinstance(row.get(k),(int,float)) else '—' for k in METRICS]
        lines.append('| '+' | '.join([row['description'],row['status'],str(row.get('n',0)),*values])+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    complete=all(row['status'] in ('complete','reused_full_log') and row.get('n')==n for row in rows)
    if complete:
        latex=[r'% Requires booktabs, graphicx, amssymb.',r'\begin{table*}[t]',r'\centering',
               r'\caption{Component ablation results (\%) on the fixed 250-case ScanRefer subset. Full reuses the main-experiment log. The no-global variant retains the same local gate prompt.}',
               r'\label{tab:component-ablation}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lcccccccccc}',r'\toprule',
               r'Configuration & AQ & TC & AR & FG & O@25 & O@50 & U@25 & U@50 & M@25 & M@50 \\',r'\midrule']
        for row in rows:
            flags=[r'$\checkmark$' if row[k] else '--' for k in ('AQ','TC','AR','FG')]
            latex.append(' & '.join([row['description'],*flags,*[f'{100*row[k]:.1f}' for k in METRICS]])+r' \\')
        latex.extend([r'\bottomrule',r'\end{tabular}%',r'}',r'\end{table*}'])
        (out/'ablation_table.tex').write_text('\n'.join(latex)+'\n')
        atomic(out/'PAPER_READY.json',dict(status='all_complete',new_ablation_runs=6,full_reused=True,cases_per_run=n))
    return complete

def worker(a):
    import fcntl
    out=Path(a.out).resolve();repo=Path(a.repo).resolve()
    with (out/'supervisor.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This output already has an active supervisor.')
        core=load_runner(repo)
        data=json.loads((repo/'data/scanrefer_250.json').read_text())
        _,full,_=core.read_full(Path(a.full_log),data)
        atomic(out/'status.json',dict(status='running',pid=os.getpid(),total_variants=len(NAMES)))
        for index,name in enumerate(RUN_ORDER,1):
            variant=out/name;variant.mkdir(exist_ok=True)
            command=[sys.executable,'-u',str(repo/'seqvlm/reablate_scanrefer.py'),'--full-log',str(a.full_log),
                     '--expected-model',a.model,'--vlm-model','glm','--variants',name,'--out',str(variant/'run'),
                     '--execute','--resume']
            atomic(variant/'command.json',command)
            result=None
            for attempt in range(a.auto_retries+1):
                print(f'[{index}/{len(NAMES)}] {name}, attempt {attempt+1}',flush=True)
                atomic(out/'status.json',dict(status='running',variant=name,variant_index=index,total_variants=len(NAMES),attempt=attempt+1,pid=os.getpid()))
                with (variant/'worker.log').open('a',encoding='utf-8',buffering=1) as log:
                    log.write(f'\n--- supervisor attempt {attempt+1} ---\n')
                    result=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode!=75 or attempt==a.auto_retries:break
                delay=min(30*(2**attempt),120)
                print(f'Temporary API failure: preserving checkpoints, retrying in {delay}s.',flush=True)
                time.sleep(delay)
            atomic(variant/'supervisor_status.json',dict(status='finished' if result.returncode==0 else 'failed',returncode=result.returncode))
            render_results(out,core,full,len(data))
            if result.returncode==76:
                atomic(out/'status.json',dict(status='blocked_authentication',variant=name,pid=os.getpid()))
                print('Authentication failed; stopped before launching other groups.',flush=True)
                return 76
            # A permanent data/code failure in one group does not prevent the
            # independent groups from running; no failed row is called complete.
        ready=render_results(out,core,full,len(data))
        atomic(out/'status.json',dict(status='all_complete' if ready else 'finished_with_failures',pid=os.getpid()))
        print('ALL COMPLETE' if ready else 'Some groups failed; see summary.md and worker.log.',flush=True)
        return 0 if ready else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--full-log',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--auto-retries',type=int,default=3,help='Restart only cases blocked by temporary API transport failures')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--model',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.auto_retries<0:p.error('auto-retries must be nonnegative')
    if a.worker:return worker(a)
    repo=a.repo.expanduser().resolve();bundle=Path(__file__).resolve().parent
    if not (repo/'seqvlm/adaptive_predictor.py').is_file():raise SystemExit('Wrong --repo: current AnchorVRA repository is required.')
    import yaml
    try:
        cfg=yaml.safe_load((repo/'config.yaml').read_text())['glm']
        model=str(cfg.get('model','')).strip()
        for key in ('model','base-url','api-key'):
            if not str(cfg.get(key,'')).strip() or str(cfg[key]).startswith('<'):
                raise ValueError(f'GLM {key} is not configured')
    except Exception as exc:raise SystemExit('Check config.yaml -> glm before launch: '+type(exc).__name__)
    full=(a.full_log.expanduser().resolve() if a.full_log else bundle/'reference/full_scanrefer_20261016.log')
    if not full.is_file():raise SystemExit('Full log not found; supply --full-log /absolute/path/to/log')
    if a.out:
        out=a.out.expanduser().resolve()
    else:
        if a.resume:raise SystemExit('--resume requires the previous --out directory')
        out=repo/'ablation_runs'/('all_scanrefer_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if out.exists() and not a.resume:raise SystemExit('Output exists; select a new directory or use --resume --out.')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'supervisor.lock').exists():
        import fcntl
        with (out/'supervisor.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('This experiment is already running. Progress:',out/'status.json')
                return 0
    backup=out/'previous_scripts';backup.mkdir(exist_ok=True)
    src=bundle/'seqvlm/reablate_scanrefer.py'
    for source,dest in [(src,repo/'seqvlm/reablate_scanrefer.py'),(Path(__file__).resolve(),repo/'seqvlm/start_all_scanrefer.py')]:
        if not source.is_file():raise SystemExit('Runner not found in the extracted bundle: '+str(source))
        if source.resolve()!=dest.resolve() and (not dest.exists() or source.read_bytes()!=dest.read_bytes()):
            if dest.exists():shutil.copy2(dest,backup/(datetime.now().strftime('%H%M%S')+'_'+dest.name))
            shutil.copy2(source,dest)
    core=load_runner(repo)
    data=json.loads((repo/'data/scanrefer_250.json').read_text())
    _,metrics,_=core.read_full(full,data)
    args=core.args_parser().parse_args(['--full-log',str(full),'--out',str(out),'--expected-model',model])
    errors,info=core.preflight(args,data,NAMES)
    atomic(out/'startup_check.json',dict(errors=errors,warnings=getattr(args,'_preflight_warnings',[]),model=info,full_metrics=metrics))
    if errors:
        print(json.dumps(errors,ensure_ascii=False,indent=2))
        raise SystemExit('Preflight failed before any API call; see '+str(out/'startup_check.json'))
    target_log=out/'full_imported.log'
    if target_log.exists():
        if core.file_hash(target_log)!=core.file_hash(full):raise SystemExit('Full log differs from the existing output; use a new directory.')
    else:shutil.copy2(full,target_log)
    run_identity=dict(model=info,full_sha256=core.file_hash(target_log),data_sha256=core.file_hash(repo/'data/scanrefer_250.json'),
                      runner_sha256=core.file_hash(repo/'seqvlm/reablate_scanrefer.py'),variants=NAMES)
    if (out/'launch.json').exists():
        old=json.loads((out/'launch.json').read_text())
        if old.get('run_identity')!=run_identity:raise SystemExit('Configuration/source differs from this output. Choose a new --out.')
    (out/'PAPER_READY.json').unlink(missing_ok=True)
    (out/'ablation_table.tex').unlink(missing_ok=True)
    cmd=[sys.executable,'-u',str(repo/'seqvlm/start_all_scanrefer.py'),'--worker','--repo',str(repo),
         '--full-log',str(target_log),'--out',str(out),'--model',model,'--auto-retries',str(a.auto_retries)]
    atomic(out/'launch.json',dict(command=cmd,model=model,variants=NAMES,full_reused=True,api_count_limit=None,
                                run_identity=run_identity,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with (out/'background.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
    (out/'supervisor.pid').write_text(str(process.pid)+'\n')
    print('STARTED IN BACKGROUND. PID:',process.pid)
    print('Model:',model,'(from config.yaml -> glm.model; must match your historical Full model)')
    print('Output:',out)
    print('Progress:',out/'background.log')
    print('Final results:',out/'summary.csv',out/'ablation_table.tex')
    print('All done only when PAPER_READY.json exists. Closing SSH does not stop this detached process.')
    return 0

if __name__=='__main__':raise SystemExit(main())
