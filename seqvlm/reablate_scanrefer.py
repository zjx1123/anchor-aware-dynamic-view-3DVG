#!/usr/bin/env python3
"""Cost-aware ScanRefer ablations. Default is offline planning, never paid execution.

Copy this file to <repo>/seqvlm/. Full is imported from an existing log only.
The existing predictor, prompts, proposal loading and IoU evaluation are retained.
"""
from __future__ import annotations
import argparse,ast,contextlib,copy,csv,hashlib,json,os,random,re,shutil,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VARIANTS={
 'baseline':dict(AQ=False,TC=False,AR=False,FG=False),
 'anchor':dict(AQ=True,TC=False,AR=False,FG=False),
 'target':dict(AQ=False,TC=True,AR=False,FG=False),
 'anchor_target':dict(AQ=True,TC=True,AR=False,FG=False),
 'local_full':dict(AQ=True,TC=True,AR=True,FG=False),
 'no_global_same_gate':dict(AQ=True,TC=True,AR=True,FG=False),
}
METRICS=['Overall@25','Overall@50','Unique@25','Unique@50','Multiple@25','Multiple@50']
CODE=['seqvlm/adaptive_predictor.py','seqvlm/anchor_utils.py','seqvlm/dynamic_view_selector.py',
      'seqvlm/query_parser.py','seqvlm/final_global_view_builder.py','seqvlm/feat_handler.py',
      'seqvlm/utils.py','seqvlm/api.py','prompts/prompt.py','seqvlm/reablate_scanrefer.py']

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def save_csv(path,rows,fields):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def read_full(log_path,data):
    """Recover printed parses, repairing only raw_query using the matched dataset.

    objprint emits apostrophes in raw_query without Python escaping. We replace
    only this redundant field; semantic fields are parsed with literal_eval.
    No arbitrary log content is ever executed.
    """
    text=Path(log_path).read_text(encoding='utf-8',errors='strict')
    chunks=re.split(r'(?m)^Case: (\d+)\s*$',text)
    cases={};parses={}
    for pos in range(1,len(chunks),2):
        i=int(chunks[pos]);body=chunks[pos+1]
        if i in cases or i>=len(data):raise ValueError('Duplicate/out-of-range Full case '+str(i))
        task=data[i]
        for key,expected in [('scene_id',task['scan_id']),('obj_id',task['target_id']),('obj_name',task['obj_name']),('caption',task['caption'])]:
            m=re.search(r'(?m)^'+key+r': (.*)$',body)
            if not m or m.group(1).strip()!=str(expected).strip():raise ValueError(f'Full/data mismatch: case={i}, field={key}')
        m=re.search(r'Parsed Query:\s*(.*?)\nAnchor Infos:',body,re.S)
        if not m:raise ValueError(f'Full has no parsed query in case {i}')
        raw=m.group(1)
        raw=re.sub(r"(?m)^  ['\"]raw_query['\"]:.*,$",lambda _:"  'raw_query': "+repr(task['caption'])+",",raw)
        parsed=ast.literal_eval(raw)
        if not isinstance(parsed,dict) or parsed.get('raw_query')!=task['caption'] or not isinstance(parsed.get('anchors'),list):
            raise ValueError(f'Invalid printed parse: case {i}')
        q=task['caption']
        if q in parses and parses[q]!=parsed:raise ValueError('Same query has different printed parses; resolve before running')
        parses[q]=parsed;cases[i]=dict(case=i,scene_id=task['scan_id'],obj_id=task['target_id'],caption=q)
    if set(cases)!=set(range(len(data))):raise ValueError('Full log must cover exactly the dataset in order')
    metrics={}
    for key in METRICS:
        values=re.findall(r'(?m)^'+re.escape(key)+r':\s*([0-9.]+)\s*$',text)
        if not values:raise ValueError('Full metric missing: '+key)
        metrics[key]=float(values[-1])
    count=re.findall(r'(?m)^Except Ratio:\s*(\d+)\s*/\s*(\d+)\s*$',text)
    if not count or int(count[-1][1])!=len(data):raise ValueError('Full final denominator is not the dataset size')
    metrics['n']=len(data);metrics['exceptions']=int(count[-1][0])
    metrics['gate_cases']=len(re.findall(r'(?m)^\[FinalGlobalGate\]',text))
    metrics['global_view_cases']=len(re.findall(r'(?m)^\[FinalGlobalView\]',text))
    metrics['logged_vlm_outputs']=len(re.findall(r'(?m)^Vision Lang Model Output:',text))
    # These are tokens reported in printed VLM outputs, not a complete bill.
    for kind in ('prompt_tokens','completion_tokens'):
        metrics['logged_'+kind]=sum(map(int,re.findall(r"['\"]"+kind+r"['\"]:\s*(\d+)",text)))
    return parses,metrics,list(cases.values())

def resolve_path(raw):
    p=Path(raw).expanduser()
    return (p if p.is_absolute() else ROOT/p).resolve()

def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-log',required=True)
    p.add_argument('--data',default='data/scanrefer_250.json')
    p.add_argument('--out',required=True)
    p.add_argument('--variants',default='baseline,anchor,target,anchor_target,local_full')
    p.add_argument('--vlm-model',default='glm',help='config.yaml alias; Full confirmed by user as GLM family')
    p.add_argument('--expected-model',help='Exact config.yaml model name used by Full; required for --execute')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-retry',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--max-props',type=int,default=40)
    p.add_argument('--seg-conf-score',type=float,default=.2)
    p.add_argument('--max-anchor-per-type',type=int,default=5)
    p.add_argument('--canvas-k',type=int,default=5)
    p.add_argument('--appearance',type=int,default=3)
    p.add_argument('--relation',type=int,default=2)
    p.add_argument('--max-final-global-anchors',type=int,default=2)
    for name,default in [('image-root','data/scanrefer_preprocessed'),('crop-root','data/crop_images'),
                         ('crop-meta','data/crop_pool_meta_scanrefer'),('view-meta','data/view_meta_scanrefer'),
                         ('posed-root','data/posed_images_rgb_pose'),('global-root','data/global_rendered_views_scanrefer_spazer')]:
        p.add_argument('--'+name,default=default)
    p.add_argument('--cache-root',default='data/cache/reablation_v1')
    p.add_argument('--max-new-api-calls',type=int,default=None,help='Optional transport-attempt cap; omitted means no API-call count limit')
    p.add_argument('--max-samples',type=int,help='Smoke test only; partial rows are never promoted to complete results')
    p.add_argument('--execute',action='store_true',help='Actually run the selected non-Full variants; otherwise offline plan/report only')
    p.add_argument('--resume',action='store_true')
    return p

def preflight(a,data,names):
    errors=[];warnings=[]
    need=['data/scannetv2-labels.combined.tsv','data/huggingface/clip-vit-base-patch16']
    scenes=sorted({t['scan_id'] for t in data})
    need += ['data/mask3d/'+s+'.npz' for s in scenes]
    need += ['data/referit3d/scan_data/pcd_with_global_alignment/'+s+'.pth' for s in scenes]
    need += ['data/referit3d/scan_data/instance_id_to_name/'+s+'.json' for s in scenes]
    missing=[s for s in need if not (ROOT/s).exists()]
    if missing:errors.append(dict(kind='missing_scene_assets',count=len(missing),examples=missing[:8]))
    if not resolve_path(a.image_root).is_dir():errors.append(dict(kind='fixed_canvas_root_missing',path=a.image_root))
    if any(VARIANTS[n]['TC'] for n in names):
        for key in ('crop_root','posed_root'):
            if not resolve_path(getattr(a,key)).is_dir():warnings.append(dict(kind='dynamic_input_root_missing_fixed_fallback_enabled',path=getattr(a,key)))
        for key in ('crop_meta','view_meta'):
            absent=[s for s in scenes if not (resolve_path(getattr(a,key))/(s+'.json')).is_file()]
            if absent:warnings.append(dict(kind='dynamic_metadata_deferred_to_candidate_check',root=getattr(a,key),count=len(absent),examples=absent[:8],
                                           policy='Preserve the original predictor fallback to the corresponding fixed canvas, regardless of candidate count.'))
    model_info={}
    try:
        import yaml
        config=yaml.safe_load((ROOT/'config.yaml').read_text())
        selected=config[a.vlm_model]
        model=str(selected.get('model','')).strip()
        endpoint=str(selected.get('base-url','')).strip()
        if not model or model.startswith('<'):errors.append(dict(kind='model_name_not_configured'))
        if a.expected_model and model!=a.expected_model:errors.append(dict(kind='model_mismatch',configured_model=model,expected_model=a.expected_model))
        # Never include credentials or the raw endpoint in exported manifests.
        model_info=dict(alias=a.vlm_model,model=model,endpoint_digest=digest(endpoint))
    except Exception as exc:
        errors.append(dict(kind='model_config_unavailable',error_type=type(exc).__name__))
    if not a.expected_model:errors.append(dict(kind='expected_model_required_for_paid_run'))
    a._preflight_warnings=warnings
    return errors,model_info

def metrics_from_cases(rows):
    n=len(rows);u=sum(r['unique'] for r in rows);m=n-u
    values={'n':n,'exceptions':sum(r['pred_box'] is None for r in rows),'vlm_cases':sum(r['use_vlm'] for r in rows)}
    for threshold,suffix in [(.25,'25'),(.5,'50')]:
        good=lambda r:r['iou'] is not None and r['iou']>=threshold
        values['Overall@'+suffix]=sum(good(r) for r in rows)/n if n else None
        values['Unique@'+suffix]=sum(good(r) and r['unique'] for r in rows)/u if u else None
        values['Multiple@'+suffix]=sum(good(r) and not r['unique'] for r in rows)/m if m else None
    return values

class BudgetStop(BaseException):pass
class TransportCaseFailure(BaseException):
    def __init__(self,message,exit_code=75):
        super().__init__(message);self.exit_code=exit_code

def asset_inventory(a,data,names):
    """Resume guard for scene inputs. Metadata/proposals have content hashes;
    large images/weights/GT tensors have path/size/mtime stamps. Actual canvas
    dependencies and all API image payloads receive content hashes as well.
    """
    records=[]
    for scene in sorted({t['scan_id'] for t in data}):
        for p,strong in [(ROOT/'data/mask3d'/(scene+'.npz'),True),
                         (ROOT/'data/referit3d/scan_data/pcd_with_global_alignment'/(scene+'.pth'),False),
                         (ROOT/'data/referit3d/scan_data/instance_id_to_name'/(scene+'.json'),True),
                         (resolve_path(a.crop_meta)/(scene+'.json'),True),
                         (resolve_path(a.view_meta)/(scene+'.json'),True)]:
            if p.is_file():
                st=p.stat();records.append([str(p.resolve()),st.st_size,st.st_mtime_ns,file_hash(p) if strong else None])
        for root in [resolve_path(a.image_root)/scene,resolve_path(a.crop_root)/scene,resolve_path(a.posed_root)/scene]:
            if root.is_dir():
                for p in sorted(root.rglob('*')):
                    if p.is_file():
                        st=p.stat();records.append([str(p.resolve()),st.st_size,st.st_mtime_ns])
    weights=ROOT/'data/huggingface/clip-vit-base-patch16'
    if weights.is_dir():
        for p in sorted(weights.rglob('*')):
            if p.is_file():
                st=p.stat();records.append([str(p.resolve()),st.st_size,st.st_mtime_ns])
    return dict(sha256=digest(records),files=len(records))

class CachedAPI:
    def __init__(self,original,a,model_info,out):
        self.original=original;self.a=a;self.calls=0;self.hits=0;self.errors=0;self.last_error_status=None;self.variant='';self.case=-1
        self.cache=resolve_path(a.cache_root)/'api';self.ledger=out/'api_usage.jsonl'
        self.identity=dict(model=model_info,api_source=file_hash(ROOT/'seqvlm/api.py'))
    def __call__(self,model,messages):
        key=digest(dict(identity=self.identity,alias=model,messages=messages))
        path=self.cache/(key+'.json')
        if path.exists():
            stored=json.loads(path.read_text());self.hits+=1
            return copy.deepcopy(stored['response'])
        if self.a.max_new_api_calls is not None and self.calls>=self.a.max_new_api_calls:raise BudgetStop('New API-call budget exhausted; resume uses successful cached requests.')
        self.calls+=1;started=time.time()
        record=dict(variant=self.variant,case=self.case,request_sha256=key,network_attempt=True)
        try:
            result=self.original(model,messages)
            atomic_json(path,dict(request_sha256=key,response=result))
            record.update(status='ok',prompt_tokens=result.get('prompt_tokens'),completion_tokens=result.get('completion_tokens'))
            return result
        except Exception as exc:
            self.errors+=1;self.last_error_status=getattr(exc,'status_code',None)
            record.update(status='error',error_type=type(exc).__name__);raise
        finally:
            record['seconds']=time.time()-started
            with self.ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')

def install_canvas_cache(predictor,a,cfg,cache_identity):
    """Extend the legacy query/target-only key with actual upstream dependencies."""
    if not cfg['TC']:return
    selector=predictor.dynamic_view_selector
    original=selector.build_query_specific_canvas
    hashes={}
    def hashed(path):
        path=Path(path).resolve()
        if not path.is_file():return 'missing:'+str(path)
        stat=path.stat()
        key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key not in hashes:hashes[key]=file_hash(path)
        return hashes[key]
    def wrapped(*args,**kw):
        import numpy as np
        def clean(v):
            if isinstance(v,np.ndarray):return v.tolist()
            if isinstance(v,np.generic):return v.item()
            if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [clean(x) for x in v]
            return v
        scene=kw['scene_id']
        target_id=kw['target_proposal_id']
        crop_meta_path=resolve_path(a.crop_meta)/(scene+'.json')
        view_meta_path=resolve_path(a.view_meta)/(scene+'.json')
        crop_meta=selector._load_json(str(crop_meta_path))
        view_meta=selector._load_json(str(view_meta_path))
        crops=[resolve_path(a.crop_root)/scene/str(target_id)/x['file_name'] for x in crop_meta.get(str(target_id),[])]
        frames=[resolve_path(a.posed_root)/scene/(str(x['image_name'])+'.jpg') for x in view_meta.get(str(target_id),[]) if 'image_name' in x] if cfg['AR'] else []
        deps=dict(identity=cache_identity,scene=scene,
                  crop_meta=hashed(crop_meta_path),view_meta=hashed(view_meta_path),
                  image_sources=[hashed(p) for p in crops+frames],
                  proposals=hashed(ROOT/'data/mask3d'/(scene+'.npz')),
                  parsed=clean(kw.get('parsed_query')) if cfg['AR'] else None,
                  anchors=clean(kw.get('anchor_infos')) if cfg['AR'] else [],
                  target=clean(kw.get('candidate_meta')))
        selector.canvas_root=str(resolve_path(a.cache_root)/'canvases'/digest(deps))
        value=original(*args,**kw)
        if value is None or not Path(value).is_file():
            # Return the original selector result. AdpativePredictor.execute
            # already implements fixed-canvas fallback; never restrict it by
            # candidate count or replace its missing-image handling.
            fixed=Path(predictor.image_path)/scene/str(target_id)/'canvas.jpg'
            events=getattr(predictor,'_canvas_fallback_events',[])
            events.append(dict(scene_id=scene,proposal_id=target_id,fixed_exists=fixed.is_file()))
            predictor._canvas_fallback_events=events
            print(f'[DynamicCanvasUnavailable] scene={scene}, proposal={target_id}; original fixed-canvas fallback retained, fixed_exists={fixed.is_file()}')
        return value
    selector.build_query_specific_canvas=wrapped

def summary(out,names,n_expected,full):
    rows=[]
    for name in names:
        cases=[json.loads(p.read_text()) for p in sorted((out/name/'cases').glob('*.json'))]
        values=metrics_from_cases(cases)
        rows.append(dict(config=name,**VARIANTS[name],status='complete' if len(cases)==n_expected else 'partial' if cases else 'planned',**values))
    rows.append(dict(config='full_imported',AQ=True,TC=True,AR=True,FG=True,status='reused_full_log',**full))
    save_csv(out/'summary.csv',rows,['config','AQ','TC','AR','FG','status','n',*METRICS,'exceptions','vlm_cases'])
    lines=['# ScanRefer ablation summary','', 'Values are percentages. Partial rows are not paper results. Full is never executed.','',
           '| Configuration | Status | N | '+ ' | '.join(METRICS)+' |', '|---|---|---:|'+ '|'.join(['---:']*6)+'|']
    for r in rows:
        vals=[f'{100*r[k]:.2f}' if isinstance(r.get(k),(float,int)) else '—' for k in METRICS]
        lines.append('| '+ ' | '.join([r['config'],r['status'],str(r['n']),*vals])+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')

def execute(a,out,data,names,parses,model_info,identity):
    # Import local inference dependencies only after the zero-cost preflight.
    os.chdir(ROOT/'seqvlm')
    sys.path[:0]=[str(ROOT),str(ROOT/'seqvlm')]
    import numpy as np
    import pandas as pd
    import torch
    import api
    cached=CachedAPI(api.invoke_api,a,model_info,out)
    api.invoke_api=cached
    from seqvlm.query_parser import QueryParser
    def frozen_parse(self,query,fallback_target=None):
        if query not in parses:raise RuntimeError('Query not present in Full parse snapshot; no paid parser fallback is allowed.')
        return copy.deepcopy(parses[query])
    QueryParser.parse=frozen_parse
    from seqvlm.adaptive_predictor import AdpativePredictor
    from utils import load_pc,calc_iou
    labels=pd.read_csv(ROOT/'data/scannetv2-labels.combined.tsv',sep='\t')
    mapping={}
    for category,class_id in zip(labels['raw_category'],labels['nyu40id']):mapping.setdefault(category,class_id)
    for name in names:
        cfg=VARIANTS[name];run=out/name;run.mkdir(exist_ok=True)
        n_app=a.appearance if cfg['AR'] else a.canvas_k
        n_rel=a.relation if cfg['AR'] else 0
        kwargs=dict(image_path=str(resolve_path(a.image_root)),vlm_model=a.vlm_model,max_retry=a.max_retry,
                    max_batch_size=a.batch_size,max_vlm_props=a.max_props,use_anchor_aware=cfg['AQ'],
                    max_anchor_per_type=a.max_anchor_per_type,seg_conf_score=a.seg_conf_score,
                    query_parse_cache_dir=str(out/'query_parse_frozen'),use_dynamic_canvas=cfg['TC'],
                    crop_image_root=str(resolve_path(a.crop_root)),crop_pool_meta_root=str(resolve_path(a.crop_meta)),
                    view_meta_root=str(resolve_path(a.view_meta)),posed_image_root=str(resolve_path(a.posed_root)),
                    dynamic_canvas_root=str(resolve_path(a.cache_root)/'canvases'),canvas_k=a.canvas_k,
                    num_appearance_views=n_app,num_relation_views=n_rel,use_global_context=False,
                    use_final_global_view=name=='no_global_same_gate',use_final_global_gate=True,
                    global_rendered_root=str(resolve_path(a.global_root)),final_global_view_root=str(run/'global_unused'),
                    max_final_global_anchors=a.max_final_global_anchors)
        atomic_json(run/'effective_config.json',kwargs)
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        predictor=AdpativePredictor(**kwargs)
        if name=='no_global_same_gate':
            parse_need=predictor.parse_need_global
            def no_global(answer):
                requested,reason=parse_need(answer)
                print('[AblationNoGlobal] original_need_global=',requested,'reason=',reason)
                return False,'global evidence disabled by ablation; '+reason
            predictor.parse_need_global=no_global
        # Prompt changes do not unnecessarily invalidate identical canvases.
        canvas_identity=dict(selector_hash=identity['code']['seqvlm/dynamic_view_selector.py'],
                             k=a.canvas_k,appearance=n_app,relation=n_rel,
                             crop_root=str(resolve_path(a.crop_root)),posed_root=str(resolve_path(a.posed_root)))
        install_canvas_cache(predictor,a,cfg,canvas_identity)
        cached.variant=name
        task_list=data[:a.max_samples] if a.max_samples else data
        for i,task in enumerate(task_list):
            path=run/'cases'/f'{i:04d}.json'
            if path.exists():
                old=json.loads(path.read_text())
                if old['task_sha256']!=digest(task):raise ValueError('Checkpoint task mismatch')
                continue
            cached.case=i
            with (run/'console.log').open('a',encoding='utf-8',buffering=1) as f, contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
                print('Case:',i,'scene_id:',task['scan_id'],'caption:',task['caption'])
                obj_ids,obj_labels,obj_locs=load_pc(task['scan_id'])
                target_index=list(obj_ids).index(int(task['target_id']))
                class_ids=[int(mapping.get(label,0)) for label in obj_labels]
                unique=class_ids.count(class_ids[target_index])==1
                started=time.time();calls0=cached.calls;hits0=cached.hits;errors0=cached.errors
                predictor._canvas_fallback_events=[]
                pred,use_vlm=predictor.execute(task['scan_id'],task['obj_name'],task['caption'],task['program'])
                if pred is None and cached.errors>errors0:
                    code=76 if cached.last_error_status in (401,403) else 75
                    raise TransportCaseFailure(f'Unfinished case {i}: API transport failed; no failure-as-prediction checkpoint was saved.',code)
                iou=float(calc_iou(pred,obj_locs[target_index])) if pred is not None else None
                row=dict(case=i,task_sha256=digest(task),scene_id=task['scan_id'],obj_id=task['target_id'],caption=task['caption'],
                         unique=unique,iou=iou,pred_box=np.asarray(pred).tolist() if pred is not None else None,
                         use_vlm=bool(use_vlm),new_api_attempts=cached.calls-calls0,api_cache_hits=cached.hits-hits0,
                         elapsed_seconds=time.time()-started)
                row['canvas_fallback_events']=predictor._canvas_fallback_events
                print('CaseResult:',json.dumps(row,ensure_ascii=False));atomic_json(path,row)
            print(f'[{name}] {i+1}/{len(task_list)}, IoU={iou}, new calls={cached.calls}, cache hits={cached.hits}',flush=True)
        results=[json.loads(p.read_text()) for p in sorted((run/'cases').glob('*.json'))]
        atomic_json(run/'metrics.json',metrics_from_cases(results))

def main():
    a=args_parser().parse_args()
    names=[x.strip() for x in a.variants.split(',') if x.strip()]
    if not names or len(names)!=len(set(names)) or any(n not in VARIANTS for n in names):
        raise SystemExit('Choose non-Full variants from '+','.join(VARIANTS)+'; Full can only be imported.')
    if min(a.max_retry,a.batch_size,a.canvas_k,a.max_anchor_per_type)<=0 or a.batch_size<2 or (a.max_new_api_calls is not None and a.max_new_api_calls<0):
        raise SystemExit('Invalid retry/batch/canvas/budget settings')
    if a.appearance+a.relation!=a.canvas_k or min(a.appearance,a.relation)<0:raise SystemExit('appearance + relation must equal canvas_k')
    if a.max_samples is not None and a.max_samples<=0:raise SystemExit('max-samples must be positive')
    data=json.loads(resolve_path(a.data).read_text())
    parses,full,case_index=read_full(resolve_path(a.full_log),data)
    code={p:file_hash(ROOT/p) for p in CODE}
    errors,model_info=preflight(a,data,names)
    identity=dict(code=code,data_sha256=file_hash(resolve_path(a.data)),full_log_sha256=file_hash(resolve_path(a.full_log)),
                  variants=names,model=model_info,settings={k:v for k,v in vars(a).items() if not k.startswith('_') and k not in ('execute','resume','max_samples','max_new_api_calls','out')})
    identity['asset_inventory']=asset_inventory(a,data,names)
    out=resolve_path(a.out)
    if out.exists():
        if not a.resume:raise SystemExit('Output exists; use a new --out, or --resume for the identical experiment.')
        prior=json.loads((out/'manifest.json').read_text())
        if prior['experiment_sha256']!=digest(identity):raise SystemExit('Source/config differs from the saved experiment. Choose a new --out; caches remain reusable.')
    else:
        out.mkdir(parents=True)
        atomic_json(out/'manifest.json',dict(experiment_sha256=digest(identity),identity=identity,
                    full_config_verification='GLM family and 3+2 user-confirmed; exact historical model/prompt/weights are not recoverable from log alone',
                    full_is_imported=True))
        snapshot=out/'source_snapshot';snapshot.mkdir()
        for p in CODE:
            dst=snapshot/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
        shutil.copy2(resolve_path(a.full_log),out/'full_imported.log')
        atomic_json(out/'query_parse_frozen.json',parses)
        save_csv(out/'full_case_index.csv',case_index,['case','scene_id','obj_id','caption'])
    atomic_json(out/'preflight.json',dict(errors=errors,warnings=getattr(a,'_preflight_warnings',[]),model=model_info,parsed_queries=len(parses),full=full))
    summary(out,names,len(data),full)
    print('Full imported:',json.dumps(full,ensure_ascii=False));print('Frozen parsed queries:',len(parses))
    print('Preflight:',json.dumps(errors,ensure_ascii=False));print('Summary:',out/'summary.md')
    if not a.execute:
        print('PLAN ONLY: no model imports, no API calls. Add --execute --resume only after checking preflight.json.');return 0
    if errors:raise SystemExit('Preflight blocked paid execution. Resolve preflight.json; no API has been called.')
    os.environ['PYTHONHASHSEED']=str(a.seed)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    # PYTHONHASHSEED takes effect only at process startup. Re-exec once.
    if os.environ.get('ANCHORVRA_ABLATION_HASH_SEED')!=str(a.seed):
        os.environ['ANCHORVRA_ABLATION_HASH_SEED']=str(a.seed)
        argv=sys.argv[1:]
        if '--resume' not in argv:argv.append('--resume')
        os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),*argv])
    try:execute(a,out,data,names,parses,model_info,identity)
    except BudgetStop as exc:
        print(str(exc));return 3
    except TransportCaseFailure as exc:
        print(str(exc));return exc.exit_code
    finally:summary(out,names,len(data),full)
    return 0

if __name__=='__main__':raise SystemExit(main())
