#!/usr/bin/env python
"""Launch the 6 core GLM LoRA ablations on selected tasks."""
import argparse,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default='ablation_core6.json'); ap.add_argument('--model',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--plan',default='train_plan.json'); ap.add_argument('--gpus',default='2,3,4,5,6,7'); ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args()
 config_path = a.config if os.path.isabs(a.config) else os.path.join(os.path.dirname(__file__), a.config)
 c=json.load(open(config_path)); g=[x.strip() for x in a.gpus.split(',') if x.strip()]; jobs=[]
 for task in c['tasks']:
  for e in c['experiments']:
   jobs.append((task,e))
 print('核心消融任务:',len(jobs),'组（',len(c['tasks']),'任务 × 6 配置）')
 def run_one(item):
  i, (task, e) = item
  d=os.path.join(a.data_root,task+'.jsonl'); out=os.path.join(a.output,e['id'],task)
  if os.path.isdir(os.path.join(out, 'final')) and not a.dry_run:
   print('[SKIP 已完成] %s/%s' % (e['id'], task), flush=True)
   return True
  cmd=[sys.executable,os.path.join(os.path.dirname(__file__),'finetune_glm4_lora.py'),'--model',a.model,'--data',d,'--output',out,'--peft-key',task,'--plan',a.plan,'--lora-rank',str(e['rank']),'--target-modules',e['target_modules'],'--eval-ratio',str(c['fixed']['eval_ratio']),'--eval-limit',str(c['fixed'].get('eval_limit',0)),'--do-eval']
  if e['train_on_prompt']: cmd.append('--train-on-prompt')
  cmd += ['--batch-size',str(c['fixed']['batch_size']),'--grad-accum',str(c['fixed']['grad_accum']),'--max-len',str(c['fixed']['max_len']),'--seed',str(c['fixed']['seed']),'--resume']
  print('[GPU %s] %s/%s: %s' % (g[i%len(g)],e['id'],task,' '.join(cmd)), flush=True)
  if not a.dry_run:
   try:
    subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=g[i%len(g)],PYTHONUNBUFFERED='1'),check=True)
   except subprocess.CalledProcessError as exc:
    print('[FAILED] %s/%s exit=%s' % (e['id'], task, exc.returncode), flush=True)
    return False
  return True
 if a.dry_run:
  for item in enumerate(jobs): run_one(item)
 else:
  with ThreadPoolExecutor(max_workers=len(g)) as ex:
   futures=[ex.submit(run_one,item) for item in enumerate(jobs)]
   failed = 0
   for f in as_completed(futures):
    if not f.result(): failed += 1
   print('消融调度结束，失败任务数:', failed, flush=True)
if __name__=='__main__': main()
