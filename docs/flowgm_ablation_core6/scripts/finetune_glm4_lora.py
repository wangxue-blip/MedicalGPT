#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finetune_glm4_lora.py —— GLM-4-9B-Chat + LoRA 复现训练脚本

为什么需要这个文件
------------------
**原始训练脚本没有随项目交付**。全盘只能找到训练产物（41 个 run 的 checkpoint、
training_args.bin、trainer_state.json），找不到任何 finetune 入口。所以这个脚本是
**按 checkpoint 里记录的真实超参反推重写的**，不是原件。

已对齐的部分（全部来自 training_args.bin / adapter_config.json，不是猜的）
  LoRA        r=16, alpha=32, dropout=0.05, target_modules=["query_key_value"]
              -> 可训练 5,570,560 参数 = 0.059%，适配器 11,154,104 字节
  优化器      adamw_torch, beta=(0.9,0.999), eps=1e-8, max_grad_norm=1.0
  批次        per_device_train_batch_size=1, gradient_accumulation_steps=4
  精度        bf16=True, fp16=False
  随机种子    seed=42
  保存        save_steps=1000
  Trainer     Seq2SeqTrainer（原始 predict_with_generate=True）
  逐 run 的 max_steps / lr / warmup / weight_decay 见 train_plan.json

**无法对齐、必须由你确认的部分**（原件丢失，下面是我的默认选择）
  1) 角色映射。数据里的 role 是 instruction / user / assistant，而 GLM-4 的
     tokenization_chatglm.py:134 断言 role 只能是 system/user/assistant/observation
     —— "instruction" **不是合法角色**。本脚本默认把 instruction 当作 user 内容，
     并丢弃恒为 "-" 的那条 user（它不携带信息）。用 --keep-dash-input 可保留。
  2) 损失掩码。Qwen 期的 preprocess_sharegpt.py 里是 labels = input_ids.copy()，
     即**对 prompt 也算损失**。GLM 期用的哪种无从考证。本脚本默认**只对 answer
     算损失**（-100 掩掉 prompt），用 --train-on-prompt 切换成前者。
     这两个选择会影响 loss 的绝对数值，**跨版本比 loss 前先统一它们**。

用法
----
  # 单个适配器
  python finetune_glm4_lora.py \
      --model /path/to/glm-4-9b-chat \
      --peft-key white-stage1 \
      --data ../glm_dataset/by_adapter/white-stage1.jsonl \
      --output ./out/white-stage1 \
      --plan train_plan.json

  # 超参手动覆盖 plan
  python finetune_glm4_lora.py ... --max-steps 3000 --lr 5e-5 --do-eval

  # 只检查环境和数据，不训练
  python finetune_glm4_lora.py ... --dry-run
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=["query_key_value"])


def _enc(tok, text):
    """把文本编成 id，不触发 ChatGLM4Tokenizer 那条坏掉的 pad 路径。

    优先用底层 tiktoken（transformers 5.x 下唯一能用的）；
    老环境（transformers 4.44）走不到这条分支时回退到标准 encode。"""
    inner = getattr(tok, "tokenizer", None)
    if inner is not None and hasattr(inner, "encode"):
        try:
            return inner.encode(text, disallowed_special=())
        except TypeError:
            pass
    return tok.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
class JsonlSFTDataset(Dataset):
    """by_adapter/*.jsonl -> input_ids / labels

    每行:
      {"peft_key": ..., "label": ...,
       "messages":[{"role":"instruction","content":...},
                   {"role":"user","content":"-"},
                   {"role":"assistant","content":...}]}
    """

    def __init__(self, path, tokenizer, max_len=2048,
                 keep_dash_input=False, train_on_prompt=False, limit=0):
        self.tok = tokenizer
        self.max_len = max_len
        self.train_on_prompt = train_on_prompt
        self.rows = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    msgs = json.loads(ln)["messages"]
                except Exception:
                    skipped += 1
                    continue
                ins = usr = ans = ""
                for m in msgs:
                    if m["role"] == "instruction":
                        ins = m["content"]
                    elif m["role"] == "user":
                        usr = m["content"]
                    elif m["role"] == "assistant":
                        ans = m["content"]
                if not ins or not ans:
                    skipped += 1
                    continue
                prompt = ins
                if keep_dash_input and usr and usr != "-":
                    prompt = ins + "\n" + usr
                self.rows.append((prompt, ans))
                if limit and len(self.rows) >= limit:
                    break
        self.skipped = skipped

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        prompt, answer = self.rows[i]
        tok = self.tok
        # GLM-4 单轮：[gMASK]<sop> <|user|>\n{prompt} <|assistant|>\n{answer} <eos>
        #
        # 这里手工拼 token id，**刻意绕开 apply_chat_template 和 tok.encode()**。
        # 实测（transformers 5.15.1 + GLM-4-9B-Chat 的 tokenization_chatglm.py）：
        #   tok.apply_chat_template(...)  -> TypeError: ChatGLM4Tokenizer._pad()
        #   tok.encode(...)                  got an unexpected keyword 'padding_side'
        #   tok.build_single_message(...) -> 正常
        #   tok.tokenizer.encode(s, disallowed_special=()) -> 正常（底层 tiktoken）
        # 老版 tokenizer 的 get_command() 在这版里也不存在。
        p_ids = list(tok.get_prefix_tokens())            # [gMASK] <sop>
        p_ids += tok.build_single_message("user", "", prompt)
        p_ids += [tok.convert_tokens_to_ids("<|assistant|>")]
        p_ids += _enc(tok, "\n")
        a_ids = _enc(tok, answer)
        a_ids += [tok.convert_tokens_to_ids("<|endoftext|>")]

        input_ids = p_ids + a_ids
        if self.train_on_prompt:
            labels = list(input_ids)
        else:
            labels = [-100] * len(p_ids) + list(a_ids)

        input_ids = input_ids[: self.max_len]
        labels = labels[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


class PadCollator(object):
    """右侧 pad 到 batch 内最长；labels 用 -100 补。"""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, lab, att = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            lab.append(f["labels"] + [-100] * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(lab, dtype=torch.long),
                "attention_mask": torch.tensor(att, dtype=torch.long)}


# ---------------------------------------------------------------------------
# 评估：ROUGE-L / BLEU-4
# ---------------------------------------------------------------------------
def build_compute_metrics(tokenizer):
    """原仓库里 **没有** compute_metrics 的实现（全库 grep 零命中），
    但 training_args 里 metric_for_best_model='eval_rouge-l'，说明当时存在过。
    这里按 ChatGLM 官方 finetune_demo 的惯例重写：jieba 分词 + rouge_chinese。"""
    try:
        import jieba
        from rouge_chinese import Rouge
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        print("[warn] 缺 jieba / rouge_chinese / nltk，评估将只返回 eval_loss")
        return None

    def compute(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
        dp = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dl = tokenizer.batch_decode(labels, skip_special_tokens=True)
        score = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        rouge = Rouge()
        for p, l in zip(dp, dl):
            ph = " ".join(jieba.cut(p)) or "-"
            lh = " ".join(jieba.cut(l)) or "-"
            try:
                s = rouge.get_scores(ph, lh)[0]
            except Exception:
                continue
            for k in ("rouge-1", "rouge-2", "rouge-l"):
                score[k].append(round(s[k]["f"] * 100, 4))
            score["bleu-4"].append(sentence_bleu(
                [list(lh)], list(ph),
                smoothing_function=SmoothingFunction().method3) * 100)
        return {k: float(np.mean(v)) if v else 0.0 for k, v in score.items()}

    return compute


# ---------------------------------------------------------------------------
def load_plan(plan_path, peft_key):
    if not plan_path or not os.path.isfile(plan_path):
        return {}
    for r in json.load(open(plan_path, encoding="utf-8"))["plan"]:
        if r["peft_key"] == peft_key and r.get("hparams"):
            return r["hparams"]
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="glm-4-9b-chat 目录（需含权重分片）")
    ap.add_argument("--data", required=True, help="by_adapter/<key>.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--peft-key", default="")
    ap.add_argument("--plan", default="train_plan.json")
    ap.add_argument("--eval-data", default="", help="留空则从训练集切 --eval-ratio")
    ap.add_argument("--eval-ratio", type=float, default=0.02)
    ap.add_argument("--eval-limit", type=int, default=0,
                    help="验证集最多取 N 条；0 表示不限制")
    ap.add_argument("--do-eval", action="store_true",
                    help="原始 41 个 run 里只有 white-stage2 开了验证")
    # 超参：不给就用 plan 里的真实值
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=-1.0)
    ap.add_argument("--weight-decay", type=float, default=-1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="原始为 False；显存不够再开，会慢约 30%%")
    ap.add_argument("--keep-dash-input", action="store_true")
    ap.add_argument("--train-on-prompt", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=0.0)
    ap.add_argument("--target-modules", default="query_key_value",
                    help="逗号分隔，如 query_key_value,dense,dense_h_to_4h,dense_4h_to_h")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条，冒烟测试用")
    ap.add_argument("--dry-run", action="store_true", help="只检查环境与数据")
    args = ap.parse_args()

    # ---- 预检 ----
    import transformers
    tv = transformers.__version__
    print("transformers %s | torch %s | cuda %s"
          % (tv, torch.__version__, torch.cuda.is_available()))
    if int(tv.split(".")[0]) >= 5:
        print("[!] GLM-4-9B-Chat 的 modeling_chatglm.py 面向 transformers 4.44，"
              "5.x 下 trust_remote_code 大概率直接报错。建议 pin 到 4.44.2。")
    if not args.dry_run and not torch.cuda.is_available():
        sys.exit("[x] 没有可用的 CUDA 设备。9B + LoRA 需要 ~24GB 显存，CPU 训不了。")

    hp = load_plan(args.plan, args.peft_key or
                   os.path.basename(args.data).replace(".jsonl", ""))
    max_steps = args.max_steps or hp.get("max_steps", 3000)
    lr = args.lr or hp.get("learning_rate", 5e-5)
    warmup = args.warmup_ratio if args.warmup_ratio >= 0 else hp.get("warmup_ratio", 0.0)
    wd = args.weight_decay if args.weight_decay >= 0 else hp.get("weight_decay", 0.0)
    do_eval = args.do_eval or (hp.get("eval_strategy", "no") not in ("no", "IntervalStrategy.NO"))
    print("超参: max_steps=%d lr=%.0e warmup=%.2f wd=%.2f eval=%s%s"
          % (max_steps, lr, warmup, wd, do_eval,
             "  (来自 train_plan.json)" if hp else "  (plan 未命中，用默认值)"))

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ds = JsonlSFTDataset(args.data, tok, args.max_len, args.keep_dash_input,
                         args.train_on_prompt, args.limit)
    print("训练样本 %d 条（跳过 %d 条）" % (len(ds), ds.skipped))
    if len(ds) == 0:
        sys.exit("[x] 数据集为空：%s" % args.data)
    if len(ds) < 150:
        print("[!] 只有 %d 条。实测收敛最差的四类（quic/vpn/tls/tor）样本量全部 <150，"
              "直接开训大概率白跑。先补数据，或改用确定性 Scapy 脚本。" % len(ds))

    eval_ds = None
    if do_eval:
        if args.eval_data:
            eval_ds = JsonlSFTDataset(args.eval_data, tok, args.max_len,
                                      args.keep_dash_input, args.train_on_prompt)
        else:
            n = max(1, int(len(ds) * args.eval_ratio))
            g = torch.Generator().manual_seed(args.seed)
            idx = torch.randperm(len(ds), generator=g).tolist()
            eval_rows = [ds.rows[i] for i in idx[:n]]
            train_rows = [ds.rows[i] for i in idx[n:]]
            eval_ds = JsonlSFTDataset.__new__(JsonlSFTDataset)
            eval_ds.__dict__.update(ds.__dict__)
            eval_ds.rows = eval_rows
            ds.rows = train_rows
            print("切出验证集 %d 条，训练集剩 %d 条" % (len(eval_ds), len(ds)))

        if args.eval_limit > 0 and len(eval_ds) > args.eval_limit:
            eval_ds.rows = eval_ds.rows[:args.eval_limit]
            print("验证集限量为 %d 条" % len(eval_ds))

    if args.dry_run:
        s = ds[0]
        print("\n--- 第 1 条编码检查 ---")
        print("input_ids 长度:", len(s["input_ids"]))
        print("参与损失的 token 数:", sum(1 for x in s["labels"] if x != -100))
        print("解码:", tok.decode(s["input_ids"])[:400])
        print("\n[dry-run] 未加载模型、未训练。")
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model, TaskType
    lora_alpha = args.lora_alpha or args.lora_rank * 2
    lora_cfg = dict(r=args.lora_rank, lora_alpha=lora_alpha,
                    lora_dropout=0.05,
                    target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()])
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, **lora_cfg))
    model.print_trainable_parameters()   # 期望 5,570,560 || 0.0590%

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output,
        max_steps=max_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup,
        weight_decay=wd,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=1,
        max_grad_norm=1.0,
        bf16=True, fp16=False,
        seed=args.seed,
        logging_steps=50,
        save_steps=args.save_steps,
        save_total_limit=None,
        eval_steps=1000 if eval_ds else None,
        eval_strategy="steps" if eval_ds else "no",
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_rouge-l" if eval_ds else None,
        greater_is_better=True,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        predict_with_generate=False,   # 打开会显著变慢；只有需要 ROUGE 时才开
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=ds, eval_dataset=eval_ds,
        data_collator=PadCollator(tok.pad_token_id),
        # Seq2SeqTrainer 默认把完整 vocab logits 交给 compute_metrics；而指标
        # 函数需要 token ids。先 argmax 可避免验证阶段把浮点 logits 当 token 解码。
        preprocess_logits_for_metrics=(
            (lambda logits, labels: logits[0].argmax(dim=-1)
             if isinstance(logits, tuple) else logits.argmax(dim=-1))
            if eval_ds else None
        ),
        compute_metrics=build_compute_metrics(tok) if eval_ds else None,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output, "final"))
    print("完成 ->", args.output)


if __name__ == "__main__":
    main()
