# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train R1 model with GRPO rl algo.
"""
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import re
from datasets import load_dataset
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_zero3_enabled


def _use_vllm_requested(argv):
    """Return whether the CLI explicitly enables vLLM generation."""
    for index, arg in enumerate(argv):
        if arg.startswith("--use_vllm="):
            return arg.split("=", 1)[1].lower() in {"1", "true", "yes", "y"}
        if arg == "--use_vllm":
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                return argv[index + 1].lower() in {"1", "true", "yes", "y"}
            return True
    return False


def _disable_unused_deepspeed_import():
    """Keep standard Trainer/DDP runs independent of an optional DeepSpeed install."""
    import accelerate.utils.other as accelerate_other

    accelerate_other.is_deepspeed_available = lambda: False


# Some TRL releases eagerly import any installed vLLM, even when GRPO runs
# with --use_vllm=False. Avoid importing an incompatible optional vLLM in that
# mode; enabling vLLM still uses TRL's normal compatibility checks.
if not _use_vllm_requested(sys.argv[1:]):
    import trl.import_utils as _trl_import_utils

    _trl_import_utils._vllm_available = False

from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
try:
    from training.medical_grpo_rewards import (
        length_repetition_penalty,
        medical_format_reward,
        medical_safety_reward,
        reference_similarity_reward,
    )
except ModuleNotFoundError:
    from medical_grpo_rewards import (
        length_repetition_penalty,
        medical_format_reward,
        medical_safety_reward,
        reference_similarity_reward,
    )

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with GRPO
    """
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing training files for local datasets."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "Path to a local JSON/JSONL training file."}
    )
    peft_path: Optional[str] = field(
        default=None, metadata={"help": "Optional existing PEFT/LoRA adapter path to continue GRPO training from."}
    )
    train_samples: Optional[int] = field(default=-1, metadata={"help": "Number of samples to train on, -1 for all"})
    subset_name: Optional[str] = field(default="main",
                                       metadata={"help": "Subset name, e.g., 'default', 'main'. default is 'default'"})
    dataset_splits: Optional[str] = field(default="train", metadata={"help": "Split name"})
    preprocessing_num_workers: Optional[int] = field(default=10,
                                                     metadata={"help": "Number of workers for preprocessing"})
    # QLoRA arguments
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    reward_type: str = field(
        default="default",
        metadata={"help": "Reward type: default for math rewards, medical for medical QA rewards."}
    )
    similarity_model: Optional[str] = field(
        default="BAAI/bge-small-zh-v1.5",
        metadata={"help": "Sentence-Transformers model/path used by medical reference similarity reward."}
    )
    similarity_device: Optional[str] = field(
        default="cpu",
        metadata={"help": "Device for the medical similarity model, e.g. cpu, cuda:0, or auto."}
    )
    reward_mode: str = field(
        default="separate",
        metadata={"help": "Medical reward composition: separate (legacy direct sum) or combined (weighted composite)."},
    )
    dataset_seed: int = field(
        default=42,
        metadata={"help": "Seed used to shuffle/select the local GRPO data before training."},
    )


def normalize_text(text):
    """Normalize text by removing extra whitespace, converting to lowercase."""
    if text is None:
        return ""
    # Remove extra whitespace and convert to lowercase
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text


def extract_answer(text):
    """Extract content between <answer> tags."""
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def accuracy_reward(completions, answer, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
    except ImportError as exc:
        raise RuntimeError(
            "The default math reward requires latex2sympy2_extended and math_verify. "
            "Medical GRPO does not require these packages; run it with --reward_type medical."
        ) from exc

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, answer):
        if '####' in sol:
            # for GSM8K
            gold_parsed = parse(sol.split("####", 1)[-1].strip())
            answer_parsed = parse(extract_answer(content))
        else:
            # First try latex parsing
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            logger.warning(f"Error in verification: {e}")
            reward = 0.0
        logger.debug(f"predict_answer: {content}, \nground_truth: {sol}, \n"
                     f"answer_parsed: {answer_parsed}, gold_parsed: {gold_parsed}, reward: {reward}\n\n")
        rewards.append(reward)
    logger.debug(f'accuracy rewards: {rewards}')
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]
    logger.debug(f'format rewards: {rewards}')
    return rewards


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

MEDICAL_SYSTEM_PROMPT = (
    "你是一个医学科普与健康咨询助手。请基于用户描述给出结构化、谨慎的中文回答，"
    "包含可能原因或分析、处理建议、风险提示和就医建议。不要替代医生诊断，"
    "不要给出具体处方剂量，涉及急症、孕妇、儿童、老人或症状加重时应提示及时就医。"
)


def build_reference_similarity_reward(similarity_model_name=None, similarity_device="cpu"):
    """Build a lazy-loading medical reference similarity reward wrapper."""
    model_holder = {"model": None, "failed": False}
    model_name = (similarity_model_name or "").strip()
    if model_name.lower() in {"none", "false", "off"}:
        model_name = ""
    device = (similarity_device or "cpu").strip()

    def _reward(completions, answer, **kwargs):
        if model_name and not model_holder["failed"] and model_holder["model"] is None:
            try:
                from sentence_transformers import SentenceTransformer
                if device == "auto":
                    model_holder["model"] = SentenceTransformer(model_name)
                else:
                    model_holder["model"] = SentenceTransformer(model_name, device=device)
                logger.info(f"Loaded medical similarity model: {model_name} on device={device}")
            except Exception as exc:
                model_holder["failed"] = True
                logger.warning(
                    f"Failed to load similarity model {model_name!r}; "
                    f"falling back to char n-gram reference similarity. Error: {exc}"
                )
        return reference_similarity_reward(
            completions,
            answer,
            embedding_model=model_holder["model"],
            **kwargs,
        )

    _reward.__name__ = "reference_similarity_reward"
    return _reward


def build_length_penalty_reward():
    """Build a reward wrapper that subtracts length/repetition penalties in GRPO."""

    def _reward(completions, **kwargs):
        return [-score for score in length_repetition_penalty(completions, **kwargs)]

    _reward.__name__ = "length_repetition_penalty"
    return _reward


def build_combined_medical_reward(script_args: ScriptArguments):
    """Build the documented weighted medical reward with the configured similarity backend.

    This deliberately uses the same lazy similarity wrapper as the legacy
    four-function setup.  It isolates reward composition while keeping the
    embedding-model/fallback behavior unchanged for the reward ablation.
    """
    similarity_reward = build_reference_similarity_reward(
        script_args.similarity_model, script_args.similarity_device
    )

    def _reward(completions, answer, **kwargs):
        fmt = medical_format_reward(completions, **kwargs)
        similarity = similarity_reward(completions, answer, **kwargs)
        safety = medical_safety_reward(completions, **kwargs)
        penalty = length_repetition_penalty(completions, **kwargs)
        return [
            max(0.0, min(1.0, 0.35 * f + 0.30 * sim + 0.25 * safe - 0.10 * pen))
            for f, sim, safe, pen in zip(fmt, similarity, safety, penalty)
        ]

    _reward.__name__ = "combined_medical_reward"
    return _reward


def get_reward_funcs(script_args: ScriptArguments):
    reward_type = (script_args.reward_type or "default").lower()
    if reward_type == "default":
        return [accuracy_reward, format_reward]
    if reward_type == "medical":
        reward_mode = (script_args.reward_mode or "separate").lower()
        if reward_mode == "combined":
            return [build_combined_medical_reward(script_args)]
        if reward_mode != "separate":
            raise ValueError(
                f"Unsupported medical reward_mode={script_args.reward_mode!r}. "
                "Use separate or combined."
            )
        return [
            medical_format_reward,
            build_reference_similarity_reward(script_args.similarity_model, script_args.similarity_device),
            medical_safety_reward,
            build_length_penalty_reward(),
        ]
    raise ValueError(f"Unsupported reward_type={script_args.reward_type!r}. Use default or medical.")


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def grpo_train(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    # Add distributed training initialization
    is_main_process = training_args.local_rank in [-1, 0]

    # Only log on main process
    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    if script_args.train_file and os.path.exists(script_args.train_file):
        dataset = load_dataset("json", data_files=script_args.train_file, split="train")
    elif script_args.train_file_dir and os.path.exists(script_args.train_file_dir):
        # Load from local directory
        dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")
    else:
        # Load from HuggingFace hub
        dataset = load_dataset(script_args.dataset_name, script_args.subset_name, split=script_args.dataset_splits)

    if script_args.train_samples > 0:
        sample_count = min(script_args.train_samples, len(dataset))
        if is_main_process and sample_count < script_args.train_samples:
            logger.warning(f"train_samples={script_args.train_samples} exceeds dataset size={len(dataset)}, using {sample_count} samples.")
        dataset = dataset.shuffle(seed=script_args.dataset_seed).select(range(sample_count))

    reward_funcs = get_reward_funcs(script_args)
    prompt_template = MEDICAL_SYSTEM_PROMPT if (script_args.reward_type or "default").lower() == "medical" else SYSTEM_PROMPT
    if is_main_process:
        logger.info(f"Using reward_type={script_args.reward_type}, reward_funcs={[fn.__name__ for fn in reward_funcs]}")

    # Prepare dataset
    with training_args.main_process_first(desc="Dataset preparation"):
        dataset = dataset.map(
            lambda x: {
                'prompt': [
                    {'role': 'system', 'content': prompt_template},
                    {'role': 'user', 'content': x['question']}
                ],
                'answer': x['answer']
            },
            num_proc=script_args.preprocessing_num_workers,
            desc="Processing dataset" if is_main_process else None,
        )

    # Split dataset only when evaluation is enabled. For medical GRPO project
    # runs we usually set eval_strategy=no and should train on the requested
    # 500/1000 examples instead of silently dropping 10%.
    if training_args.eval_strategy != "no" and len(dataset) > 1:
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
    else:
        train_dataset = dataset
        test_dataset = None

    if is_main_process:
        logger.info("*** Initializing model kwargs ***")

    # Model initialization
    torch_dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )

    # Set up distributed training config
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    # Check for QLoRA compatibility
    if script_args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")

    # Check quantization settings
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")

    # Set up quantization config
    quantization_config = None
    if script_args.qlora and (model_args.load_in_4bit or model_args.load_in_8bit):
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif model_args.load_in_4bit or model_args.load_in_8bit:
        # Support quantization even without qlora flag
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
    )

    num_gpus = torch.cuda.device_count()
    # Let Trainer/DDP place one complete model on each worker's local GPU.
    # `device_map="auto"` makes Accelerate treat even a single-GPU model as
    # model-parallel, which is incompatible with ordinary Trainer/GRPOTrainer.
    model_kwargs["device_map"] = None

    if is_main_process:
        logger.info(f"Using {num_gpus} GPUs")
        logger.info(f"model_kwargs={model_kwargs}")

    config = AutoModelForCausalLM.config_class if hasattr(AutoModelForCausalLM, 'config_class') else None
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            revision=model_args.model_revision,
        )
    except Exception:
        config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Patch MoE modules for DeepSpeed ZeRO-3
    model_type = getattr(config, "model_type", None) if config else getattr(model.config, "model_type", None)
    if model_type == "mixtral" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if model_type == "deepseek_v3" and is_deepspeed_zero3_enabled():
        for layer in model.model.layers:
            if 'DeepseekV3MoE' in str(type(layer.mlp)):
                layer.mlp._z3_leaf = True

    if model_type == "qwen3_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

    if model_type == "qwen3_5_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3_5MoeSparseMoeBlock])

    if is_main_process and hasattr(model, 'hf_device_map'):
        logger.info(f"Model Device Map: {model.hf_device_map.items()}")
    elif is_main_process and num_gpus > 1:
        logger.info("Model Device Map:")
        for name, param in model.named_parameters():
            if hasattr(param, 'device'):
                logger.info(f"  {name}: {param.device}")
                break

    loaded_peft_adapter = False
    if script_args.peft_path:
        if not os.path.exists(script_args.peft_path):
            raise ValueError(f"peft_path does not exist: {script_args.peft_path}")
        if is_main_process:
            logger.info(f"Loading trainable PEFT adapter from: {script_args.peft_path}")
        model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        loaded_peft_adapter = True

    # Configure LoRA if enabled
    if model_args.use_peft:
        if loaded_peft_adapter:
            if is_main_process:
                logger.info("Fine-tuning method: continue existing LoRA(PEFT) adapter")
            model.print_trainable_parameters()
        else:
            if is_main_process:
                logger.info("Fine-tuning method: LoRA(PEFT)")
            if training_args.gradient_checkpointing:
                logger.warning("Gradient checkpointing is enabled. It may cause issues with LoRA, setting it to False.")
                training_args.gradient_checkpointing = False
            target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
            if target_modules == 'all' or (target_modules and 'all' in target_modules):
                target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)
            if is_main_process:
                logger.info(f"Peft target_modules: {target_modules}, lora rank: {model_args.lora_r}, ")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
            )
            model = get_peft_model(model, peft_config)
            # Fixed FP16 ValueError for quantized models
            for param in filter(lambda p: p.requires_grad, model.parameters()):
                param.data = param.data.to(torch.float32)
            model.print_trainable_parameters()
    else:
        if is_main_process:
            logger.info("Fine-tuning method: Full parameters training")

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")

    # Initialize GRPO trainer with distributed training support
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    logger.info("*** GRPO Trainer initialized ***")
    logger.debug(f"Trainer: {trainer}")

    # Training
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if is_main_process:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
            f'{training_args.num_train_epochs} epochs ***'
        )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log and save metrics on main process
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("*** Training complete ***")
        logger.info("*** Save model ***")

    # Save model
    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

        # Create model card and save config
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["r1", "grpo"],
        }
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if is_main_process:
        logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    if getattr(training_args, "deepspeed", None) is None:
        _disable_unused_deepspeed_import()

    # Run the main training loop
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
