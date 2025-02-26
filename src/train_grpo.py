#!/usr/bin/env python
"""
train_grpo.py

Refactored script that:
  - Dynamically loads reward functions (including your new XML-based ones).
  - Has a 'deepseek_pipeline' that doesn't add extra prompts if they're already included.
  - Uses a modular approach for clarity and maintainability.
"""

import argparse
import importlib
import inspect
import json
import os
import datetime
import re

from datasets import load_dataset, Dataset as HFDataset

# ========== 1) Attempt TRL Import ==========
try:
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    raise ImportError("Please install the trl library (pip install trl) to run this script.")

# ========== 2) Reward Functions ==========

def reward_correctness(prompts, completions, **kwargs):
    """
    Example correctness function for simple text equality
    (this is different from your snippet's specialized correctness).
    """
    references = kwargs.get("reference", [None] * len(prompts))
    rewards = []
    for output, ref in zip(completions, references):
        out_text = output.strip() if isinstance(output, str) else str(output)
        ref_text = ref.strip() if isinstance(ref, str) else str(ref)
        if out_text == ref_text:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def reward_keyword_match(prompts, completions, **kwargs):
    """
    Example function that checks for SEO keywords in the generated text.
    """
    seo_keywords_pt = kwargs.get("product_type_seo_keywords", [])
    seo_keywords_cat = kwargs.get("cat_seo_keywords", [])
    all_keywords = seo_keywords_pt + seo_keywords_cat

    rewards = []
    for completion in completions:
        count = sum(1 for kw in all_keywords if kw.lower() in completion.lower())
        rewards.append(float(count) / (len(all_keywords) or 1))
    return rewards

#
# -------- New Reward Functions from Your Snippet --------
#

def extract_xml_answer(text: str) -> str:
    """Helper to parse text between <answer>...</answer>."""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    return text.split("<answer>")[-1].split("</answer>")[0].strip()

def correctness_reward_func(prompts, completions, **kwargs):
    """
    More specialized correctness that compares the <answer> from the generated text to
    a ground-truth 'answer' in **kwargs.
    
    In your snippet, you had `answer` passed separately. Here, we assume it's stored
    in `kwargs.get("reference", [...])` or `kwargs["answer"]`. 
    """
    ground_truths = kwargs.get("answer", kwargs.get("reference", [None] * len(completions)))
    # The completions might be a list of dictionaries if role-based. If it's already strings, adjust accordingly.
    # We'll assume completions is a list of strings for simplicity:
    rewards = []
    for completion, ref in zip(completions, ground_truths):
        # If your completions come in nested form: [ [ {role: X, content: str}, ... ], ... ]
        # then you might do something like: `text = completion[0]["content"]` 
        text = str(completion)
        model_answer = extract_xml_answer(text)
        if model_answer == str(ref):
            rewards.append(2.0)  # your snippet used 2.0 if correct
        else:
            rewards.append(0.0)
    return rewards

def int_reward_func(prompts, completions, **kwargs):
    """
    Reward if the extracted <answer> is strictly an integer digit (like '42').
    """
    rewards = []
    for completion in completions:
        text = str(completion)
        model_answer = extract_xml_answer(text)
        if model_answer.isdigit():
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def strict_format_reward_func(prompts, completions, **kwargs):
    """
    Checks if the completion matches a strict XML format:
       <reasoning>
         ...
       </reasoning>
       <answer>
         ...
       </answer>
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$"
    # The question mark after \n? means optional final newline
    rewards = []
    for completion in completions:
        text = str(completion)
        match = re.match(pattern, text, flags=re.DOTALL)
        rewards.append(0.5 if match else 0.0)
    return rewards

def soft_format_reward_func(prompts, completions, **kwargs):
    """
    More lenient check that simply looks for <reasoning>...</reasoning> and <answer>...</answer>.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    rewards = []
    for completion in completions:
        text = str(completion)
        match = re.search(pattern, text, flags=re.DOTALL)
        rewards.append(0.5 if match else 0.0)
    return rewards

def count_xml(text) -> float:
    """Your snippet's logic to count how 'well-formed' the XML is."""
    score = 0.0
    if text.count("<reasoning>\n") == 1:
        score += 0.125
    if text.count("\n</reasoning>\n") == 1:
        score += 0.125
    if text.count("\n<answer>\n") == 1:
        score += 0.125
    if text.count("\n</answer>") == 1:
        score += 0.125
    return score

def xmlcount_reward_func(prompts, completions, **kwargs):
    """
    Score the completion based on partial checks for well-formed <reasoning> / <answer> blocks.
    """
    return [count_xml(str(c)) for c in completions]

#
# -------- Consolidated BUILTIN_REWARDS dictionary --------
#
BUILTIN_REWARDS = {
    "correctness": reward_correctness,
    "keyword_match": reward_keyword_match,
    "correctness_xml": correctness_reward_func,
    "int_reward": int_reward_func,
    "strict_format": strict_format_reward_func,
    "soft_format": soft_format_reward_func,
    "xml_count": xmlcount_reward_func,
}

# ========== 3) Preprocessing Pipelines ==========

def preprocess_generic_pipeline(raw_dataset):
    """
    Generic fallback that just stores the dataset's 'prompt' and 'answer' in standard fields.
    (If your raw dataset has role-based messages, you'll probably override this with a custom pipeline.)
    """
    processed = []
    for item in raw_dataset:
        prompt = item.get("prompt", "No prompt found.")
        reference = item.get("answer", None)
        processed.append({"prompt": prompt, "reference": reference})
    return processed

def deepseek_pipeline(raw_dataset):
    """
    If your dataset items already contain a system prompt and user messages, 
    we simply pass them along without adding extra text.
    
    Example structure for each item:
      {
        'prompt': [ { 'role': 'system', 'content': '...' },
                    { 'role': 'user', 'content': 'some question' } ],
        'answer': '42' 
      }
    We'll store exactly that under 'prompt' and 'reference' for the trainer.
    """
    processed = []
    for item in raw_dataset:
        # item['prompt'] might already be a list of {role, content} 
        # no extra system text is added here
        prompt = item.get('prompt', [])
        reference = item.get('answer', None)
        processed.append({"prompt": prompt, "reference": reference})
    return processed

PREPROCESSING_PIPELINES = {
    "generic": preprocess_generic_pipeline,
    "deepseek": deepseek_pipeline,
}

# ========== 4) CLI & Main Logic ==========

def parse_arguments():
    parser = argparse.ArgumentParser(description="GRPO Training Script (with new reward funcs & deepseek pipeline).")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or local path.")
    parser.add_argument("--dataset", type=str, required=True, help="Local JSON/jsonl or dataset identifier.")
    parser.add_argument("--reward_function", type=str, required=True,
                        help="E.g. 'correctness', 'xml_count', 'correctness_xml', 'module:function', etc.")
    parser.add_argument("--preprocessing_pipeline", type=str, default="generic",
                        help="Pipeline name: 'generic' or 'deepseek' or custom.")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--use_unsloth", action='store_true', help="Use Unsloth-based GRPO if available.")
    return parser.parse_args()

def load_dataset_local_or_hub(path_or_name):
    """
    Attempts to load a dataset from local file if it's .json/.jsonl, 
    otherwise uses load_dataset from Hugging Face. 
    """
    if os.path.isfile(path_or_name) and path_or_name.endswith((".json", ".jsonl")):
        with open(path_or_name, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    else:
        ds = load_dataset(path_or_name, split="train")
        return ds

def preprocess_data(raw_data, pipeline_name):
    if pipeline_name not in PREPROCESSING_PIPELINES:
        raise ValueError(f"Unknown pipeline '{pipeline_name}'. Available: {list(PREPROCESSING_PIPELINES.keys())}")
    processed = PREPROCESSING_PIPELINES[pipeline_name](raw_data)
    try:
        return HFDataset.from_list(processed)
    except Exception:
        print("[Warning] Could not convert to HFDataset. Using raw list directly.")
        return processed

def load_reward(reward_arg):
    # Built-in or dynamic import
    if reward_arg in BUILTIN_REWARDS:
        return BUILTIN_REWARDS[reward_arg]
    elif ":" in reward_arg:
        mod_name, func_name = reward_arg.split(":")
        mod = importlib.import_module(mod_name)
        reward_func = getattr(mod, func_name)
        # optional signature check
        sig = inspect.signature(reward_func)
        params = list(sig.parameters.keys())
        if len(params) < 2 or params[:2] != ["prompts", "completions"]:
            raise ValueError(f"Reward function '{func_name}' must have signature like (prompts, completions, **kwargs).")
        return reward_func
    else:
        raise ValueError(f"Reward function '{reward_arg}' not recognized.")

def create_config(args):
    return GRPOConfig(
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        logging_steps=50,
        save_steps=0,
        save_total_limit=1,
    )

def load_model_and_tokenizer(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[{datetime.datetime.now()}] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer

def create_trainer(args, model, tokenizer, reward_func, dataset, config):
    if args.use_unsloth:
        print("[INFO] Using Unsloth-based GRPO approach.")
        try:
            from unsloth import UnslothGRPO
        except ImportError:
            raise ImportError("Unsloth not installed. Remove --use_unsloth or install 'unsloth'.")
        return UnslothGRPO(
            model=model,
            tokenizer=tokenizer,
            reward_func=reward_func,
            config=config,
            dataset=dataset
        )
    else:
        # standard TRL
        return GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[reward_func], 
            args=config,
            train_dataset=dataset
        )

def main():
    args = parse_arguments()

    # 1. Load model/tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)

    # 2. Load dataset
    raw_data = load_dataset_local_or_hub(args.dataset)
    print(f"[{datetime.datetime.now()}] Loaded {len(raw_data)} records in dataset.")

    # 3. Preprocess
    processed_dataset = preprocess_data(raw_data, args.preprocessing_pipeline)

    # 4. Reward function
    reward_func = load_reward(args.reward_function)
    print(f"[{datetime.datetime.now()}] Using reward function: {args.reward_function}")

    # 5. Config
    config = create_config(args)

    # 6. Create trainer
    trainer = create_trainer(args, model, tokenizer, reward_func, processed_dataset, config)

    # 7. Train
    print(f"[{datetime.datetime.now()}] Starting GRPO training...")
    trainer.train()

    # 8. Save model
    print(f"[{datetime.datetime.now()}] Training complete. Saving model to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[INFO] Model and tokenizer saved.")

if __name__ == "__main__":
    main()
