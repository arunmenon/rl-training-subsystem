#!/usr/bin/env python
"""
train_grpo.py

A parameterized GRPO training script that supports dynamic reward function imports and modular preprocessing.

Usage examples:
    python train_grpo.py \
      --model facebook/opt-1.3b \
      --dataset /data/title_data.json \
      --preprocessing_pipeline title \
      --reward_function correctness \
      --max_steps 300 --learning_rate 1e-5 --batch_size 2

    python train_grpo.py \
      --model EleutherAI/pythia-2.8b \
      --dataset my_qa_dataset \
      --preprocessing_pipeline generic \
      --reward_function my_rewards:custom_reward \
      --num_generations 8 --max_steps 600 --batch_size 1
"""

import argparse
import importlib
import json
import os
import datetime

from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    raise ImportError("Please install the trl library (e.g., pip install trl) to run this script.")

def reward_correctness(prompts, completions, **kwargs):
    references = kwargs.get("reference", [None] * len(prompts))
    rewards = []
    for output, ref in zip(completions, references):
        score = 1.0 if ref and output.strip() == ref.strip() else 0.0
        rewards.append(score)
    return rewards

def reward_keyword_match(prompts, completions, **kwargs):
    seo_keywords = []
    if "product_type_seo_keywords" in kwargs:
        seo_keywords.extend(kwargs["product_type_seo_keywords"])
    if "cat_seo_keywords" in kwargs:
        seo_keywords.extend(kwargs["cat_seo_keywords"])
    rewards = []
    for output in completions:
        count = sum(1 for kw in seo_keywords if kw.lower() in output.lower())
        rewards.append(float(count) / (len(seo_keywords) or 1))
    return rewards

BUILTIN_REWARDS = {
    "correctness": reward_correctness,
    "keyword_match": reward_keyword_match,
}

def preprocess_title_pipeline(raw_dataset):
    processed = []
    for item in raw_dataset:
        product_type = item.get("product-type", "unknown product")
        category = item.get("category", "general")
        attr_bag = item.get("product-type-attribute-bag", {})
        seo_keywords_pt = item.get("product-type-seo-keywords", [])
        seo_keywords_cat = item.get("cat-seo-keywords", [])
        attr_list = ", ".join(f"{k}: {v}" for k, v in attr_bag.items())
        prompt = (f"Enhance the product title for a {product_type} in the {category} category. "
                  f"Important attributes: {attr_list}. Use SEO keywords if appropriate. "
                  f"Generate a catchy and optimized title.")
        reference = item.get("answer", None)
        processed.append({
            "prompt": prompt,
            "reference": reference,
            "product_type": product_type,
            "category": category,
            "product_type_attribute_bag": attr_bag,
            "product_type_seo_keywords": seo_keywords_pt,
            "cat_seo_keywords": seo_keywords_cat
        })
    return processed

def preprocess_generic_pipeline(raw_dataset):
    processed = []
    for item in raw_dataset:
        prompt = item.get("prompt", str(item))
        reference = item.get("answer", None)
        processed.append({
            "prompt": prompt,
            "reference": reference
        })
    return processed

PREPROCESSING_PIPELINES = {
    "title": preprocess_title_pipeline,
    "generic": preprocess_generic_pipeline,
}

def main():
    parser = argparse.ArgumentParser(description="Parameterized GRPO Training Script")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or local path.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset file (json/jsonl) or Hub identifier.")
    parser.add_argument("--reward_function", type=str, required=True,
                        help="Reward function name (e.g., 'correctness', 'keyword_match' or 'module:function').")
    parser.add_argument("--preprocessing_pipeline", type=str, default="generic",
                        help="Preprocessing pipeline name (e.g., 'title' or 'generic').")
    parser.add_argument("--max_steps", type=int, default=500, help="Total training steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of completions per prompt for GRPO.")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Max token length for prompts.")
    parser.add_argument("--max_completion_length", type=int, default=128, help="Max token length for outputs.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs and checkpoints.")
    args = parser.parse_args()

    print(f"[{datetime.datetime.now()}] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print(f"[{datetime.datetime.now()}] Loading dataset from: {args.dataset}")
    if os.path.isfile(args.dataset) and (args.dataset.endswith(".json") or args.dataset.endswith(".jsonl")):
        with open(args.dataset, "r") as f:
            data = json.load(f)
        raw_dataset = data if isinstance(data, list) else [data]
    else:
        raw_dataset = load_dataset(args.dataset, split="train")
    print(f"[{datetime.datetime.now()}] Loaded {len(raw_dataset)} samples.")

    pipeline_name = args.preprocessing_pipeline.lower()
    if pipeline_name in PREPROCESSING_PIPELINES:
        print(f"[{datetime.datetime.now()}] Applying preprocessing pipeline: {pipeline_name}")
        processed_dataset = PREPROCESSING_PIPELINES[pipeline_name](raw_dataset)
    else:
        raise ValueError(f"Unknown preprocessing pipeline '{pipeline_name}'.")
    try:
        train_dataset = HFDataset.from_list(processed_dataset)
    except Exception as e:
        print("Warning: Unable to convert to HF Dataset; using raw list.")
        train_dataset = processed_dataset

    if args.reward_function in BUILTIN_REWARDS:
        reward_func = BUILTIN_REWARDS[args.reward_function]
        print(f"[{datetime.datetime.now()}] Using built-in reward function: {args.reward_function}")
    elif ":" in args.reward_function:
        module_name, func_name = args.reward_function.split(":")
        print(f"[{datetime.datetime.now()}] Importing reward function '{func_name}' from module '{module_name}'")
        mod = importlib.import_module(module_name)
        reward_func = getattr(mod, func_name)
    else:
        raise ValueError(f"Reward function '{args.reward_function}' not recognized.")

    config = GRPOConfig(
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

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[reward_func],
        args=config,
        train_dataset=train_dataset
    )

    print(f"[{datetime.datetime.now()}] Starting GRPO training...")
    trainer.train()
    print(f"[{datetime.datetime.now()}] Training complete. Saving model to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[{datetime.datetime.now()}] Model and tokenizer saved.")

if __name__ == "__main__":
    main()
