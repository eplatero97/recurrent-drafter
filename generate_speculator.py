#!/usr/bin/env python3
"""
Standalone generation script for recurrent drafting speculator.

This script provides equivalent functionality to Apple's recurrent_drafting.cmd.generate
but uses the speculators framework implementation instead of the original Apple code.

Based on: recurrent_drafting/cmd/generate.py from ml-recurrent-drafter
Original Copyright (C) 2024 Apple Inc. All Rights Reserved.

Example usage:

# Interactive chat mode
python generate_speculator.py \
    --base_model gpt2 \
    --speculator_path ./models/gpt2-recurrent-drafter \
    --interactive

# Benchmark with MT-Bench dataset
python generate_speculator.py \
    --base_model gpt2 \
    --speculator_path ./models/gpt2-recurrent-drafter \
    --eval_mt_bench \
    --max_num_prompts 64 \
    --batch_size 4

# Compare with autoregressive baseline
python generate_speculator.py \
    --base_model gpt2 \
    --speculator_path ./models/gpt2-recurrent-drafter \
    --eval_mt_bench \
    --autoregressive  # Disable speculative decoding
"""

import argparse
import os
import time
from typing import Generator, List, Optional

import datasets
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, 'speculators/src')

from speculators.models.recurrent_drafting import RecurrentDraftingConfig, RecurrentDraftingSpeculator

# Optional wandb import for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class GenerationStats:
    """Track generation statistics for benchmarking."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_tokens_generated = 0
        self.total_time = 0.0
        self.total_prompts = 0
        self.acceptance_rates = []
        self.tokens_per_second = []
    
    def add_generation(self, tokens_generated: int, time_taken: float, acceptance_rate: float = 0.0):
        self.total_tokens_generated += tokens_generated
        self.total_time += time_taken
        self.total_prompts += 1
        self.acceptance_rates.append(acceptance_rate)
        if time_taken > 0:
            self.tokens_per_second.append(tokens_generated / time_taken)
    
    def get_summary(self):
        if self.total_prompts == 0:
            return {}
        
        avg_acceptance = sum(self.acceptance_rates) / len(self.acceptance_rates) if self.acceptance_rates else 0
        avg_tokens_per_sec = sum(self.tokens_per_second) / len(self.tokens_per_second) if self.tokens_per_second else 0
        
        return {
            "total_prompts": self.total_prompts,
            "total_tokens": self.total_tokens_generated,
            "total_time": self.total_time,
            "avg_tokens_per_prompt": self.total_tokens_generated / self.total_prompts,
            "avg_tokens_per_second": avg_tokens_per_sec,
            "avg_acceptance_rate": avg_acceptance,
            "overall_tokens_per_second": self.total_tokens_generated / self.total_time if self.total_time > 0 else 0
        }


def load_model_and_tokenizer(base_model: str, speculator_path: Optional[str], device: torch.device, dtype: torch.dtype):
    """Load base model, tokenizer, and optionally the speculator."""
    
    print(f"🔄 Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    speculator = None
    if speculator_path:
        print(f"🔄 Loading speculator: {speculator_path}")
        try:
            # Load speculator configuration and model
            speculator = RecurrentDraftingSpeculator.from_pretrained(speculator_path)
            speculator.attach_verifier(base_model_obj)
            speculator.to(device)
            speculator.eval()
            print("✅ Speculator loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load speculator: {e}")
            print("🔄 Falling back to autoregressive generation")
            speculator = None
    
    base_model_obj.to(device)
    base_model_obj.eval()
    
    return base_model_obj, tokenizer, speculator


def create_vicuna_prompt(user_input: str) -> str:
    """Create Vicuna-style prompt format."""
    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    )
    return f"{system_prompt}USER: {user_input.strip()} ASSISTANT:"


def load_mt_bench_prompts(max_length: int, max_num_prompts: int) -> Generator[str, None, None]:
    """Load MT-Bench evaluation prompts."""
    
    print(f"🔄 Loading MT-Bench dataset (max {max_num_prompts} prompts)...")
    try:
        eval_dataset = datasets.load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    except Exception as e:
        print(f"⚠️ Could not load MT-Bench dataset: {e}")
        print("🔄 Using sample prompts instead...")
        
        # Fallback sample prompts
        sample_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "What are the benefits of renewable energy?",
            "How does photosynthesis work?",
            "Describe the process of making coffee.",
            "What is the difference between AI and machine learning?",
        ]
        
        for i, prompt in enumerate(sample_prompts):
            if max_num_prompts >= 0 and i >= max_num_prompts:
                break
            yield create_vicuna_prompt(prompt)
        return
    
    n_prompts = 0
    for row in eval_dataset:
        if max_num_prompts >= 0 and n_prompts >= max_num_prompts:
            break
        
        prompt = row["prompt"][0].strip()
        if 2 < len(prompt) < max_length:
            n_prompts += 1
            yield create_vicuna_prompt(prompt)


def batch_prompts(prompts: Generator[str, None, None], batch_size: int) -> Generator[List[str], None, None]:
    """Batch prompts for efficient processing."""
    batch = []
    for prompt in prompts:
        batch.append(prompt)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:  # Yield remaining prompts
        yield batch


def generate_with_speculator(
    model: RecurrentDraftingSpeculator,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    beam_width: int,
    beam_length: int,
    temperature: float,
    greedy: bool
) -> tuple[List[str], float]:
    """Generate text using the speculator model."""
    
    # Tokenize prompts
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    
    start_time = time.time()
    
    # Generate with speculator
    with torch.no_grad():
        output_ids = model.generate(
            inputs=input_ids,
            max_new_tokens=max_new_tokens,
            beam_width=beam_width,
            beam_length=beam_length,
            temperature=temperature,
            greedy=greedy,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = time.time() - start_time
    
    # Decode outputs
    generated_texts = []
    for i, output in enumerate(output_ids):
        # Remove input tokens to get only generated text
        generated_tokens = output[len(input_ids[i]):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts, generation_time


def generate_autoregressive(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    greedy: bool
) -> tuple[List[str], float]:
    """Generate text using standard autoregressive generation."""
    
    # Tokenize prompts
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    
    start_time = time.time()
    
    # Generate autoregressively
    with torch.no_grad():
        if greedy:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    generation_time = time.time() - start_time
    
    # Decode outputs
    generated_texts = []
    for i, output in enumerate(output_ids):
        # Remove input tokens to get only generated text
        generated_tokens = output[len(input_ids[i]):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts, generation_time


def interactive_chat(base_model, tokenizer, speculator, args):
    """Interactive chat mode."""
    
    print("🚀 Interactive Chat Mode")
    print("Type 'exit' or 'quit' to stop")
    print("=" * 50)
    
    stats = GenerationStats()
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in {"exit", "quit", ""}:
                break
            
            # Create prompt
            prompt = create_vicuna_prompt(user_input)
            print(f"\n🤖 Assistant: ", end="", flush=True)
            
            # Generate response
            if speculator and not args.autoregressive:
                generated_texts, gen_time = generate_with_speculator(
                    speculator, tokenizer, [prompt],
                    args.max_generation_length,
                    args.beam_width, args.beam_length,
                    args.temperature, args.greedy_search
                )
            else:
                generated_texts, gen_time = generate_autoregressive(
                    base_model, tokenizer, [prompt],
                    args.max_generation_length,
                    args.temperature, args.greedy_search
                )
            
            response = generated_texts[0]
            print(response)
            
            # Update stats
            tokens_generated = len(tokenizer.encode(response))
            stats.add_generation(tokens_generated, gen_time)
            
            # Show generation stats
            tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
            print(f"\n📊 Generated {tokens_generated} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    # Final stats
    summary = stats.get_summary()
    if summary:
        print(f"\n📈 Session Summary:")
        print(f"   Total prompts: {summary['total_prompts']}")
        print(f"   Total tokens: {summary['total_tokens']}")
        print(f"   Average tokens/sec: {summary['avg_tokens_per_second']:.1f}")


def evaluate_mt_bench(base_model, tokenizer, speculator, args):
    """Evaluate on MT-Bench dataset."""
    
    print("🔍 MT-Bench Evaluation")
    print("=" * 50)
    
    stats = GenerationStats()
    
    # Load prompts and batch them
    prompts = load_mt_bench_prompts(args.max_prompt_length, args.max_num_prompts)
    batched_prompts = batch_prompts(prompts, args.batch_size)
    
    all_outputs = []
    
    for batch in tqdm.tqdm(batched_prompts, desc="Generating"):
        try:
            # Generate responses
            if speculator and not args.autoregressive:
                generated_texts, gen_time = generate_with_speculator(
                    speculator, tokenizer, batch,
                    args.max_generation_length,
                    args.beam_width, args.beam_length,
                    args.temperature, args.greedy_search
                )
                method = "Speculative"
            else:
                generated_texts, gen_time = generate_autoregressive(
                    base_model, tokenizer, batch,
                    args.max_generation_length,
                    args.temperature, args.greedy_search
                )
                method = "Autoregressive"
            
            # Update stats
            total_tokens = sum(len(tokenizer.encode(text)) for text in generated_texts)
            stats.add_generation(total_tokens, gen_time)
            
            # Store outputs
            for prompt, response in zip(batch, generated_texts):
                all_outputs.append({
                    "prompt": prompt,
                    "response": response,
                    "method": method
                })
            
            # Show progress
            tokens_per_sec = total_tokens / gen_time if gen_time > 0 else 0
            print(f"Batch: {len(batch)} prompts, {total_tokens} tokens, {tokens_per_sec:.1f} tok/s")
            
        except Exception as e:
            print(f"⚠️ Error processing batch: {e}")
            continue
    
    # Print summary
    summary = stats.get_summary()
    print(f"\n📊 Evaluation Summary:")
    print(f"   Method: {method}")
    print(f"   Total prompts: {summary['total_prompts']}")
    print(f"   Total tokens: {summary['total_tokens']}")
    print(f"   Total time: {summary['total_time']:.2f}s")
    print(f"   Average tokens/prompt: {summary['avg_tokens_per_prompt']:.1f}")
    print(f"   Overall tokens/sec: {summary['overall_tokens_per_second']:.1f}")
    
    # Log to wandb if available
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "eval/method": method,
            "eval/total_prompts": summary['total_prompts'],
            "eval/total_tokens": summary['total_tokens'],
            "eval/tokens_per_second": summary['overall_tokens_per_second'],
            "eval/avg_tokens_per_prompt": summary['avg_tokens_per_prompt'],
        })
    
    # Save outputs if requested
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(all_outputs, f, indent=2)
        print(f"💾 Outputs saved to: {args.output_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate text using recurrent drafting speculator")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model name or path")
    parser.add_argument("--speculator_path", type=str, help="Path to trained speculator model")
    
    # Generation arguments
    parser.add_argument("--max_prompt_length", type=int, default=500, help="Maximum prompt length")
    parser.add_argument("--max_generation_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width for speculator")
    parser.add_argument("--beam_length", type=int, default=4, help="Beam length for speculator")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--greedy_search", action="store_true", help="Use greedy search")
    
    # Evaluation arguments
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--eval_mt_bench", action="store_true", help="Evaluate on MT-Bench dataset")
    parser.add_argument("--max_num_prompts", type=int, default=-1, help="Max number of prompts to evaluate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--autoregressive", action="store_true", help="Use autoregressive baseline")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda:0, etc.)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Model dtype")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, help="Save outputs to JSON file")
    parser.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="recurrent-drafting-eval", help="W&B project name")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Setup dtype
    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    
    print(f"🔧 Using device: {device}, dtype: {dtype}")
    
    # Setup wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            tags=["generation", "recurrent-drafting"]
        )
    
    # Load models
    base_model, tokenizer, speculator = load_model_and_tokenizer(
        args.base_model, args.speculator_path, device, dtype
    )
    
    # Run evaluation or interactive mode
    if args.interactive:
        interactive_chat(base_model, tokenizer, speculator, args)
    elif args.eval_mt_bench:
        evaluate_mt_bench(base_model, tokenizer, speculator, args)
    else:
        print("❌ Please specify either --interactive or --eval_mt_bench")
        return
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()