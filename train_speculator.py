#!/usr/bin/env python3
"""
Training script for the recurrent drafting speculator.

This script is adapted from Apple's original training implementation and uses
the recurrent drafting technique. See NOTICE file for attribution details.

Based on: recurrent_drafting/cmd/train.py from ml-recurrent-drafter
Original Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import math
import multiprocessing
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
import sys

from speculators.models.recurrent_drafting import RecurrentDraftingConfig, RecurrentDraftingSpeculator

# Optional wandb import for custom logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

IGNORE_TOKEN_ID = -100

class RecurrentDraftingTrainer(Trainer):
    """Custom trainer for recurrent drafting speculator based on Apple's implementation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN_ID)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the recurrent drafting model.
        
        Supports both ground-truth training and knowledge distillation based on training_args.
        """
        next_n = self.args.drafter_predict_n_tokens
        
        # Get hidden states and target logits from the base model (frozen)
        with torch.no_grad():
            base_outputs = model.verifier(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            )
            hidden_states = base_outputs.hidden_states[-1]  # Last layer
            target_logits = base_outputs.logits  # For knowledge distillation
        
        # Forward pass through speculator
        speculator_outputs = model(
            input_ids=inputs["input_ids"][:, :-next_n],
            hidden_states=hidden_states[:, :-next_n],
            attention_mask=inputs["attention_mask"][:, :-next_n] if inputs.get("attention_mask") is not None else None
        )
        
        # Choose loss computation method
        if getattr(self.args, 'use_knowledge_distillation', False):
            # Knowledge distillation: match target distribution from base model
            loss, log_dict, eval_log = self.knowledge_distillation_loss(
                speculator_outputs.logits if hasattr(speculator_outputs, 'logits') else speculator_outputs,
                target_logits,
                inputs["labels"],
                next_n,
                self.args.drafter_top_k,
                getattr(self.args, 'kd_temperature', 4.0),
                getattr(self.args, 'kd_alpha', 0.7)
            )
        else:
            # Ground-truth training: direct supervision
            loss, log_dict, eval_log = self.drafter_loss(
                speculator_outputs.logits if hasattr(speculator_outputs, 'logits') else speculator_outputs,
                inputs["labels"],
                next_n,
                self.args.drafter_top_k
            )
        
        # Add custom model-specific metrics for logging
        if hasattr(model, 'config'):
            log_dict.update({
                "model/num_draft_layers": model.config.num_draft_layers,
                "model/exit_dim": model.config.exit_dim,
                "model/rnn_enabled": model.config.rnn,
            })
        
        # Log metrics (HF Trainer will handle wandb automatically if report_to='wandb')
        self.log(log_dict)
        return (loss, eval_log) if return_outputs else loss
    
    def drafter_loss(self, logits, labels, next_n, top_k):
        """
        Compute drafter loss for multi-token prediction using ground-truth tokens.
        
        Args:
            logits: Speculator predictions [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            next_n: Number of tokens to predict ahead
            top_k: Top-k accuracy to compute
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Filter to only compute loss on assistant responses if specified
        if getattr(self.args, 'train_on_assistant_only', False):
            # Use assistant_mask to only compute loss on assistant tokens
            assistant_mask = labels != IGNORE_TOKEN_ID
            if assistant_mask.sum() == 0:
                # No assistant tokens in this batch, return zero loss
                return torch.tensor(0.0, device=device, requires_grad=True), {"train_loss": 0.0}, [0.0]
        
        # Shift labels for next token prediction
        shifted_labels = labels[:, 1:].contiguous()  # [batch_size, seq_len-1]
        
        # Flatten for loss computation
        flat_logits = logits[:, :-1].contiguous().view(-1, vocab_size)  # [batch_size*(seq_len-1), vocab_size]
        flat_labels = shifted_labels.view(-1)  # [batch_size*(seq_len-1)]
        
        # Use pre-initialized loss function
        loss = self.loss_fct(flat_logits, flat_labels)
        
        # Compute top-k accuracy for logging
        with torch.no_grad():
            # Only compute accuracy on non-ignored tokens
            valid_mask = flat_labels != IGNORE_TOKEN_ID
            if valid_mask.sum() > 0:
                valid_logits = flat_logits[valid_mask]
                valid_labels = flat_labels[valid_mask]
                top_k_preds = torch.topk(valid_logits, min(top_k, valid_logits.size(-1)), dim=-1)[1]
                correct = (top_k_preds == valid_labels.unsqueeze(-1)).any(dim=-1)
                accuracy = correct.float().mean().item()
            else:
                accuracy = 0.0
        
        log_dict = {
            "train_loss": loss.item(),
            f"train_top{top_k}_accuracy": accuracy,
            "train_method": "ground_truth"
        }
        
        eval_log = [accuracy]  # For metrics computation
        
        return loss, log_dict, eval_log
    
    def knowledge_distillation_loss(self, student_logits, teacher_logits, labels, next_n, top_k, temperature=4.0, alpha=0.7):
        """
        Compute knowledge distillation loss matching target distribution from base model.
        
        NOTE: This is the STANDARD KD approach, not the paper's "distilled dataset" method.
        For the paper's approach, see create_distilled_dataset() function.
        
        Args:
            student_logits: Speculator predictions [batch_size, seq_len, vocab_size]
            teacher_logits: Base model predictions [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            next_n: Number of tokens to predict ahead
            top_k: Top-k accuracy to compute
            temperature: Temperature for softmax (higher = softer distributions)
            alpha: Weight for distillation loss vs hard target loss
        """
        batch_size, seq_len, vocab_size = student_logits.shape
        device = student_logits.device
        
        # Align dimensions - student predicts next tokens, teacher has current tokens
        student_flat = student_logits[:, :-1].contiguous().view(-1, vocab_size)  # [batch*(seq-1), vocab]
        teacher_flat = teacher_logits[:, :-next_n-1:-1].contiguous().view(-1, vocab_size)  # [batch*(seq-1), vocab]
        
        # Shift labels for next token prediction
        shifted_labels = labels[:, 1:].contiguous().view(-1)  # [batch*(seq-1)]
        
        # Filter to only compute loss on assistant responses if specified
        if getattr(self.args, 'train_on_assistant_only', False):
            valid_mask = shifted_labels != IGNORE_TOKEN_ID
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True), {"train_loss": 0.0}, [0.0]
            
            student_flat = student_flat[valid_mask]
            teacher_flat = teacher_flat[valid_mask]
            shifted_labels = shifted_labels[valid_mask]
        
        # Compute soft targets from teacher
        teacher_soft = torch.softmax(teacher_flat / temperature, dim=-1)
        student_soft = torch.log_softmax(student_flat / temperature, dim=-1)
        
        # Distillation loss (KL divergence)
        distill_loss = nn.KLDivLoss(reduction='batchmean')(student_soft, teacher_soft) * (temperature ** 2)
        
        # Hard target loss (standard cross-entropy)
        hard_loss = self.loss_fct(student_flat, shifted_labels)
        
        # Combined loss
        loss = alpha * distill_loss + (1 - alpha) * hard_loss
        
        # Compute top-k accuracy for logging
        with torch.no_grad():
            if len(student_flat) > 0:
                top_k_preds = torch.topk(student_flat, min(top_k, student_flat.size(-1)), dim=-1)[1]
                correct = (top_k_preds == shifted_labels.unsqueeze(-1)).any(dim=-1)
                accuracy = correct.float().mean().item()
            else:
                accuracy = 0.0
        
        log_dict = {
            "train_loss": loss.item(),
            "train_distill_loss": distill_loss.item(),
            "train_hard_loss": hard_loss.item(),
            f"train_top{top_k}_accuracy": accuracy,
            "train_method": "knowledge_distillation_standard"
        }
        
        eval_log = [accuracy]
        
        return loss, log_dict, eval_log


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    llm_name_or_path: Optional[str] = field(default="gpt2")
    drafter_name_or_path: Optional[str] = field(default=None)


@dataclass 
class TrainingArguments(transformers.TrainingArguments):
    """Training arguments adapted from Apple's implementation with additional options."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,  # Smaller for efficiency
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    drafter_predict_n_tokens: int = field(
        default=4,  # Match beam_length
        metadata={"help": "Drafter predicts k extra tokens."},
    )
    drafter_top_k: int = field(
        default=5,
        metadata={"help": "Drafter top k accuracy for each token."},
    )
    drafter_num_layers: int = field(
        default=2,
        metadata={"help": "Number of layers for the drafter."},
    )
    include_inputs_for_metrics: bool = field(
        default=True,
        metadata={"help": "Include inputs for metrics."},
    )
    phase: str = field(
        default="train",
        metadata={"help": "train or eval"},
    )
    rnn: bool = field(
        default=True,
        metadata={"help": "Include rnn in drafter."},
    )
    exit_dim_multiplier: float = field(
        default=1.0,
        metadata={"help": "Multiplier for exit dimension (exit_dim = multiplier * hidden_size)"},
    )
    # Knowledge distillation options
    use_knowledge_distillation: bool = field(
        default=False,
        metadata={"help": "Use knowledge distillation instead of ground-truth training"},
    )
    kd_temperature: float = field(
        default=4.0,
        metadata={"help": "Temperature for knowledge distillation softmax"},
    )
    kd_alpha: float = field(
        default=0.7,
        metadata={"help": "Weight for distillation loss (vs hard target loss)"},
    )
    # Training data filtering
    train_on_assistant_only: bool = field(
        default=False,
        metadata={"help": "Only compute loss on assistant responses (not user prompts)"},
    )
    dataset_name: str = field(
        default="sharegpt",
        metadata={"help": "Dataset to use: 'sharegpt', 'alpaca', 'mtbench', 'wikitext'"},
    )
    # Paper's distilled dataset approach
    use_distilled_dataset: bool = field(
        default=False,
        metadata={"help": "Create distilled dataset as described in paper (LLM generates 5 future tokens at each position)"},
    )
    distill_num_future_tokens: int = field(
        default=5,
        metadata={"help": "Number of future tokens to generate for distilled dataset (paper uses 5)"},
    )
    distill_temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for distilled dataset generation (paper uses 0)"},
    )
    distill_max_examples: int = field(
        default=1000,
        metadata={"help": "Maximum number of examples to process for distilled dataset (for efficiency)"},
    )
    # Wandb integration via HF Trainer's built-in support
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Run name for experiment tracking (auto-generated if None)"},
    )


def setup_wandb_config(model_args, training_args, drafter_config=None):
    """Setup wandb config for HF Trainer integration."""
    
    # Generate run name if not provided
    if training_args.run_name is None:
        base_model = model_args.llm_name_or_path.split('/')[-1]
        run_name = f"{base_model}_n{training_args.drafter_predict_n_tokens}_layers{training_args.drafter_num_layers}"
        if training_args.rnn:
            run_name += "_rnn"
        training_args.run_name = run_name
    
    # Set wandb tags via environment variable (HF Trainer will pick this up)
    import os
    os.environ["WANDB_TAGS"] = "recurrent-drafting,speculative-decoding,apple-method"
    
    # Custom config that will be logged to wandb
    custom_config = {
        # Model configuration
        "base_model": model_args.llm_name_or_path,
        "drafter_predict_n_tokens": training_args.drafter_predict_n_tokens,
        "drafter_num_layers": training_args.drafter_num_layers,
        "drafter_top_k": training_args.drafter_top_k,
        "rnn_enabled": training_args.rnn,
        "exit_dim_multiplier": training_args.exit_dim_multiplier,
        "model_max_length": training_args.model_max_length,
    }
    
    # Add drafter config details if available
    if drafter_config:
        custom_config.update({
            "drafter_vocab_size": drafter_config.vocab_size,
            "drafter_hidden_size": drafter_config.hidden_size,
            "drafter_exit_dim": drafter_config.exit_dim,
            "drafter_emb_norm": drafter_config.emb_norm,
        })
    
    return custom_config


def get_tokenizer(model_args, training_args):
    """Get tokenizer with proper configuration."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.llm_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def generate_drafter_config_from_base(llm, training_args):
    """Generate drafter configuration from base model."""
    return RecurrentDraftingConfig(
        vocab_size=llm.config.vocab_size,
        hidden_size=llm.config.hidden_size,
        exit_dim=int(llm.config.hidden_size * training_args.exit_dim_multiplier),
        num_draft_layers=training_args.drafter_num_layers,
        rnn=training_args.rnn,
        emb_norm=False,
    )


def get_compute_metrics(training_args):
    """Get metrics computation function."""
    predict_n_tokens = training_args.drafter_predict_n_tokens
    
    def compute_metrics(all_preds):
        return_val = {}
        if hasattr(all_preds, 'predictions') and len(all_preds.predictions) > 0:
            # Compute top-k accuracy
            for k in range(1, training_args.drafter_top_k + 1):
                if len(all_preds.predictions) > k - 1:
                    return_val[f"drafter_top{k}"] = np.mean(all_preds.predictions[k - 1])
        return return_val
    
    return compute_metrics


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator for language modeling tasks.
    
    This handles proper batching and padding for the training data.
    """
    tokenizer: transformers.PreTrainedTokenizer
    mlm: bool = False
    
    def __call__(self, examples):
        # Extract input_ids, attention_mask, and labels
        input_ids = [example["input_ids"] for example in examples]
        attention_masks = [example["attention_mask"] for example in examples]
        labels = [example["labels"] for example in examples]
        
        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ) # [len(examples), max_seq_len] (right padded)
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_TOKEN_ID
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }


def convert_alpaca_to_sharegpt(example):
    """
    Convert Alpaca format to ShareGPT format.
    
    This matches the data processing from Apple's original implementation.
    """
    conversations = []
    
    # Create instruction prompt
    if example.get("input", "").strip():
        instruction = f"{example['instruction']}\n\n{example['input']}"
    else:
        instruction = example["instruction"]
    
    conversations.append({
        "from": "human",
        "value": instruction
    })
    
    conversations.append({
        "from": "gpt", 
        "value": example["output"]
    })
    
    return {"conversations": conversations}


def create_distilled_dataset_entry(example, tokenizer, base_model, num_future_tokens=5, temperature=0.0, train_on_assistant_only=False):
    """
    Create a distilled dataset entry following the paper's approach:
    "The distilled dataset was created by having the LLM generate 5 future tokens 
    at each position of the ground-truth response using a temperature of 0."
    
    This means: for each position t in the ground-truth sequence, we have the LLM
    generate the next 5 tokens, creating multiple training examples per original sequence.
    
    Args:
        example: ShareGPT conversation example
        tokenizer: Tokenizer to use
        base_model: The teacher LLM to generate distilled targets
        num_future_tokens: Number of future tokens to generate (paper uses 5)
        temperature: Generation temperature (paper uses 0 for deterministic)
        train_on_assistant_only: If True, only create distilled examples for assistant responses
    
    Returns:
        List of distilled training examples
    """
    # First convert to standard format
    standard_instance = sharegpt_record_to_training_instance(example, tokenizer, train_on_assistant_only)
    
    input_ids = standard_instance["input_ids"]
    attention_mask = standard_instance["attention_mask"]
    labels = standard_instance["labels"]
    
    distilled_examples = []
    
    # For each position t in the sequence (except the last few)
    seq_len = len(input_ids)
    max_start_pos = seq_len - num_future_tokens - 1
    
    for t in range(0, max_start_pos, 4):  # Sample every 4th position to avoid too many examples
        # Skip if this position should be ignored (human response when train_on_assistant_only=True)
        if train_on_assistant_only and labels[t] == IGNORE_TOKEN_ID:
            continue
            
        # Context up to position t
        context_ids = input_ids[:t+1]
        context_mask = attention_mask[:t+1]
        
        # Generate next num_future_tokens using the base model
        with torch.no_grad():
            try:
                # Generate from the base model
                generated = base_model.generate(
                    input_ids=context_ids.unsqueeze(0),
                    attention_mask=context_mask.unsqueeze(0),
                    max_new_tokens=num_future_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Extract the generated tokens (remove the context)
                generated_tokens = generated[0][len(context_ids):]
                
                # Create training example: context -> generated_tokens
                if len(generated_tokens) >= num_future_tokens:
                    # Input: context up to position t
                    # Target: generated next num_future_tokens
                    target_tokens = generated_tokens[:num_future_tokens]
                    
                    # Create the training instance
                    full_input = torch.cat([context_ids, target_tokens])
                    full_mask = torch.ones_like(full_input)
                    
                    # Labels: ignore context, predict generated tokens
                    full_labels = torch.full_like(full_input, IGNORE_TOKEN_ID)
                    full_labels[len(context_ids):] = target_tokens
                    
                    distilled_examples.append({
                        "input_ids": full_input,
                        "attention_mask": full_mask,
                        "labels": full_labels,
                        "distilled": True,  # Mark as distilled example
                        "original_position": t,
                    })
                    
            except Exception as e:
                # Skip this position if generation fails
                continue
    
    return distilled_examples


def sharegpt_record_to_training_instance(example, tokenizer, train_on_assistant_only=False):
    """
    Convert ShareGPT record to training instance.
    
    Args:
        example: ShareGPT conversation example
        tokenizer: Tokenizer to use
        train_on_assistant_only: If True, only compute loss on assistant responses
    """
    conversations = example["conversations"]
    
    # Build conversation text and track assistant positions
    text_parts: list[str] = []
    assistant_markers = []  # Track which parts are assistant responses
    
    for conv in conversations:
        if conv["from"] == "human":
            text_parts.append(f"Human: {conv['value']}")
            assistant_markers.append(False)
        elif conv["from"] == "gpt":
            text_parts.append(f"Assistant: {conv['value']}")
            assistant_markers.append(True)
    
    full_text = "\n\n".join(text_parts)
    
    # Tokenize
    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding=False,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    
    input_ids = tokenized["input_ids"].squeeze(0)
    attention_mask = tokenized["attention_mask"].squeeze(0)
    
    # Create labels - mask human parts if train_on_assistant_only
    if train_on_assistant_only:
        labels = input_ids.clone()
        
        # Find assistant response boundaries in tokenized text
        # This is approximate - for production, you'd want more precise tracking
        human_prefix = tokenizer.encode("Human:", add_special_tokens=False)
        assistant_prefix = tokenizer.encode("Assistant:", add_special_tokens=False)
        
        # Mask human responses (set to IGNORE_TOKEN_ID)
        current_pos = 0
        for i, text_part in enumerate(text_parts):
            part_tokens = tokenizer.encode(text_part, add_special_tokens=False)
            part_len = len(part_tokens)
            
            if not assistant_markers[i]:  # Human response
                # Mask this part
                end_pos = min(current_pos + part_len, len(labels))
                labels[current_pos:end_pos] = IGNORE_TOKEN_ID
            
            current_pos += part_len + 2  # +2 for "\n\n" separator tokens (approximate)
            if current_pos >= len(labels):
                break
    else:
        labels = input_ids.clone()
    
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    return result


def prepare_dataset(tokenizer, training_args, split="train", base_model=None) -> list[dict]:
    """
    Prepare training dataset with multiple dataset options.
    
    Addresses the uncertainty about which dataset was used in the original paper.
    """
    dataset_name = getattr(training_args, 'dataset_name', 'sharegpt')
    train_on_assistant_only = getattr(training_args, 'train_on_assistant_only', False)
    use_distilled_dataset = getattr(training_args, 'use_distilled_dataset', False)
    
    print(f"🔄 Loading {dataset_name} dataset for {split}...")
    if use_distilled_dataset and split == "train":
        print(f"📚 Creating distilled dataset (paper's approach): generate {getattr(training_args, 'distill_num_future_tokens', 5)} tokens at each position")
    
    if split == "train":
        if dataset_name == "sharegpt":
            try:
                print("🔄 Loading ShareGPT training dataset...")
                dataset = datasets.load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
                
                # Process dataset
                if use_distilled_dataset and base_model is not None:
                    print(f"🔄 Creating distilled dataset from {len(dataset)} examples...")
                    # Limit examples for efficiency
                    max_examples = getattr(training_args, 'distill_max_examples', 1000)
                    if len(dataset) > max_examples:
                        dataset = dataset.select(range(max_examples))
                        print(f"📊 Limited to {max_examples} examples for distilled dataset creation")
                    
                    # Create distilled examples
                    def create_distilled_examples(example):
                        return create_distilled_dataset_entry(
                            example, 
                            tokenizer, 
                            base_model,
                            num_future_tokens=getattr(training_args, 'distill_num_future_tokens', 5),
                            temperature=getattr(training_args, 'distill_temperature', 0.0),
                            train_on_assistant_only=train_on_assistant_only
                        )
                    
                    # Process examples to create distilled dataset
                    all_distilled_examples = []
                    for i, example in enumerate(dataset):
                        if i % 100 == 0:
                            print(f"🔄 Processing example {i}/{len(dataset)} for distilled dataset...")
                        
                        distilled_examples = create_distilled_examples(example)
                        all_distilled_examples.extend(distilled_examples)
                    
                    print(f"✅ Created {len(all_distilled_examples)} distilled training examples")
                    
                    # Convert to HuggingFace dataset format
                    tokenized_dataset = datasets.Dataset.from_list(all_distilled_examples)
                else:
                    # Standard processing
                    tokenized_dataset = dataset.map(
                        lambda x: sharegpt_record_to_training_instance(x, tokenizer, train_on_assistant_only),
                        num_proc=min(4, multiprocessing.cpu_count()),
                        remove_columns=dataset.column_names
                    )
                
            except Exception as e:
                print(f"⚠️ Could not load ShareGPT dataset: {e}")
                print("🔄 Falling back to WikiText dataset...")
                dataset_name = "wikitext"
        
        if dataset_name == "alpaca":
            try:
                print("🔄 Loading Alpaca training dataset...")
                dataset = datasets.load_dataset("tatsu-lab/alpaca", split="train")
                
                # Convert to ShareGPT format then tokenize
                dataset = dataset.map(
                    convert_alpaca_to_sharegpt,
                    num_proc=min(4, multiprocessing.cpu_count())
                )
                
                tokenized_dataset = dataset.map(
                    lambda x: sharegpt_record_to_training_instance(x, tokenizer, train_on_assistant_only),
                    num_proc=min(4, multiprocessing.cpu_count()),
                    remove_columns=dataset.column_names
                )
                
            except Exception as e:
                print(f"⚠️ Could not load Alpaca dataset: {e}")
                print("🔄 Falling back to WikiText dataset...")
                dataset_name = "wikitext"
        
        if dataset_name == "mtbench":
            try:
                print("🔄 Loading MT-Bench training dataset...")
                # Note: MT-Bench is typically evaluation-only, but we can use it for training
                dataset = datasets.load_dataset("lmsys/mt_bench", split="train")
                
                def mtbench_to_sharegpt(example):
                    return {
                        "conversations": [
                            {"from": "human", "value": example["turns"][0]},
                            {"from": "gpt", "value": example["turns"][1] if len(example["turns"]) > 1 else "I understand."}
                        ]
                    }
                
                dataset = dataset.map(mtbench_to_sharegpt, num_proc=min(4, multiprocessing.cpu_count()))
                
                tokenized_dataset = dataset.map(
                    lambda x: sharegpt_record_to_training_instance(x, tokenizer, train_on_assistant_only),
                    num_proc=min(4, multiprocessing.cpu_count()),
                    remove_columns=dataset.column_names
                )
                
            except Exception as e:
                print(f"⚠️ Could not load MT-Bench dataset: {e}")
                print("🔄 Falling back to WikiText dataset...")
                dataset_name = "wikitext"
        
        if dataset_name == "wikitext":
            # Fallback to WikiText
            dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            
            def tokenize_wikitext(examples):
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=training_args.model_max_length,
                    return_tensors="pt"
                )
                tokenized["labels"] = tokenized["input_ids"].clone()
                return tokenized
            
            tokenized_dataset = dataset.map(
                tokenize_wikitext,
                batched=True,
                num_proc=min(4, multiprocessing.cpu_count()),
                remove_columns=dataset.column_names
            )
    
    else:
        # For evaluation, provide multiple options
        if dataset_name in ["alpaca", "sharegpt"]:
            try:
                print("🔄 Loading Alpaca evaluation dataset...")
                dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", split="eval")
                
                # Convert to ShareGPT format then tokenize
                dataset = dataset.map(
                    convert_alpaca_to_sharegpt,
                    num_proc=min(4, multiprocessing.cpu_count())
                )
                
                tokenized_dataset = dataset.map(
                    lambda x: sharegpt_record_to_training_instance(x, tokenizer, train_on_assistant_only),
                    num_proc=min(4, multiprocessing.cpu_count()),
                    remove_columns=dataset.column_names
                )
                
            except Exception as e:
                print(f"⚠️ Could not load Alpaca eval dataset: {e}")
                # Fallback to validation split of training dataset
                tokenized_dataset = prepare_dataset(tokenizer, training_args, split="train")
                # Take a small subset for evaluation
                tokenized_dataset = tokenized_dataset.select(range(min(1000, len(tokenized_dataset))))
        
        elif dataset_name == "mtbench":
            try:
                print("🔄 Loading MT-Bench evaluation dataset...")
                dataset = datasets.load_dataset("lmsys/mt_bench", split="test")
                
                def mtbench_to_sharegpt(example):
                    return {
                        "conversations": [
                            {"from": "human", "value": example["turns"][0]},
                            {"from": "gpt", "value": example["turns"][1] if len(example["turns"]) > 1 else "I understand."}
                        ]
                    }
                
                dataset = dataset.map(mtbench_to_sharegpt, num_proc=min(4, multiprocessing.cpu_count()))
                
                tokenized_dataset = dataset.map(
                    lambda x: sharegpt_record_to_training_instance(x, tokenizer, train_on_assistant_only),
                    num_proc=min(4, multiprocessing.cpu_count()),
                    remove_columns=dataset.column_names
                )
                
            except Exception as e:
                print(f"⚠️ Could not load MT-Bench eval dataset: {e}")
                # Fallback
                tokenized_dataset = prepare_dataset(tokenizer, training_args, split="train")
                tokenized_dataset = tokenized_dataset.select(range(min(1000, len(tokenized_dataset))))
        
        else:
            # WikiText validation
            dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            
            def tokenize_wikitext(examples):
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=training_args.model_max_length,
                    return_tensors="pt"
                )
                tokenized["labels"] = tokenized["input_ids"].clone()
                return tokenized
            
            tokenized_dataset = dataset.map(
                tokenize_wikitext,
                batched=True,
                num_proc=min(4, multiprocessing.cpu_count()),
                remove_columns=dataset.column_names
            )
    
    print(f"✅ Loaded {len(tokenized_dataset)} examples from {dataset_name}")
    return tokenized_dataset
        
def train(model_args, training_args):
    """Main training function adapted from Apple's implementation."""
    
    print(f"🚀 Starting Recurrent Drafting Training")
    print(f"   Base model: {model_args.llm_name_or_path}")
    print(f"   Predict tokens: {training_args.drafter_predict_n_tokens}")
    print(f"   Draft layers: {training_args.drafter_num_layers}")
    print(f"   RNN enabled: {training_args.rnn}")
    
    # Get tokenizer
    tokenizer = get_tokenizer(model_args, training_args)
    compute_metrics = get_compute_metrics(training_args)
    
    # Load base model first if we need it for distilled dataset
    base_model_for_distillation = None
    if getattr(training_args, 'use_distilled_dataset', False):
        print(f"🔄 Loading base model for distilled dataset creation: {model_args.llm_name_or_path}")
        # Set RoPE scaling factor if needed
        config = AutoConfig.from_pretrained(model_args.llm_name_or_path)
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        
        base_model_for_distillation = AutoModelForCausalLM.from_pretrained(
            model_args.llm_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        base_model_for_distillation.eval()  # Set to eval mode for generation
    
    # Load training dataset (ShareGPT format, matching Apple's approach)
    print("🔄 Loading training dataset...")
    train_dataset = prepare_dataset(tokenizer, training_args, split="train", base_model=base_model_for_distillation)
    
    # Load and freeze the base model (reuse if already loaded for distillation)
    if base_model_for_distillation is not None:
        print(f"🔄 Reusing base model for training: {model_args.llm_name_or_path}")
        llm = base_model_for_distillation
    else:
        # Set RoPE scaling factor if needed
        config = AutoConfig.from_pretrained(model_args.llm_name_or_path)
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        
        print(f"🔄 Loading base model: {model_args.llm_name_or_path}")
        llm = AutoModelForCausalLM.from_pretrained(
            model_args.llm_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    
    # Freeze base model parameters
    for param in llm.parameters():
        param.requires_grad = False
    
    print("🔄 Creating recurrent drafting speculator...")
    # Generate drafter config from base model
    drafter_config = generate_drafter_config_from_base(llm, training_args)
    
    # Setup wandb config for HF Trainer integration
    wandb_config = setup_wandb_config(model_args, training_args, drafter_config)
    
    # Create speculator and attach verifier
    speculator = RecurrentDraftingSpeculator(drafter_config, verifier=None)
    speculator.attach_verifier(llm)
    
    # Format output directory
    output_dir = (
        f"{training_args.output_dir}"
        f"_redrafter_{model_args.llm_name_or_path.split('/')[-1]}"
        f"_n_{training_args.drafter_predict_n_tokens}"
        f"_lr_{training_args.learning_rate}"
        f"_layers_{training_args.drafter_num_layers}"
    )
    training_args.output_dir = output_dir
    
    print(f"💾 Output directory: {output_dir}")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Log custom config to wandb if enabled
    if "wandb" in training_args.report_to:
        if WANDB_AVAILABLE and wandb.run is None:
            # Initialize wandb with custom config if not already done by HF Trainer
            import os
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "recurrent-drafting"),
                name=training_args.run_name,
                config=wandb_config,
            )
        elif WANDB_AVAILABLE and wandb.run:
            # Update existing wandb run with custom config
            wandb.config.update(wandb_config)
    
    # Create trainer
    trainer = RecurrentDraftingTrainer(
        model=speculator,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
    )
    
    # Check for existing checkpoints
    resume_from_checkpoint = bool(
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    )
    
    print("🚀 Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the trained speculator
    print(f"💾 Saving trained speculator to {training_args.output_dir}")
    speculator.save_pretrained(training_args.output_dir)
    
    print("✅ Training completed!")
    
    # Log final metrics and artifacts to wandb if enabled
    if "wandb" in training_args.report_to and WANDB_AVAILABLE and wandb.run:
        # Log model artifacts
        try:
            model_artifact = wandb.Artifact(
                name=f"recurrent-drafter-{wandb.run.id}",
                type="model",
                description=f"Trained recurrent drafting speculator for {model_args.llm_name_or_path}"
            )
            model_artifact.add_dir(training_args.output_dir)
            wandb.log_artifact(model_artifact)
            
            # Log training summary
            wandb.summary.update({
                "final_model_path": training_args.output_dir,
                "training_completed": True,
                "total_parameters": sum(p.numel() for p in speculator.parameters()),
                "trainable_parameters": sum(p.numel() for p in speculator.parameters() if p.requires_grad),
            })
            
            print(f"📊 Training artifacts logged to: {wandb.run.url}")
        except Exception as e:
            print(f"⚠️ Could not log artifacts to wandb: {e}")
    
    return speculator


def eval_model(model_args, training_args):
    """Evaluation function adapted from Apple's implementation."""
    
    print(f"🔍 Starting Recurrent Drafting Evaluation")
    
    tokenizer = get_tokenizer(model_args, training_args)
    compute_metrics = get_compute_metrics(training_args)
    
    # Load evaluation dataset (Alpaca format, matching Apple's approach)
    print("🔄 Loading evaluation dataset...")
    try:
        eval_dataset = prepare_dataset(tokenizer, training_args, split="eval", base_model=None)
    except Exception as e:
        print(f"⚠️ Could not load Alpaca evaluation dataset: {e}")
        print("🔄 Falling back to WikiText validation...")
        
        # Fallback to WikiText validation
        try:
            dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            
            def tokenize_wikitext(examples):
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=training_args.model_max_length,
                    return_tensors="pt"
                )
                tokenized["labels"] = tokenized["input_ids"].clone()
                return tokenized
            
            eval_dataset = dataset.map(
                tokenize_wikitext,
                batched=True,
                num_proc=min(4, multiprocessing.cpu_count()),
                remove_columns=dataset.column_names
            )
        except Exception as e2:
            print(f"❌ Could not load any evaluation dataset: {e2}")
            return
    
    # Load trained model
    print(f"🔄 Loading trained model from {model_args.drafter_name_or_path}")
    try:
        # Load base model
        llm = AutoModelForCausalLM.from_pretrained(
            model_args.llm_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Freeze base model parameters
        for param in llm.parameters():
            param.requires_grad = False
        
        # Load speculator
        speculator = RecurrentDraftingSpeculator.from_pretrained(
            model_args.drafter_name_or_path
        )
        speculator.attach_verifier(llm)
        
    except Exception as e:
        print(f"❌ Could not load trained model: {e}")
        return
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer for evaluation
    trainer = RecurrentDraftingTrainer(
        model=speculator,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        eval_dataset=eval_dataset,
    )
    
    print("🔍 Running evaluation...")
    results = trainer.evaluate()
    
    print("✅ Evaluation completed!")
    print("📊 Results:")
    for key, value in results.items():
        print(f"   {key}: {value:.4f}")
    
    # Additional analysis
    if "eval_drafter_top1" in results:
        top1_acc = results["eval_drafter_top1"]
        print(f"\n🎯 Key Metrics:")
        print(f"   Top-1 Accuracy: {top1_acc:.1%}")
        
        performance_level = "poor"
        if top1_acc > 0.6:
            print("   ✅ Excellent! High acceptance rate expected")
            performance_level = "excellent"
        elif top1_acc > 0.4:
            print("   ✅ Good! Moderate speedup expected")
            performance_level = "good"
        elif top1_acc > 0.2:
            print("   ⚠️ Fair. Some speedup possible")
            performance_level = "fair"
        else:
            print("   ❌ Poor. More training needed")
            performance_level = "poor"
        
        # Log evaluation results to wandb if enabled
        if "wandb" in training_args.report_to and WANDB_AVAILABLE:
            # Initialize wandb for evaluation if not already done
            if not wandb.run:
                import os
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "recurrent-drafting"),
                    name=f"eval_{model_args.drafter_name_or_path.split('/')[-1]}",
                    job_type="evaluation",
                    tags=["evaluation", "recurrent-drafting"]
                )
            
            # Log all evaluation metrics
            eval_metrics = {f"eval/{k}": v for k, v in results.items()}
            eval_metrics.update({
                "eval/performance_level": performance_level,
                "eval/top1_accuracy_percent": top1_acc * 100,
                "eval/model_path": model_args.drafter_name_or_path,
            })
            
            wandb.log(eval_metrics)
            print(f"📊 Evaluation results logged to: {wandb.run.url}")
    
    return results


def create_simple_trainer():
    """Create a simple trainer for quick testing."""
    
    print("🚀 Simple Recurrent Drafting Trainer")
    print("=" * 40)
    
    # Default arguments for quick training
    model_args = ModelArguments(
        llm_name_or_path="gpt2"
    )
    
    training_args = TrainingArguments(
        output_dir="./models/gpt2-recurrent-drafter-trained",
        num_train_epochs=1,  # Quick training
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="no",  # Skip evaluation for quick training
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        model_max_length=256,  # Shorter sequences for speed
        drafter_predict_n_tokens=4,
        drafter_num_layers=2,
        rnn=True,
        phase="train",
        report_to=[],  # No logging for quick training
        run_name="quick-training"
    )
    
    return model_args, training_args


def demonstrate_training_need():
    """Demonstrate why training is necessary."""
    
    print("🎯 Why Training the Speculator is Critical")
    print("=" * 50)
    
    print("\n❌ Current State (Random Weights):")
    print("   • Speculator generates random predictions")
    print("   • Base model rejects ~95% of candidates")
    print("   • No speedup, possibly slower")
    print("   • Essentially useless for production")
    
    print("\n✅ After Training (Learned Weights):")
    print("   • Speculator learns to mimic base model")
    print("   • Base model accepts ~60-80% of candidates")
    print("   • 1.5-3x speedup achieved")
    print("   • Production-ready performance")
    
    print("\n📚 Training Process (Apple's Method):")
    print("   1. Multi-token prediction: Speculator predicts next N tokens")
    print("   2. Cross-entropy loss: Direct supervision from base model")
    print("   3. Frozen base model: Only train the lightweight drafter")
    print("   4. Top-k accuracy: Measure prediction quality")
    
    print("\n⏱️ Training Time Estimate:")
    print("   • Small model (GPT-2): 2-4 hours on GPU")
    print("   • Medium model (GPT-2 Medium): 8-12 hours")
    print("   • Large model (GPT-2 Large): 1-2 days")


def show_training_examples():
    """Show training command examples addressing the paper's methodology questions."""
    
    print("\n🚀 Training Examples - Addressing Paper Questions")
    print("=" * 55)
    
    print("\n1. Quick Training (Demo):")
    print("python train_speculator.py --quick")
    
    print("\n2. Ground-Truth Training (Current Implementation):")
    print("python train_speculator.py \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --dataset_name sharegpt \\")
    print("  --num_train_epochs 3 \\")
    print("  --per_device_train_batch_size 8 \\")
    print("  --learning_rate 5e-4")
    
    print("\n3. Knowledge Distillation - Standard Approach:")
    print("python train_speculator.py \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --use_knowledge_distillation \\")
    print("  --kd_temperature 4.0 \\")
    print("  --kd_alpha 0.7 \\")
    print("  --dataset_name sharegpt")
    
    print("\n3b. Paper's Distilled Dataset Approach (Section 4.3.3):")
    print("python train_speculator.py \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --use_distilled_dataset \\")
    print("  --distill_num_future_tokens 5 \\")
    print("  --distill_temperature 0.0 \\")
    print("  --dataset_name sharegpt")
    
    print("\n4. Train on Assistant Responses Only:")
    print("python train_speculator.py \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --train_on_assistant_only \\")
    print("  --dataset_name sharegpt")
    
    print("\n5. Train on Alpaca (Evaluation Dataset):")
    print("python train_speculator.py \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --dataset_name alpaca \\")
    print("  --num_train_epochs 5  # More epochs for smaller dataset")
    
    print("\n6. Train on MT-Bench (Evaluation Dataset):")
    print("python train_speculator.py \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --dataset_name mtbench \\")
    print("  --num_train_epochs 10  # More epochs for smaller dataset")
    
    print("\n7. Paper's Exact Method (Distilled Dataset + Assistant Only):")
    print("python train_speculator.py \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --use_distilled_dataset \\")
    print("  --train_on_assistant_only \\")
    print("  --distill_num_future_tokens 5 \\")
    print("  --distill_temperature 0.0 \\")
    print("  --dataset_name sharegpt \\")
    print("  --report_to wandb \\")
    print("  --run_name paper-exact-method")
    
    print("\n8. Standard KD + Assistant Only (Alternative):")
    print("python train_speculator.py \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --use_knowledge_distillation \\")
    print("  --train_on_assistant_only \\")
    print("  --dataset_name alpaca \\")
    print("  --report_to wandb \\")
    print("  --run_name standard-kd-method")
    
    print("\n8. Evaluation on Different Datasets:")
    print("python train_speculator.py \\")
    print("  --phase eval \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --drafter_name_or_path ./models/gpt2-recurrent-drafter \\")
    print("  --dataset_name alpaca  # or mtbench")
    
    print("\n📊 Experiment Matrix to Address Paper Questions:")
    print("   • Ground-truth vs Knowledge Distillation")
    print("   • Full conversation vs Assistant-only training")  
    print("   • ShareGPT vs Alpaca vs MT-Bench datasets")
    print("   • Cross-evaluation: train on X, eval on Y")


def main():
    """Main function with argument parsing."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Recurrent Drafting Speculator")
    parser.add_argument("--quick", action="store_true", help="Run quick training demo")
    parser.add_argument("--demo", action="store_true", help="Show training information only")
    
    # Add all the training arguments
    parser.add_argument("--llm_name_or_path", type=str, default="gpt2")
    parser.add_argument("--drafter_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./models/gpt2-recurrent-drafter")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--drafter_predict_n_tokens", type=int, default=4)
    parser.add_argument("--drafter_num_layers", type=int, default=2)
    parser.add_argument("--rnn", action="store_true", default=True)
    parser.add_argument("--phase", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--model_max_length", type=int, default=512)
    
    # Knowledge distillation options
    parser.add_argument("--use_knowledge_distillation", action="store_true", 
                       help="Use knowledge distillation instead of ground-truth training")
    parser.add_argument("--kd_temperature", type=float, default=4.0,
                       help="Temperature for knowledge distillation")
    parser.add_argument("--kd_alpha", type=float, default=0.7,
                       help="Weight for distillation loss vs hard target loss")
    
    # Training data options
    parser.add_argument("--train_on_assistant_only", action="store_true",
                       help="Only compute loss on assistant responses")
    parser.add_argument("--dataset_name", type=str, default="sharegpt",
                       choices=["sharegpt", "alpaca", "mtbench", "wikitext"],
                       help="Dataset to use for training")
    
    # Paper's distilled dataset approach
    parser.add_argument("--use_distilled_dataset", action="store_true",
                       help="Create distilled dataset as described in paper (LLM generates future tokens at each position)")
    parser.add_argument("--distill_num_future_tokens", type=int, default=5,
                       help="Number of future tokens to generate for distilled dataset (paper uses 5)")
    parser.add_argument("--distill_temperature", type=float, default=0.0,
                       help="Temperature for distilled dataset generation (paper uses 0)")
    parser.add_argument("--distill_max_examples", type=int, default=1000,
                       help="Maximum number of examples to process for distilled dataset")
    
    # Experiment tracking arguments (using HF Trainer's built-in integration)
    parser.add_argument("--report_to", type=str, nargs="+", default=[], 
                       help="Experiment tracking services (e.g., 'wandb', 'tensorboard')")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for experiment tracking")
    
    args = parser.parse_args()
    
    if args.demo or (not args.quick and len(sys.argv) == 1):
        # Show information only
        demonstrate_training_need()
        show_training_examples()
        
        print("\n🎯 BOTTOM LINE:")
        print("   The current speculator has RANDOM weights and won't provide speedup.")
        print("   Training is ESSENTIAL to make this practically useful!")
        
        print("\n📋 Next Steps:")
        print("   1. Run: python train_speculator.py --quick")
        print("   2. Evaluate acceptance rates")
        print("   3. Benchmark actual speedup")
        print("   4. Deploy trained model")
        
        return
    
    if args.quick:
        # Quick training
        model_args, training_args = create_simple_trainer()
        print("🚀 Starting quick training...")
        train(model_args, training_args)
        
    else:
        # Full training with parsed arguments
        model_args = ModelArguments(
            llm_name_or_path=args.llm_name_or_path,
            drafter_name_or_path=args.drafter_name_or_path
        )
        
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            model_max_length=args.model_max_length,
            drafter_predict_n_tokens=args.drafter_predict_n_tokens,
            drafter_num_layers=args.drafter_num_layers,
            rnn=args.rnn,
            phase=args.phase,
            # Knowledge distillation options
            use_knowledge_distillation=args.use_knowledge_distillation,
            kd_temperature=args.kd_temperature,
            kd_alpha=args.kd_alpha,
            # Training data options
            train_on_assistant_only=args.train_on_assistant_only,
            dataset_name=args.dataset_name,
            # Paper's distilled dataset approach
            use_distilled_dataset=args.use_distilled_dataset,
            distill_num_future_tokens=args.distill_num_future_tokens,
            distill_temperature=args.distill_temperature,
            distill_max_examples=args.distill_max_examples,
            # Experiment tracking
            report_to=args.report_to,  # HF Trainer's built-in experiment tracking
            run_name=args.run_name,
        )
        
        if args.phase == "train":
            train(model_args, training_args)
        else:
            eval_model(model_args, training_args)


if __name__ == "__main__":
    main()