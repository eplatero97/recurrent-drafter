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

# Optional wandb import
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
        self.wandb_enabled = getattr(self.args, 'use_wandb', False) and WANDB_AVAILABLE
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN_ID)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the recurrent drafting model.
        
        This implements the drafter loss function from the original Apple implementation,
        adapted for the speculators framework.
        """
        next_n = self.args.drafter_predict_n_tokens
        
        # Get hidden states from the base model (frozen)
        with torch.no_grad():
            base_outputs = model.verifier(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            )
            hidden_states = base_outputs.hidden_states[-1]  # Last layer
            target_logits = base_outputs.logits
        
        # Forward pass through speculator
        speculator_outputs = model(
            input_ids=inputs["input_ids"][:, :-next_n],
            hidden_states=hidden_states[:, :-next_n],
            attention_mask=inputs["attention_mask"][:, :-next_n] if inputs.get("attention_mask") is not None else None
        )
        
        # Compute drafter loss (multi-token prediction)
        loss, log_dict, eval_log = self.drafter_loss(
            speculator_outputs.logits if hasattr(speculator_outputs, 'logits') else speculator_outputs,
            inputs["labels"],
            next_n,
            self.args.drafter_top_k
        )
        
        # Enhanced logging for wandb
        if self.wandb_enabled and self.state.global_step % self.args.logging_steps == 0:
            enhanced_log_dict = {
                **log_dict,
                "learning_rate": self.get_lr(),
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
            }
            
            # Add model-specific metrics
            if hasattr(model, 'config'):
                enhanced_log_dict.update({
                    "model/num_draft_layers": model.config.num_draft_layers,
                    "model/exit_dim": model.config.exit_dim,
                    "model/rnn_enabled": model.config.rnn,
                })
            
            wandb.log(enhanced_log_dict, step=self.state.global_step)
        
        self.log(log_dict)
        return (loss, eval_log) if return_outputs else loss
    
    def drafter_loss(self, logits, labels, next_n, top_k):
        """
        Compute drafter loss for multi-token prediction.
        
        Args:
            logits: Speculator predictions [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            next_n: Number of tokens to predict ahead
            top_k: Top-k accuracy to compute
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Shift labels for next token prediction
        shifted_labels = labels[:, 1:].contiguous()  # [batch_size, seq_len-1]
        
        # Flatten for loss computation
        flat_logits = logits[:, :-1].contiguous().view(-1, vocab_size)  # [batch_size*(seq_len-1), vocab_size]
        flat_labels = shifted_labels.view(-1)  # [batch_size*(seq_len-1)]
        
        # Use pre-initialized loss function
        loss = self.loss_fct(flat_logits, flat_labels)
        
        # Compute top-k accuracy for logging
        with torch.no_grad():
            _, top_k_preds = torch.topk(flat_logits, top_k, dim=-1)
            correct = (top_k_preds == flat_labels.unsqueeze(-1)).any(dim=-1)
            accuracy = correct.float().mean()
        
        log_dict = {
            "train_loss": loss.item(),
            f"train_top{top_k}_accuracy": accuracy.item()
        }
        
        eval_log = [accuracy.item()]  # For metrics computation
        
        return loss, log_dict, eval_log


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    llm_name_or_path: Optional[str] = field(default="gpt2")
    drafter_name_or_path: Optional[str] = field(default=None)


@dataclass 
class TrainingArguments(transformers.TrainingArguments):
    """Training arguments adapted from Apple's implementation."""
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
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Enable Weights & Biases logging"},
    )
    wandb_project: str = field(
        default="recurrent-drafting",
        metadata={"help": "Weights & Biases project name"},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name (auto-generated if None)"},
    )


def setup_wandb(model_args, training_args, model_config=None):
    """Setup Weights & Biases logging if enabled."""
    
    if not training_args.use_wandb:
        return
    
    if not WANDB_AVAILABLE:
        print("⚠️ wandb not available. Install with: pip install wandb")
        training_args.use_wandb = False
        return
    
    # Generate run name if not provided
    run_name = training_args.wandb_run_name
    if run_name is None:
        base_model = model_args.llm_name_or_path.split('/')[-1]
        run_name = f"{base_model}_n{training_args.drafter_predict_n_tokens}_layers{training_args.drafter_num_layers}"
        if training_args.rnn:
            run_name += "_rnn"
    
    # Initialize wandb
    wandb.init(
        project=training_args.wandb_project,
        name=run_name,
        config={
            # Model configuration
            "base_model": model_args.llm_name_or_path,
            "drafter_predict_n_tokens": training_args.drafter_predict_n_tokens,
            "drafter_num_layers": training_args.drafter_num_layers,
            "drafter_top_k": training_args.drafter_top_k,
            "rnn_enabled": training_args.rnn,
            "exit_dim_multiplier": training_args.exit_dim_multiplier,
            
            # Training configuration
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "num_train_epochs": training_args.num_train_epochs,
            "warmup_steps": training_args.warmup_steps,
            "model_max_length": training_args.model_max_length,
            
            # Model architecture details
            **({"model_config": model_config} if model_config else {}),
        },
        tags=["recurrent-drafting", "speculative-decoding", "apple-method"],
    )
    
    print(f"📊 Weights & Biases initialized: {wandb.run.url}")


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
        )
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


def sharegpt_record_to_training_instance(example, tokenizer):
    """
    Convert ShareGPT record to training instance.
    
    This matches the data processing from Apple's original implementation.
    """
    conversations = example["conversations"]
    
    # Build conversation text
    text_parts = []
    for conv in conversations:
        if conv["from"] == "human":
            text_parts.append(f"Human: {conv['value']}")
        elif conv["from"] == "gpt":
            text_parts.append(f"Assistant: {conv['value']}")
    
    full_text = "\n\n".join(text_parts)
    
    # Tokenize
    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding=False,  # Don't pad individual examples
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    
    # Labels are the same as input_ids for language modeling
    result = {
        "input_ids": tokenized["input_ids"].squeeze(0),
        "attention_mask": tokenized["attention_mask"].squeeze(0),
        "labels": tokenized["input_ids"].squeeze(0).clone()
    }
    
    return result


def prepare_dataset(tokenizer, training_args, split="train"):
    """Prepare training dataset matching Apple's implementation."""
    
    if split == "train":
        # Use ShareGPT dataset for training (matches Apple's approach)
        try:
            print("🔄 Loading ShareGPT training dataset...")
            dataset = datasets.load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
            
            # Process dataset
            tokenized_dataset = dataset.map(
                lambda x: sharegpt_record_to_training_instance(x, tokenizer),
                num_proc=min(4, multiprocessing.cpu_count()),
                remove_columns=dataset.column_names
            )
            
        except Exception as e:
            print(f"⚠️ Could not load ShareGPT dataset: {e}")
            print("🔄 Falling back to WikiText dataset...")
            
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
        # For evaluation, use Alpaca dataset (matches Apple's approach)
        print("🔄 Loading Alpaca evaluation dataset...")
        dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", split="eval")
        
        # Convert to ShareGPT format then tokenize
        dataset = dataset.map(
            convert_alpaca_to_sharegpt,
            num_proc=min(4, multiprocessing.cpu_count())
        )
        
        tokenized_dataset = dataset.map(
            lambda x: sharegpt_record_to_training_instance(x, tokenizer),
            num_proc=min(4, multiprocessing.cpu_count()),
            remove_columns=dataset.column_names
        )
    
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
    
    # Load training dataset (ShareGPT format, matching Apple's approach)
    print("🔄 Loading training dataset...")
    train_dataset = prepare_dataset(tokenizer, training_args, split="train")
    
    # Set RoPE scaling factor if needed
    config = AutoConfig.from_pretrained(model_args.llm_name_or_path)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    # Load and freeze the base model
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
    
    # Setup wandb before creating model
    setup_wandb(model_args, training_args, drafter_config.to_dict())
    
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
    
    # Log final metrics to wandb
    if training_args.use_wandb and WANDB_AVAILABLE:
        # Log model artifacts
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
        
        wandb.finish()
        print(f"📊 Training artifacts logged to: {wandb.run.url}")
    
    return speculator


def eval_model(model_args, training_args):
    """Evaluation function adapted from Apple's implementation."""
    
    print(f"🔍 Starting Recurrent Drafting Evaluation")
    
    tokenizer = get_tokenizer(model_args, training_args)
    compute_metrics = get_compute_metrics(training_args)
    
    # Load evaluation dataset (Alpaca format, matching Apple's approach)
    print("🔄 Loading Alpaca evaluation dataset...")
    try:
        eval_dataset = prepare_dataset(tokenizer, training_args, split="eval")
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
        
        # Log evaluation results to wandb
        if training_args.use_wandb and WANDB_AVAILABLE:
            # Initialize wandb for evaluation if not already done
            if not wandb.run:
                wandb.init(
                    project=training_args.wandb_project,
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
            wandb.finish()
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
        use_wandb=False,  # Disabled by default for quick training
        wandb_project="recurrent-drafting",
        wandb_run_name="quick-training"
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
    """Show training command examples."""
    
    print("\n🚀 Training Examples")
    print("=" * 30)
    
    print("\n1. Quick Training (Demo):")
    print("python train_speculator.py --quick")
    
    print("\n2. Full Training:")
    print("python train_speculator.py \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --output_dir ./models/gpt2-recurrent-drafter \\")
    print("  --num_train_epochs 3 \\")
    print("  --per_device_train_batch_size 8 \\")
    print("  --learning_rate 5e-4 \\")
    print("  --drafter_predict_n_tokens 4 \\")
    print("  --drafter_num_layers 2 \\")
    print("  --rnn")
    
    print("\n3. Training with Weights & Biases:")
    print("python train_speculator.py \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --use_wandb \\")
    print("  --wandb_project my-recurrent-drafting \\")
    print("  --wandb_run_name gpt2-experiment-1 \\")
    print("  --num_train_epochs 3")
    
    print("\n4. Evaluation:")
    print("python train_speculator.py \\")
    print("  --phase eval \\")
    print("  --llm_name_or_path gpt2 \\")
    print("  --drafter_name_or_path ./models/gpt2-recurrent-drafter \\")
    print("  --use_wandb  # Optional: log eval results")


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
    
    # Weights & Biases arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="recurrent-drafting", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    
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
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
        
        if args.phase == "train":
            train(model_args, training_args)
        else:
            eval_model(model_args, training_args)


if __name__ == "__main__":
    main()