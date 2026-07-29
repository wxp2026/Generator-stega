import argparse
import os
import time
from typing import Optional, Tuple
import torch
import torch.distributed as dist
import transformers
import yaml
from datasets import (
    Dataset,
    load_dataset,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
)

# Set logging level for transformers
transformers.logging.set_verbosity_info()


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0) in distributed training.

    Returns:
        bool: True if this is the main process, False otherwise
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def dist_print(*args, **kwargs) -> None:
    """
    Print only from the main process (rank 0) in distributed training.
    Prevents duplicate outputs in multi-GPU settings.

    Args:
        *args: Arguments to pass to print function
        **kwargs: Keyword arguments to pass to print function
    """
    if is_main_process():
        print(*args, **kwargs)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for causal language model fine-tuning.

    Returns:
        argparse.Namespace: Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for causal language modeling"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="Name of the dataset on HuggingFace Hub",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default=None,
        help="Name of the subset of the dataset (if applicable)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="GenerTeam/GENERator-eukaryote-1.2b-base",
        help="HuggingFace model path or name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per GPU for training",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=16384,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating model",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fine_tuning",
        help="Path to save the fine-tuned model",
    )
    # In appropriate situations, we recommend setting --pad_to_multiple_of_six true by default to avoid generating <oov> at the end of sequences.
    parser.add_argument(
        "--pad_to_multiple_of_six",
        action="store_true",
        help="Pad sequences to multiple of 6 with 'A'. ",
    )
    parser.add_argument(
        "--hf_config_path",
        type=str,
        default="configs/hf_configs/fine_tuning.yaml",
        help="Path to the YAML configuration file for HuggingFace Trainer",
    )
    parser.add_argument(
        "--distributed_type",
        type=str,
        default="ddp",
        choices=["ddp", "deepspeed", "fsdp"],
        help="Type of distributed training to use",
    )
    parser.add_argument(
        "--sequence_column",
        type=str,
        default="dna_sequence",  # 设置默认值为 dna_sequence
        help="Name of the column containing sequences in the dataset"
    )
    return parser.parse_args()


def setup_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load and configure tokenizer for causal language modeling.

    Args:
        model_name: Name or path of the HuggingFace model

    Returns:
        PreTrainedTokenizer: Configured tokenizer for the model
    """
    dist_print(f"🔤 Loading tokenizer from: {model_name}")
    start_time = time.time()

    # Load tokenizer with trust_remote_code to support custom models
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad_token to eos_token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dist_print(
        f"⏱️ Tokenizer loading completed in {time.time() - start_time:.2f} seconds"
    )

    return tokenizer


def setup_dataset(
        dataset_name: str,
        subset_name: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        pad_to_multiple_of_six: bool = False,
        sequence_column: str = "dna_sequence",
) -> Tuple[Dataset, Dataset, Dataset]:
    # ... 前面的代码保持不变 ...

    # 检查是否是本地目录（包含手动划分的数据集文件）
    if os.path.isdir(dataset_name):
        dist_print("🔍 Loading manually split datasets from local directory")

        # 构建训练集、验证集、测试集文件路径
        train_file = os.path.join(dataset_name, "train.parquet")
        val_file = os.path.join(dataset_name, "val.parquet")
        test_file = os.path.join(dataset_name, "test.parquet")
        start_time = time.time()
        # 检查文件是否存在
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not os.path.exists(val_file):
            raise FileNotFoundError(f"Validation file not found: {val_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        # 🔧 修复：使用正确的split名称
        train_dataset = load_dataset('parquet', data_files=train_file, split='train')
        val_dataset = load_dataset('parquet', data_files=val_file, split='train')  # ✅ 
        test_dataset = load_dataset('parquet', data_files=test_file, split='train')  # ✅


    else:
        # 原有的从HuggingFace加载的逻辑
        if subset_name is None:
            dataset = load_dataset(dataset_name, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, subset_name, trust_remote_code=True)

        # 确保数据集有训练、验证和测试划分
        if "train" not in dataset:
            raise ValueError("Dataset must contain a 'train' split")
        if "validation" not in dataset and "val" not in dataset:
            raise ValueError("Dataset must contain a 'validation' or 'val' split")
        if "test" not in dataset:
            raise ValueError("Dataset must contain a 'test' split")

        train_dataset = dataset["train"]
        val_dataset = dataset.get("validation", dataset.get("val", None))
        test_dataset = dataset["test"]

    # 检查序列列是否存在
    if sequence_column not in train_dataset.column_names:
        available_columns = train_dataset.column_names
        raise ValueError(
            f"Sequence column '{sequence_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )

    dist_print(f"⚡ Dataset loaded in {time.time() - start_time:.2f} seconds")
    dist_print(f"📊 Training set: {len(train_dataset)} examples")
    dist_print(f"📊 Validation set: {len(val_dataset)} examples")
    dist_print(f"📊 Test set: {len(test_dataset)} examples")
    dist_print(f"📋 Available columns: {train_dataset.column_names}")

    # 定义数据处理函数
    def _process_function(examples):
        # 使用指定的序列列
        sequences = examples[sequence_column]

        # 如果请求，对原始序列应用填充
        if pad_to_multiple_of_six:
            padded_sequences = []
            for seq in sequences:
                remainder = len(seq) % 6
                if remainder != 0:
                    pad_len = 6 - remainder
                    seq = seq + "A" * pad_len
                padded_sequences.append(seq)
            sequences = padded_sequences

        # 对序列进行tokenization
        tokenized = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            padding=False,
        )

        return tokenized

    # 对训练集、验证集和测试集分别应用tokenization
    dist_print("🔧 Processing training dataset...")
    train_dataset = train_dataset.map(
        _process_function,
        batched=True,
        remove_columns=train_dataset.column_names,  # 这会移除所有原始列，包括 protein_id 等
    )

    dist_print("🔧 Processing validation dataset...")
    val_dataset = val_dataset.map(
        _process_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    dist_print("🔧 Processing test dataset...")
    test_dataset = test_dataset.map(
        _process_function,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    return train_dataset, val_dataset, test_dataset

def setup_model(model_name: str) -> PreTrainedModel:
    """
    Load and configure model for causal language modeling.

    Args:
        model_name: Name or path of the HuggingFace model

    Returns:
        PreTrainedModel: Configured pre-trained model for causal language modeling
    """
    dist_print(f"🤖 Loading AutoModelForCausalLM from: {model_name}")
    start_time = time.time()

    # Load model with trust_remote_code to support custom models
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure pad_token_id is set
    if model.config.pad_token_id is None and hasattr(model.config, "eos_token_id"):
        model.config.pad_token_id = model.config.eos_token_id

    # Report model size for reference
    total_params = sum(p.numel() for p in model.parameters())
    dist_print(f"📊 Model size: {total_params / 1e6:.1f}M parameters")
    dist_print(f"⏱️ Model loading completed in {time.time() - start_time:.2f} seconds")

    return model


def setup_training_args(yaml_path=None, cli_args=None, **kwargs):
    """
    Create a TrainingArguments instance from YAML, CLI arguments, and code arguments.
    Priority: code kwargs > CLI args > YAML config

    Args:
        yaml_path: Path to YAML configuration file
        cli_args: Parsed command line arguments
        **kwargs: Direct arguments that take highest priority

    Returns:
        TrainingArguments: Configured training arguments
    """
    # Start with yaml configuration if provided
    yaml_kwargs = {}
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            yaml_kwargs = yaml.safe_load(f)

    # Create a dictionary from CLI arguments
    cli_kwargs = {}
    if cli_args is not None:
        # Add basic training parameters
        if hasattr(cli_args, "output_dir"):
            cli_kwargs["output_dir"] = cli_args.output_dir
        if hasattr(cli_args, "batch_size"):
            cli_kwargs["per_device_train_batch_size"] = cli_args.batch_size
        if hasattr(cli_args, "learning_rate"):
            cli_kwargs["learning_rate"] = cli_args.learning_rate
        if hasattr(cli_args, "gradient_accumulation_steps"):
            cli_kwargs["gradient_accumulation_steps"] = (
                cli_args.gradient_accumulation_steps
            )
        if hasattr(cli_args, "num_train_epochs"):
            cli_kwargs["num_train_epochs"] = cli_args.num_train_epochs

        # Handle distributed training configurations
        if hasattr(cli_args, "distributed_type"):
            if cli_args.distributed_type == "deepspeed":
                cli_kwargs["deepspeed"] = "configs/ds_configs/zero1.json"
            elif cli_args.distributed_type == "fsdp":
                cli_kwargs["fsdp"] = "shard_grad_op auto_wrap"
                cli_kwargs["fsdp_config"] = "configs/ds_configs/fsdp.json"

    # Handle BF16 precision based on GPU capability
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        cli_kwargs["bf16"] = True

    # Merge all configurations, with priority: kwargs > cli_kwargs > yaml_kwargs
    final_kwargs = {**yaml_kwargs, **cli_kwargs, **kwargs}

    # Add defaults for saving strategy
    if "save_strategy" not in final_kwargs:
        final_kwargs["save_strategy"] = "epoch"

    # Add logging steps if not provided
    if "logging_steps" not in final_kwargs:
        final_kwargs["logging_steps"] = 10

    # Create and return the TrainingArguments instance
    return TrainingArguments(**final_kwargs)


def train_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        args: argparse.Namespace,
) -> Trainer:
    dist_print("🚀 Setting up training...")
    start_time = time.time()

    # Configure training arguments
    training_args = setup_training_args(
        yaml_path=args.hf_config_path,
        cli_args=args,
    )

    # 🔧 强制覆盖评估相关设置
    training_args.do_eval = True
    training_args.evaluation_strategy = "epoch"
    training_args.save_strategy = "epoch"
    training_args.save_steps = None
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "eval_loss"
    training_args.greater_is_better = False
    training_args.logging_steps = 50

    # 🔧 增强调试信息
    dist_print(f"📋 Final training config:")
    dist_print(f"   do_eval: {training_args.do_eval}")
    dist_print(f"   evaluation_strategy: {training_args.evaluation_strategy}")
    dist_print(f"   eval_steps: {training_args.eval_steps}")
    dist_print(f"   eval_dataset length: {len(val_dataset) if val_dataset else 0}")
    dist_print(f"   eval_dataset features: {val_dataset.features if val_dataset else 'None'}")
    dist_print(f"   eval_dataset column names: {val_dataset.column_names if val_dataset else 'None'}")

    # 🔧 添加数据集验证
    if val_dataset is None or len(val_dataset) == 0:
        dist_print("❌ WARNING: Validation dataset is empty or None!")
    else:
        dist_print(f"✅ Validation dataset loaded successfully with {len(val_dataset)} examples")

    # Initialize Trainer with validation dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 🔧 添加训练前的检查
    total_steps = len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps) * args.num_train_epochs
    dist_print(f"📊 Training setup completed in {time.time() - start_time:.2f} seconds")
    dist_print(f"📊 Training examples: {len(train_dataset)}")
    dist_print(f"📊 Validation examples: {len(val_dataset)}")
    dist_print(f"📊 Test examples: {len(test_dataset)}")
    dist_print(f"📊 Estimated total steps: {total_steps}")
    dist_print(f"📊 First evaluation at step: {training_args.eval_steps}")
    dist_print(f"📊 Logging every {training_args.logging_steps} steps")
    dist_print(f"📊 Checkpoint saving every {training_args.save_steps} steps")
    
    # 🔧 强制在训练开始前进行一次验证集测试
    dist_print("🧪 Running pre-training validation check...")
    try:
        initial_eval = trainer.evaluate(eval_dataset=val_dataset)
        dist_print(f"📊 Initial validation results: {initial_eval}")
    except Exception as e:
        dist_print(f"❌ Initial validation failed: {str(e)}")
    
    dist_print("🏋️ Starting model training...")
    training_start_time = time.time()

    # Train the model
    trainer.train()

   
    dist_print(
        f"✅ Training completed in {(time.time() - training_start_time) / 60:.2f} minutes"
    )

    # Evaluate on test set after training
    dist_print("🧪 Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    dist_print(f"📊 Test set results: {test_results}")

    # Save the best model (if load_best_model_at_end is enabled)
    if training_args.load_best_model_at_end:
        dist_print("💾 Saving best model...")
        trainer.save_model(args.output_dir)

    return trainer

def save_model(
    trainer: Trainer, tokenizer: PreTrainedTokenizer, output_dir: str
) -> None:
    """
    Save the fine-tuned model and tokenizer.

    Args:
        trainer: Trained model trainer
        tokenizer: Tokenizer for the model
        output_dir: Directory to save the model
    """
    dist_print(f"💾 Saving fine-tuned model to {output_dir}")
    start_time = time.time()

    # Save the model
    trainer.save_model(output_dir)

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)

    dist_print(f"✅ Model saved in {time.time() - start_time:.2f} seconds")


def display_progress_header() -> None:
    """
    Display a stylized header for the causal language model fine-tuning.
    """
    dist_print("\n" + "=" * 80)
    dist_print("🔥  CAUSAL LANGUAGE MODEL FINE-TUNING PIPELINE  🔥")
    dist_print("=" * 80 + "\n")


def main() -> None:
    """
    Main function to run the causal language model fine-tuning pipeline.
    """
    # Display header
    display_progress_header()

    # Start timer for total execution
    total_start_time = time.time()

    # Parse command line arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup tokenizer first
    tokenizer = setup_tokenizer(args.model_name)

    # Load and prepare data - 修改为接收三个数据集
    train_dataset, val_dataset, test_dataset = setup_dataset(  # 修改这行
        args.dataset_name,
        args.subset_name,
        tokenizer,
        args.max_length,
        args.pad_to_multiple_of_six,
        sequence_column=args.sequence_column,  # 添加这行
    )

    # Initialize model
    model = setup_model(args.model_name)

    # Train model - 修改为传入三个数据集
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, test_dataset, args)  # 修改这行

    # Save fine-tuned model - 修改条件判断
    if not trainer.args.load_best_model_at_end:
        save_model(trainer, tokenizer, args.output_dir)

    # Print total execution time
    total_time = time.time() - total_start_time
    minutes, seconds = divmod(total_time, 60)
    dist_print(f"\n⏱️ Total execution time: {int(minutes)}m {seconds:.2f}s")
    dist_print("✨ Fine-tuning completed successfully! ✨\n")


if __name__ == "__main__":
    main()
