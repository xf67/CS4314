"""
The main program for finetuning LLMs with Huggingface Transformers Library.

ALL SECTIONS WHERE CODE POSSIBLY NEEDS TO BE FILLED IN ARE MARKED AS TODO.
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import torch
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM
import datasets

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.cache_size_limit = 32
# torch._dynamo.config.dynamic_shapes = True

# Define the arguments required for the main program.
# NOTE: You can customize any arguments you need to pass in.
@dataclass
class ModelArguments:
    """Arguments for model
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the LLM to fine-tune or its name on the Hugging Face Hub."
        }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype."
            ),
            "choices": ["bfloat16", "float16", "float32"],
        },
    )
    # TODO: add your model arguments here
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention to optimize memory usage"}
    )
    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use LoRA"
        }
    )

@dataclass
class DataArguments:
    """Arguments for data
    """
    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the fine-tuning dataset or its name on the Hugging Face Hub."
        }
    )
    # TODO: add your data arguments here
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total sequence length after tokenization."
        }
    )
    mask_prefix: bool = field(
        default=True,
        metadata={
            "help": "Whether to mask the prefix (instruction and input) tokens in loss computation"
        }
    )
    bos_token: bool = field(
        default=False,
        metadata={
            "help": "Whether to manually add bos token to the input"
        }
    )
    sample_ratio: float = field(
        default=1,
        metadata={
            "help": "The ratio of dataset sampling."
        }
    )

from peft import LoraConfig, TaskType  #注意要装0.13，不然和opencompass对hugging face的版本要求冲突
from transformers import AutoModelForCausalLM
from peft import get_peft_model

# The main function
# NOTE You can customize some logs to monitor your program.
def finetune():
    # TODO Step 1: Define an arguments parser and parse the arguments
    # NOTE Three parts: model arguments, data arguments, and training arguments
    # HINT: Refer to 
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/internal/trainer_utils#transformers.HfArgumentParser
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/trainer#transformers.TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # TODO Step 2: Load tokenizer and model
    # HINT 1: Refer to
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/tokenizer#tokenizer
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/qwen2
    # HINT 2: To save training GPU memory, you need to set the model's parameter precision to half-precision (float16 or bfloat16).
    #         You may also check other strategies to save the memory!
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/llama2#usage-tips
    #   * https://huggingface.co/docs/transformers/perf_train_gpu_one
    #   * https://www.53ai.com/news/qianyanjishu/2024052494875.html
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    torch_dtype = (
        getattr(torch, model_args.torch_dtype)
        if model_args.torch_dtype in ["float16", "bfloat16"]
        else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=True,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention else None,
        trust_remote_code=True,
        device_map="cuda:0",
    )

    if model_args.use_lora:
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # TODO Step 3: Load dataset
    # HINT: https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.Dataset
    dataset = datasets.load_dataset(
        data_args.dataset_path,
        split="train"
    )

    if data_args.sample_ratio < 1.0:
        total_samples = len(dataset)
        num_samples = int(total_samples * data_args.sample_ratio)
        dataset = dataset.shuffle(seed=42).select(range(num_samples))

    # TODO Step 4: Define the data collator function
    # NOTE During training, for each model parameter update, we fetch a batch of data, perform a forward and backward pass,
    # and then update the model parameters. The role of the data collator is to process the data (e.g., padding the data within
    # a batch to the same length) and format the batch into the input required by the model.
    #
    # In this assignment, the purpose of the custom data_collator is to process each batch of data from the dataset loaded in
    # Step 3 into the format required by the model. This includes tasks such as tokenizing the data, converting each token into 
    # an ID sequence, applying padding, and preparing labels.
    # 
    # HINT:
    #   * Before implementation, you should:
    #      1. Clearly understand the format of each sample in the dataset loaded in Step 3.
    #      2. Understand the input format required by the model (https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2ForCausalLM).
    #         Reading its source code also helps!

    def data_collator(batch: List[Dict]):
        """
        batch: list of dict, each dict of the list is a sample in the dataset.
        """
        formatted_texts = []
        prefix_lengths = []
        
        if data_args.bos_token:
            for item in batch:
                if item["input"]:
                    text = (
                        f"<|im_start|>user\n{item['instruction']}\n\n{item['input']}<|im_end|>\n"
                        f"<|im_start|>assistant\n{item['output']}<|im_end|>"
                    )
                else:
                    text = (
                        f"<|im_start|>user\n{item['instruction']}<|im_end|>\n"
                        f"<|im_start|>assistant\n{item['output']}<|im_end|>"
                    )
                formatted_texts.append(text)
        else:
            for item in batch:
                # 先计算instruction和input部分的token长度
                prefix = item['instruction'] + (item['input'] if item['input'] else '')
                prefix_length = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
                prefix_lengths.append(prefix_length)
                
                # 然后拼接完整文本
                if item["input"]:
                    text = item['instruction'] + item['input'] + item['output']
                else:
                    text = item['instruction'] + item['output']
                formatted_texts.append(text)
    
        # tokenize完整文本
        tokenized = tokenizer(
            formatted_texts,
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length,
            add_special_tokens=False
        )

        # 处理labels
        labels = [ids.copy() for ids in tokenized["input_ids"]]
        
        # 使用预先计算的长度标记-100
        if data_args.mask_prefix:
            if data_args.bos_token:
                for idx, text in enumerate(formatted_texts):
                    assistant_pos = text.find("<|im_start|>assistant\n")
                    assistant_token_start = len(tokenizer(text[:assistant_pos], add_special_tokens=False)["input_ids"])
                    labels[idx][:assistant_token_start] = [-100] * assistant_token_start
            else:
                for idx, prefix_length in enumerate(prefix_lengths):
                    labels[idx][:prefix_length] = [-100] * prefix_length

        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor(labels)
        }

    # TODO Step 5: Define the Trainer
    # HINT: https://huggingface.co/docs/transformers/main_classes/trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Step 6: Train!
    trainer.train()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.argv = [
   "notebook",
    "--model_name_or_path", "./Qwen2.5-1.5B",
    "--dataset_path", "./alpaca-cleaned",
    "--output_dir", "./outputs/",
    "--num_train_epochs", "2",
    "--per_device_train_batch_size", "2",
    # "--gradient_accumulation_steps", "4",
    "--learning_rate", "1e-5",
    "--warmup_ratio", "0.05",
    "--logging_steps", "100",
    "--save_strategy", "epoch",
    "--bf16",
    "--torch_dtype", "bfloat16",
    "--max_seq_length", "1024",
    # "--gradient_checkpointing",
    "--remove_unused_columns", "False",
    # "--max_grad_norm", "1.0",
    # "--weight_decay", "0.01"
    # "--torch_compile",
    "--mask_prefix", "True",
    "--bos_token", "False",
    # "--use_flash_attention", "True",
    "--use_lora",
    "--sample_ratio", "0.5"
]
finetune()