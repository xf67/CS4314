# README



This is the course project of CS4314 (Natural Language Processing).



## Instruction Tuning

` python finetune_l.py `

### Usage

You can modify the parameters in the bottom of the `finetune_l.py` file.


For example:

```python
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

```

Special Parameters:

- Speed up the training process:

    `--torch_compile` parameter is used to use torch compile to speed up the training process.

    `--use_flash_attention` parameter is used to use flash attention 2 to speed up the training process.

    `--use_lora` parameter is used to use LoRA to speed up the training process.

- Scaling Law:

    `--sample_ratio` parameter is used to sample the dataset.

- Other parameters:

    `--mask_prefix` parameter is used to mask the prefix of the input when calculating the loss.

    `--bos_token` parameter is used to add the bos token to the input manually.

## Chatbot

