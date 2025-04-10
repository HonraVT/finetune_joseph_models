import modal


HUGGINGFACE_TOKEN = "hf_wasdwasdwasdwasdwasdwa"  # Troque pelo seu token do huggingface
HUGGINGFACE_USERNAME = "fuze-eletrik"  # Troque pelo seu nome de usuário do huggingface
GGUF_MODEL_NAME = "llama-3.1-8B-lexi-uncensored-V2-jose-GGUF"  # Troque pelo nome desejar
DATASET_FROM_GITHUB_GIST = "https://gist.githubusercontent.com/HonraVT/45319e12ecb462212ebad40fd41d99fc/raw/30b3d378351a232f9309f6635b8b13e0c38a9da0/jose_dataset_2405.json"

MINUTES = 60
cuda_version = "12.6.3"
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

def format_chat(conversation):
    system_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
    user_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{conversation['input']}<|eot_id|>"
    assistant_response = f"<|start_header_id|>assistant<|end_header_id|>\n\n{conversation['output']}<|eot_id|>"

    return {"text": system_prompt + user_prompt + assistant_response}

# Create a persistent volumes to store the huggingface cache and output checkpoints
huggingface_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("outputs", create_if_missing=True)

# set image and pre install llama.cpp to prevent unsloth error.
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install(
        "git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev"
    )
    .run_commands("git clone https://github.com/ggerganov/llama.cpp")
    .run_commands(
        "cmake llama.cpp -B llama.cpp/build "
        "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON "
    )
    .run_commands(  # this one takes a few minutes!
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli"
    )
    .run_commands("cp llama.cpp/build/bin/llama-* llama.cpp")
    .env({"HF_HOME": "/root/models/"})
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install(["unsloth==2025.3.18", "unsloth_zoo==2025.3.16"])
    .entrypoint([])  # remove NVIDIA base container entrypoint
)

app = modal.App("unsloth", image=image)


@app.function(
    gpu="t4",
    # scaledown_window=120 * MINUTES, # 2 hours
    timeout=120 * MINUTES, # 2 hours
    enable_memory_snapshot=True,
    volumes={"/root/models": huggingface_cache_volume, "/root/outputs": outputs_volume},
)
def run_trainer():
    import subprocess

    # result = subprocess.run(["ls", "/data/Llama-3.1-8B-Lexi-Uncensored-V2"], capture_output=True, text=True)
    result = subprocess.run(["ls", "-a"], capture_output=True, text=True)
    print(result.stdout)

    #######

    from unsloth import FastLanguageModel

    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
        max_seq_length=max_seq_length,
        dtype=dtype
    )

    huggingface_cache_volume.commit()

    #######

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    #######

    from datasets import load_dataset

    dataset = load_dataset("json", data_files={
        "train": DATASET_FROM_GITHUB_GIST},
                           split="train")
    dataset = dataset.map(format_chat)

    print(dataset[5]["text"])

    #######

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        # data_collator = data_collator,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        # data_collator = data_collator,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=2,  # Set this for 1 full training run.
            # max_steps = 300,
            # max_steps = 60,
            # learning_rate = 2e-4,
            # learning_rate = 5e-7,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_strategy="steps",
            save_steps=300,
            report_to="none",  # Use this for WandB etc
        ),
    )

    print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))


    ########

    print("Training...")
    trainer_stats = trainer.train()
    # trainer_stats = self.trainer.train(resume_from_checkpoint=True)

    outputs_volume.commit()

    ########

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")

    ########

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    conv = {
        "input": "Você finge que trabalha?",
        "output": ""
    }

    inputs = tokenizer(
        [
            format_chat(conv)["text"]
        ], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    print(tokenizer.batch_decode(outputs))

    ########

    model.push_to_hub_gguf(
        f"{HUGGINGFACE_USERNAME}/{GGUF_MODEL_NAME}",
        tokenizer,
        quantization_method=["q4_k_m", "q8_0", ],
        token=HUGGINGFACE_TOKEN,
    )

    outputs_volume.commit()
    huggingface_cache_volume.commit()


@app.function(
    gpu="t4",
    # scaledown_window=120 * MINUTES, # 2 hours
    timeout=120 * MINUTES,  # 2 hours
    enable_memory_snapshot=True,
    volumes={"/root/models": huggingface_cache_volume, "/root/outputs": outputs_volume},
)
def inference():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    max_seq_length = 2048
    dtype = None

    model, tokenizer = FastLanguageModel.from_pretrained(
        # Substitua xxx pelo número do passo de treinamento desejado
        # em https://modal.com/storage/<you username>/main/outputs ou
        # rode: modal volume ls outputs
        model_name="outputs/checkpoint-500",
        max_seq_length=max_seq_length,
        dtype=dtype
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": "cristione gosta de vc?"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)

    print(tokenizer.batch_decode(outputs))


@app.function(
    gpu="t4",
    # scaledown_window=120 * MINUTES, # 2 hours
    timeout=120 * MINUTES,  # 2 hours
    enable_memory_snapshot=True,
    volumes={"/root/models": huggingface_cache_volume, "/root/outputs": outputs_volume},
)
def save_model_from_checkpoint():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    max_seq_length = 2048
    dtype = None

    model, tokenizer = FastLanguageModel.from_pretrained(
        # Substitua xxx pelo número do passo de treinamento desejado
        # para ver seu armazenamento vá em https://modal.com/storage/<you username>/main/outputs ou
        # rode: modal volume ls outputs
        model_name="outputs/checkpoint-300",
        max_seq_length=max_seq_length,
        dtype=dtype
    )

    model.push_to_hub_gguf(
        f"{HUGGINGFACE_USERNAME}/{GGUF_MODEL_NAME}",
        tokenizer,
        quantization_method=["q4_k_m", "q8_0", ],
        token=HUGGINGFACE_TOKEN,
    )


# rode na linha de comando: modal run train_on_modal.py

@app.local_entrypoint()
def main():
    run_trainer.remote()
    # inference.remote()
    # save_model_from_checkpoint.remote()
