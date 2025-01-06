from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, pipeline
from datasets import load_dataset

openwebtext = load_dataset("openwebtext", split="train", trust_remote_code=True)  # Open Web Text
openwebtext = openwebtext.shuffle(seed=42).select(range(1000))


# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Assign a padding token to the GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as the padding token

# Tokenize the data
# Tokenize the data with labels for causal language modeling
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding=True, truncation=True, max_length=256)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Set labels as input_ids for causal language modeling
    return tokenized


tokenized_dataset = openwebtext.map(tokenize_function, batched=True)

train_test_split = openwebtext.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]


# Fine-tune using Hugging Face Trainer
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=5e-5,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")