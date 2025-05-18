from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset

model_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_id)
model     = BlipForConditionalGeneration.from_pretrained(model_id)

train_ds = load_dataset("nlphuji/flickr8k", split="train[:90%]")
val_ds   = load_dataset("nlphuji/flickr8k", split="train[90%:]")

def transform(batch):
    inputs = processor(images=batch["image"],
                       text=batch["caption"],
                       padding="max_length",
                       truncation=True,
                       max_length=30,
                       return_tensors="pt")
    inputs["labels"] = inputs["input_ids"]
    inputs.pop("input_ids")
    return inputs

train_ds = train_ds.map(transform, batched=True, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(transform, batched=True, remove_columns=val_ds.column_names)

cols = ["pixel_values", "labels"]
train_ds.set_format(type="torch", columns=cols)
val_ds.set_format(type="torch", columns=cols)

args = TrainingArguments(
    output_dir="blip_ft",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    fp16=False,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(model=model,
                  args=args,
                  train_dataset=train_ds,
                  eval_dataset=val_ds,
                  tokenizer=processor)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("blip_ft")
