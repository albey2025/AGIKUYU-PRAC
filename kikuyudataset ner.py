from transformers import AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from nltk import wordpunct_tokenize
model_name = "xlm-roberta-base-english"
tokenizer =wordpunct_tokenize
model = AutoModelForTokenClassification.from_pretrained(model_name)

#translation of the dataset: 'Maina has gone to Nairobi to take a parcel and take it to Muranga'

train_dataset = [
    {
        "tokens": ["Maina", "athie", "Nairobi", "kuoya", "murigo", "atware", "Muranga"],
        "ner_tags": ["PERSON", "O", "LOCATION", "O", "O", "O", "LOCATION"]
    }
]
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)
trainer.train()