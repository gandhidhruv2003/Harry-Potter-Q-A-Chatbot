from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "google/flan-t5-large"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=1024, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
