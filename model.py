from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_id = "google/flan-t5-large"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# pipeline
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
