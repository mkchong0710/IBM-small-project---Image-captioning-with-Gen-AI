from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Keep conversation history
conversation_history = []

while True:
    # Encoding conversation history
    history_string = "\n".join(conversation_history)

    # Prompt
    input_text ="hello, how are you doing?"

    # Tokenization of user prompt and chat history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate model response
    outputs = model.generate(**inputs)

    # Decode mode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print(response)

    # Add to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

