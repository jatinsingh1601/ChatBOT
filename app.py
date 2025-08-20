from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time

app = Flask(__name__)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Specify the path to your fine-tuned model and tokenizer
model_path = r"/Users/jatinsinghchauhan/Desktop/ChatBot/trained_model"  # Replace with the actual path to your local model directory

# Load the locally fine-tuned DialoGPT model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    
    # Set pad_token_id separately if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model_loaded = True
    print("Model loaded successfully")
except Exception as e:
    model_loaded = False
    print(f"Error loading model: {e}")

# Chat history placeholder
chat_history_ids = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    global chat_history_ids
    user_input = request.json.get("prompt", "")
    if not user_input:
        return jsonify({"response": "No input received."})
    
    if not model_loaded:
        return jsonify({"response": "Model is not loaded. Please check server logs."})

    try:
        # Add a small delay to simulate processing time
        time.sleep(0.5)
        
        # Encode user input and move tensor to GPU
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)

        # Concatenate with chat history if available
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1).to(device) if chat_history_ids is not None else new_user_input_ids

        # Generate bot response with a longer sequence
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # Decode the bot response and move tensors back to CPU if necessary
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"response": f"I encountered an error while processing your request. Please try again with a different input."})

@app.route("/new-chat", methods=["POST"])
def new_chat():
    global chat_history_ids
    # Reset the chat history
    chat_history_ids = None
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)