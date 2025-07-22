from flask import Flask, render_template, request, jsonify
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

# Download WordNet data if needed
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model.eval()

def get_wordnet_definition(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    return synsets[0].definition()

def explain_definition(definition):
    prompt = f"Explain this definition in simple terms: {definition}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return explanation

@app.route("/")
def index():
    return render_template("./templates/index.html")

@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json()
    word = data.get("word", "").strip()
    if not word:
        return jsonify({"error": "Please enter a word."}), 400

    definition = get_wordnet_definition(word)
    if not definition:
        return jsonify({"error": f"No dictionary definition found for '{word}'."}), 404

    explanation = explain_definition(definition)
    return jsonify({
        "word": word,
        "definition": definition,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True)
