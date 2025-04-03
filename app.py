from flask import Flask, render_template, request
import tensorflow as tf
from transformers import T5Tokenizer, T5EncoderModel
import torch
import numpy as np
from model import DeepScan, process_sequence 
app = Flask(__name__)


MAXSEQ = 1000
NUM_FEATURE = 1024
NUM_CLASSES = 2
WINDOW_SIZES = [4,8,16,24,32,48,56,64]
NUM_FILTER = 64
NUM_HIDDEN = 64
DEVICE = torch.device("cpu")  


print("Loading ProtT5 model...")
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
pt_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", output_hidden_states=True)
pt_model = pt_model.to(DEVICE)
pt_model.eval()
print("ProtT5 model loaded.")


print("Initializing DeepScan...")
model = DeepScan(
    num_filters=NUM_FILTER,
    num_hidden=NUM_HIDDEN,
    window_sizes=WINDOW_SIZES
)


dummy_input = tf.random.normal((1, 1, MAXSEQ, NUM_FEATURE))
model(dummy_input)


model.load_weights("model/weights-53.weights.h5")
model.summary()
print("DeepScan model loaded successfully.")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        sequence = request.form["sequence"].strip().upper()

        # Encode ProtT5
        embedding = process_sequence(sequence, tokenizer, pt_model, MAXSEQ, DEVICE)
        embedding = embedding.reshape(1, 1, MAXSEQ, NUM_FEATURE)


        prediction = model(embedding, training=False).numpy()
        print("🔎 Raw prediction output:", prediction)

        label = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][label]
        result = {
            "label": "SNARE" if label == 1 else "Non-SNARE",
            "confidence": f"{confidence:.2%}"
        }
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
