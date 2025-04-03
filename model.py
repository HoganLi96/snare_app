import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import T5Tokenizer, T5EncoderModel
import torch
import numpy as np

MAXSEQ = 1000
NUM_FEATURE = 1024
DEVICE = "cpu"

# Load ProtT5
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model_pt = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", output_hidden_states=True)
model_pt = model_pt.to(DEVICE)
model_pt.eval()

def process_sequence(sequence, tokenizer, pt_model, MAXSEQ, DEVICE):
    sequence = sequence[:MAXSEQ]
    sequence = ' '.join(list(sequence))  
    inputs = tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True, max_length=MAXSEQ)
    input_ids = inputs["input_ids"].to(DEVICE)

    with torch.no_grad():
        outputs = pt_model(input_ids)
        last_5_hidden = outputs.hidden_states[-5:]  
        stacked = torch.stack(last_5_hidden)  
        mean_embeddings = torch.mean(stacked, dim=0).squeeze(0)  

    seq_list = []
    for i in range(MAXSEQ):
        residue_embedding = mean_embeddings[i].cpu().numpy().tolist()
        seq_list.append(residue_embedding)

    return np.array(seq_list)



class DeepScan(Model):
    def __init__(self, input_shape=(1, MAXSEQ, NUM_FEATURE), window_sizes=[4,8,16,24,32,48,56,64], num_filters=64, num_hidden=64):
        super(DeepScan, self).__init__()
        self.window_sizes = window_sizes
        self.conv2d = []
        self.maxpool = []
        self.flatten = []

        for window_size in self.window_sizes:
            self.conv2d.append(layers.SeparableConv2D(
                filters=num_filters,
                kernel_size=(1, window_size),
                activation='relu',
                padding='same',
                depthwise_initializer=tf.keras.initializers.GlorotUniform(),
                pointwise_initializer=tf.keras.initializers.GlorotUniform(),
                bias_initializer=tf.constant_initializer(0.1),
                depthwise_regularizer=tf.keras.regularizers.l2(1e-4),
                pointwise_regularizer=tf.keras.regularizers.l2(1e-4)
            ))
            self.maxpool.append(layers.MaxPooling2D(
                pool_size=(1, MAXSEQ - window_size + 1),
                strides=(1, MAXSEQ),
                padding='same'))
            self.flatten.append(layers.Flatten())

        self.dropout = layers.Dropout(rate=0.7)
        self.fc1 = layers.Dense(
            num_hidden,
            activation='relu',
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )
        self.fc2 = layers.Dense(2, activation='softmax',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, x, training=False):
        _x = []
        for i in range(len(self.window_sizes)):
            x_conv = self.conv2d[i](x)
            x_maxp = self.maxpool[i](x_conv)
            x_flat = self.flatten[i](x_maxp)
            _x.append(x_flat)

        x = tf.concat(_x, axis=1)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
