import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from PIL import Image
import pickle

# Train LSTM for encryption
def train_lstm(msg):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts([msg])
    
    # Convert message to numerical sequence
    encoded_msg = tokenizer.texts_to_sequences([msg])[0]
    
    # Padding
    max_length = len(encoded_msg)
    padded_msg = pad_sequences([encoded_msg], maxlen=max_length, padding="post")[0]

    # Save tokenizer for decoding
    with open("tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)

    # Define LSTM model
    model = Sequential([
       Embedding(input_dim=1000, output_dim=64)  # Remove input_length
,
        LSTM(32, return_sequences=True),
        Dense(len(tokenizer.word_index) + 1, activation="softmax")
    ])
    
    # Train LSTM model (dummy training for demonstration)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    model.fit(np.expand_dims(padded_msg, axis=0), np.expand_dims(padded_msg, axis=0), epochs=5, verbose=0)


    # Get encoded message as LSTM output
    lstm_encoded_msg = padded_msg  # Using original encoding for simplicity
    
    return lstm_encoded_msg, max_length

# Open image
org_img = Image.open('original_image.png')
org_pixelMap = org_img.load()

# Create a new image
enc_img = Image.new(org_img.mode, org_img.size)
enc_pixelsMap = enc_img.load()

# Get message input and encrypt it using LSTM
msg = input("Enter the message: ")
lstm_encoded_msg, msg_len = train_lstm(msg)
msg_index = 0

# Hide the message in image
for row in range(org_img.size[0]):
    for col in range(org_img.size[1]):
        r, g, b = org_pixelMap[row, col]

        if row == 0 and col == 0:
            enc_pixelsMap[row, col] = (msg_len, g, b)  # Store message length
        elif msg_index < msg_len:
            enc_pixelsMap[row, col] = (lstm_encoded_msg[msg_index], g, b)
            msg_index += 1
        else:
            enc_pixelsMap[row, col] = (r, g, b)

org_img.close()
enc_img.save("encrypted_image.png")
enc_img.close()
print("Message encrypted and saved in 'encrypted_image.png'.")

