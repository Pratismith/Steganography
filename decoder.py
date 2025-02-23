import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Load the tokenizer for decoding
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Open the encrypted image
enc_img = Image.open('encrypted_image.png')
enc_pixelMap = enc_img.load()

# Extract the hidden message
msg_len = 0
msg_seq = []
msg_index = 0

for row in range(enc_img.size[0]):
    for col in range(enc_img.size[1]):
        r, g, b = enc_pixelMap[row, col]

        if row == 0 and col == 0:
            msg_len = r  # Retrieve message length
        elif msg_index < msg_len:
            msg_seq.append(r)
            msg_index += 1

enc_img.close()

# Convert sequence back to text
msg_seq = np.array([msg_seq])
decoded_msg = tokenizer.sequences_to_texts(msg_seq)[0]

print("\nThe hidden message is:\n")
print(decoded_msg)
