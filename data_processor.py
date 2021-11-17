import nltk
import torch
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def predict(text, model, max_len, vocab):
    """Memprediksi probabilitas seorang penulis (teks) ini introvert."""

    # Melakukan tokenisasi, padding, dan encoding pada teks
    tokens = word_tokenize(text.lower())
    padded_tokens = tokens + ['<pad>'] * (max_len - len(tokens))
    input_id = [vocab.get(token, vocab['<unk>']) for token in padded_tokens]

    # Konversi ke PyTorch tensors
    input_id = torch.tensor(input_id).unsqueeze(dim=0)

    # Menghitung logits (layer akhir yang akan dimasukkan ke dalam softmax)
    logits = model.forward(input_id)

    # Menghitung probabilitas
    probs = F.softmax(logits, dim=1).squeeze(dim=0)
    return probs[0] * 100, probs[1] * 100