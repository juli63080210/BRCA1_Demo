import torch
import numpy as np
from Bio import SeqIO

def one_hot_encode_variable_length(seq, max_len=7500):
    """
    Codifica una secuencia de ADN en formato one-hot, manejando longitudes variables.
    Si la secuencia es más larga que max_len, la trunca.
    Si es más corta, la rellena con ceros.
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_len = min(len(seq), max_len)
    encoded = np.zeros((max_len, 4))
    
    for i, base in enumerate(seq[:seq_len]):
        if base in mapping:
            encoded[i, mapping[base]] = 1
    
    return encoded

def preprocess_fasta_for_model(fasta_path, max_len=7500):
    """
    Preprocesa un archivo FASTA para ser utilizado por los modelos H-Net y Turbit.
    """
    record = SeqIO.read(fasta_path, "fasta")
    seq = str(record.seq).upper()
    encoded_seq = one_hot_encode_variable_length(seq, max_len)
    seq_tensor = torch.tensor(encoded_seq, dtype=torch.float32).unsqueeze(0) # (1, L, 4)
    return seq_tensor

if __name__ == "__main__":
    # Prueba con brca1_variante_1.fasta
    seq_tensor = preprocess_fasta_for_model("/home/ubuntu/upload/brca1_variante_1.fasta")
    print(f"Tensor shape: {seq_tensor.shape}")
    print(f"Non-zero elements: {torch.sum(seq_tensor > 0).item()}")

