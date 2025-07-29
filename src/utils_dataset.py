import tensorflow as tf
import pickle
import numpy as np
from collections import Counter

def save_dataset_to_pickle(dataset, output_path):
    """
    Salva um dataset como um arquivo pickle, preservando as tuplas (features, labels).
    Remove a estrutura de batch antes de salvar.
    """
    data = []
    for features_batch, labels_batch in dataset:
        for feature, label in zip(features_batch.numpy(), labels_batch.numpy()):
            data.append((feature, label))  # Salva exemplos individuais
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Dataset salvo em {output_path}.")


def load_dataset_from_pickle(input_path, batch_size):
    """
    Carrega um dataset salvo em um arquivo pickle.
    Aplica batching durante o carregamento.
    """
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    print(f"Dataset carregado de {input_path}.")
    
    # Reconstrói o dataset a partir das tuplas (features, labels)
    features, labels = zip(*data)
    features = tf.convert_to_tensor(features)
    labels = tf.convert_to_tensor(labels)
    
    # Aplica batching aqui
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Função para analisar a distribuição dos rótulos
def analyze_label_distribution(dataset):
    all_labels = []
    for _, labels in dataset:
        all_labels.extend(labels.numpy())

    label_counts = Counter(all_labels)
    print("Distribuição dos rótulos:")
    for label, count in label_counts.items():
        print(f"Rótulo {label}: {count} exemplos")
