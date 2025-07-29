import os
import numpy as np
import tensorflow as tf
import pickle
import numpy as np


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.12, shuffle=True, shuffle_size=1000):
    # Verifique se a soma das divisões é menor que 1
    assert (train_split + val_split) < 1

    # Obtém o tamanho do dataset usando cardinality
    ds_size = tf.data.experimental.cardinality(ds).numpy()

    if ds_size == tf.data.experimental.INFINITE_CARDINALITY or ds_size == -2:
        raise ValueError("O tamanho do dataset é desconhecido. Certifique-se de não usar `repeat()` sem um número definido.")

    # Embaralha o dataset se necessário
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    # Define o tamanho dos conjuntos de treino, validação e teste
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    # Divide o dataset em treino, validação e teste
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size)
    
    print(f"Tamanho do treino: {len(list(train_ds))}")
    print(f"Tamanho da validação: {len(list(val_ds))}")
    print(f"Tamanho do teste: {len(list(test_ds))}")

    return (
        train_ds.cache().shuffle(500).prefetch(buffer_size=tf.data.AUTOTUNE),
        val_ds.cache().shuffle(500).prefetch(buffer_size=tf.data.AUTOTUNE),
        test_ds.cache().shuffle(500).prefetch(buffer_size=tf.data.AUTOTUNE)
    )
