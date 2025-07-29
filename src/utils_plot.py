import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def plot_images_from_dataset(dataset, images_per_row=10, max_images=100):
        """
        Plota até 100 imagens e rótulos do dataset em um layout de grade otimizado.

        Args:
            dataset (tf.data.Dataset): Dataset contendo imagens e rótulos.
            images_per_row (int): Número de imagens por linha na grade.
            max_images (int): Número máximo de imagens a serem plotadas.
        """
        import matplotlib.pyplot as plt
        import math

        all_images = []
        all_labels = []

        # Extraindo imagens e rótulos com limite de 100
        for images, labels in dataset:
            for img, lbl in zip(images, labels):
                all_images.append(img.numpy())
                all_labels.append(lbl.numpy())
                
                if len(all_images) >= max_images:
                    break
            if len(all_images) >= max_images:
                break

        total_images = len(all_images)
        rows = math.ceil(total_images / images_per_row)

        # Ajuste do tamanho da figura para reduzir o espaçamento
        plt.figure(figsize=(images_per_row * 1.5, rows * 1.5))  # Reduzido para diminuir espaçamento

        for idx, (image, label) in enumerate(zip(all_images, all_labels)):
            # Normalizando a imagem se necessário
            if image.max() <= 1.0:
                image = (image * 255).astype("uint8")

            # Ajuste do formato (C, H, W) -> (H, W, C)
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)

            plt.subplot(rows, images_per_row, idx + 1)
            plt.imshow(image)
            plt.title(f"{label}", fontsize=8)  # Tamanho da fonte ajustado
            plt.axis("off")

        # Reduz o espaçamento entre subplots
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        plt.tight_layout()
        plt.show()


def visualizar_amostras_dataset(dataloader, num_amostras=5):
    """
    Visualiza as primeiras imagens do DataLoader junto com seus rótulos e imprime o valor máximo dos pixels.
    """
    imagens, rotulos = next(iter(dataloader))  # Pega o primeiro batch
    for i in range(min(num_amostras, len(imagens))):
        img = imagens[i].permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        rotulo = rotulos[i].item()

        # Verificar o valor máximo do pixel
        max_valor_pixel = np.max(img)
        print(f"Imagem {i+1} - Rótulo: {rotulo} - Valor máximo do pixel: {max_valor_pixel}")

        # Ajustar escala se os valores estiverem entre 0 e 1
        if img.max() <= 1.0:
            img = img * 255.0

        # Garantir que os valores estejam no intervalo [0, 255] e no tipo correto
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Plotar imagem
        plt.figure(figsize=(2, 2))
        plt.imshow(img)
        plt.title(f"Rótulo: {rotulo}")
        plt.axis('off')
        plt.show()


def print_dataset_shape(dataloader, name="Dataset"):
    for images, labels in dataloader:
        print(f"{name} - Images Shape: {images.shape}, Labels Shape: {labels.shape}")
        break  # Print only the first batch shape to avoid excessive output


# Função para plotar métricas simples
def plot_single_metric(num, histories, metric, title, ylabel):
    plt.figure(figsize=(10, 6))
    for model_name, history in histories.items():
        plt.plot(history[metric], label=f'{model_name} - {metric}')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    # Cria o diretório se não existir
    os.makedirs(os.path.join('tcc_documento','plot'), exist_ok=True)
    
    # Salva o plot
    filename = os.path.join('tcc_documento','plot',f'{title}{num}.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()


# Função para plotar métricas duplas
def plot_dual_metric(histories, metric1, metric2, title, ylabel1, ylabel2):
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    
    for model_name, history in histories.items():
        ax1.plot(history[metric1], label=f'{model_name} - {metric1}', linestyle='-', alpha=0.7)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(ylabel1, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title(title)
    ax1.grid()

    ax2 = ax1.twinx()
    for model_name, history in histories.items():
        ax2.plot(history[metric2], label=f'{model_name} - {metric2}', linestyle='--', alpha=0.7)
    ax2.set_ylabel(ylabel2, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    plt.show()