import os
import numpy as np
import tensorflow as tf
import numpy as np

def augmentation_e_gerar_dataset(data_dir, standard_datagen, augmentation_datagen, target_resolution=(128, 128)):
    def get_class_counts(directory):
        return {cls: len(os.listdir(os.path.join(directory, cls))) for cls in os.listdir(directory)}
    class_counts = get_class_counts(data_dir)
    max_class_count = max(class_counts.values())
    print("Distribuição inicial por classes:", class_counts)
    print("Tamanho da maior classe:", max_class_count)

    balanced_images = []
    balanced_labels = []

    class_indices = {cls: idx for idx, cls in enumerate(sorted(class_counts.keys()))}
    print(f"Mapeamento de classes: {class_indices}")

    for class_label, count in class_counts.items():
        class_dir = os.path.join(data_dir, class_label)

        if not os.path.exists(class_dir) or not os.listdir(class_dir):
            print(f"A pasta {class_dir} está vazia ou não existe. Pule essa classe.")
            continue

        num_images_to_generate = max_class_count - count
        label_idx = class_indices[class_label]

        generator = standard_datagen.flow_from_directory(
            data_dir,
            target_size=target_resolution,
            batch_size=1,
            classes=[class_label],
            class_mode='sparse',
            shuffle=True
        )

        for _ in range(count):
            try:
                img, label = next(generator)
                if img is not None and len(img) > 0:
                    img_resized = tf.image.resize(img[0], target_resolution).numpy()
                    img_resized = np.squeeze(img_resized)
                    balanced_images.append(img_resized)
                    balanced_labels.append(label_idx)
            except StopIteration:
                print("O gerador terminou sem produzir dados suficientes.")
                break

        if num_images_to_generate > 0:
            augmentation_generator = augmentation_datagen.flow_from_directory(
                data_dir,
                target_size=target_resolution,
                batch_size=1,
                classes=[class_label],
                class_mode='sparse',
                shuffle=True
            )
            for _ in range(num_images_to_generate):
                try:
                    img, label = next(augmentation_generator)
                    if img is not None and len(img) > 0:
                        img_resized = tf.image.resize(img[0], target_resolution).numpy()
                        img_resized = np.squeeze(img_resized)
                        balanced_images.append(img_resized)
                        balanced_labels.append(label_idx)
                except StopIteration:
                    print(f"Augmentation gerador terminou antes de gerar {num_images_to_generate} imagens.")
                    break

    balanced_images = np.array(balanced_images)
    balanced_labels = np.array(balanced_labels)

    if balanced_images.size == 0 or balanced_labels.size == 0:
        raise ValueError("O dataset balanceado está vazio. Verifique os diretórios de entrada.")

    unique, counts = np.unique(balanced_labels, return_counts=True)
    print(f"Distribuição final de classes: {dict(zip(unique, counts))}")

    balanced_dataset = tf.data.Dataset.from_tensor_slices((balanced_images, balanced_labels))
    return balanced_dataset
