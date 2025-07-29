import os, shutil, torch, numpy as np, tensorflow as tf

from torch.utils.data import DataLoader, TensorDataset

from src.partitions_dataset import get_dataset_partitions_tf
from src.data_augmentation import augmentation_e_gerar_dataset
from src.utils_dataset import save_dataset_to_pickle, load_dataset_from_pickle
from src.preprocessamento import preprocess_image


def gerar_dataset(args, original_dir, selected_classes, augmentation_datagen, standard_datagen, input_shape):

    filtered_dir = os.path.join("dataset_filtrado")
    os.makedirs(filtered_dir, exist_ok=True)

    for class_name in selected_classes:
        src = os.path.join(original_dir, class_name)
        dst = os.path.join(filtered_dir, class_name)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    IMAGE_SIZE_H, IMAGE_SIZE_W, CHANNELS = input_shape

    # Balancear o dataset
    dataset = augmentation_e_gerar_dataset(filtered_dir, standard_datagen = standard_datagen, augmentation_datagen = augmentation_datagen, target_resolution=(IMAGE_SIZE_H, IMAGE_SIZE_W))
    dataset = dataset.shuffle(len(dataset)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # LIMPAR O DIRETÓRIO APÓS O CARREGAMENTO
    if os.path.exists(filtered_dir):
        shutil.rmtree(filtered_dir)
        print(f"Pasta '{filtered_dir}' foi apagada após o carregamento do dataset.")

    dataset = preprocess_image(dataset, IMAGE_SIZE_H, IMAGE_SIZE_W)

    train_ds_tf, val_ds_tf, test_ds_tf = get_dataset_partitions_tf(dataset)

    # Salvar datasets com pickle
    os.makedirs("datasets_utilizados", exist_ok=True)
    save_dataset_to_pickle(train_ds_tf, os.path.join("datasets_utilizados", "train_dataset.pkl"))
    save_dataset_to_pickle(val_ds_tf, os.path.join("datasets_utilizados", "val_dataset.pkl"))
    save_dataset_to_pickle(test_ds_tf, os.path.join("datasets_utilizados", "test_dataset.pkl"))


def converter_dataset_para_pytorch(tf_dataset, input_shape, BATCH_SIZE):
    IMAGE_SIZE_H, IMAGE_SIZE_W, CHANNELS = input_shape
    imagens, rotulos = [], []

    for batch_imgs, batch_labels in tf_dataset:
        batch_imgs = tf.image.resize(batch_imgs, [IMAGE_SIZE_H, IMAGE_SIZE_W])  # Redimensionar batch inteiro
        batch_imgs = batch_imgs.numpy()  # Converter para numpy

        for img_resized, label in zip(batch_imgs, batch_labels):
            # Garantir que a imagem tenha 3 canais
            if img_resized.ndim == 2:
                img_resized = np.stack([img_resized] * 3, axis=-1)
            elif img_resized.ndim == 3:
                if img_resized.shape[-1] == 1:
                    img_resized = np.concatenate([img_resized] * 3, axis=-1)
                elif img_resized.shape[-1] == 4:
                    img_resized = img_resized[:, :, :3]
            else:
                raise ValueError(f"Imagem com formato inesperado: {img_resized.shape}")

            # Converter para tensor e ajustar para [C, H, W]
            img_tensor = torch.tensor(img_resized, dtype=torch.float32)
            
            if img_tensor.ndim != 3:
                raise ValueError(f"Tentativa de permutar tensor com shape inválido: {img_tensor.shape}")

            imagens.append(img_tensor.permute(2, 0, 1))  # [C, H, W]
            rotulos.append(torch.tensor(int(label), dtype=torch.long))


    # Garantir que todas as imagens tenham o mesmo tamanho
    min_shape = imagens[0].shape
    imagens = [img if img.shape == min_shape else torch.zeros_like(imagens[0]) for img in imagens]

    dataset_pytorch = TensorDataset(torch.stack(imagens), torch.stack(rotulos))
    return DataLoader(dataset_pytorch, batch_size=BATCH_SIZE, shuffle=True)


# def rodar_dataset_torch(input_shape, BATCH_SIZE, caminho_base = 'datasets_utilizados'):
#     train_ds_tf = load_dataset_from_pickle(os.path.join(caminho_base,"train_dataset.pkl"), BATCH_SIZE)
#     val_ds_tf = load_dataset_from_pickle(os.path.join(caminho_base, "val_dataset.pkl"), BATCH_SIZE)
#     test_ds_tf = load_dataset_from_pickle(os.path.join(caminho_base, "test_dataset.pkl"), BATCH_SIZE)

#     # Converter datasets para PyTorch DataLoader
#     train_ds = converter_dataset_para_pytorch(train_ds_tf, input_shape, BATCH_SIZE)
#     val_ds = converter_dataset_para_pytorch(val_ds_tf, input_shape, BATCH_SIZE)
#     test_ds = converter_dataset_para_pytorch(test_ds_tf, input_shape, BATCH_SIZE)
    
#     return train_ds, val_ds, test_ds




def rodar_dataset_torch(input_shape, BATCH_SIZE, caminho_base='datasets_utilizados', filter_test_labels=None):
    train_ds_tf = load_dataset_from_pickle(os.path.join(caminho_base, "train_dataset.pkl"), BATCH_SIZE)
    val_ds_tf = load_dataset_from_pickle(os.path.join(caminho_base, "val_dataset.pkl"), BATCH_SIZE)
    test_ds_tf = load_dataset_from_pickle(os.path.join(caminho_base, "test_dataset.pkl"), BATCH_SIZE)

    # Filter test dataset if required
    if filter_test_labels is not None:
        def filter_fn(x, y):
            return tf.reduce_any(tf.equal(y, filter_test_labels))
        test_ds_tf = test_ds_tf.filter(filter_fn)

    # Convert datasets to PyTorch DataLoader
    train_ds = converter_dataset_para_pytorch(train_ds_tf, input_shape, BATCH_SIZE)
    val_ds = converter_dataset_para_pytorch(val_ds_tf, input_shape, BATCH_SIZE)
    test_ds = converter_dataset_para_pytorch(test_ds_tf, input_shape, BATCH_SIZE)
    
    return train_ds, val_ds, test_ds




def rodar_dataset_torch2(input_shape, BATCH_SIZE, caminho_base='datasets_utilizados', filter_test_labels=None):
    train_ds_tf = load_dataset_from_pickle(os.path.join(caminho_base, "train_dataset.pkl"), BATCH_SIZE)
    val_ds_tf = load_dataset_from_pickle(os.path.join(caminho_base, "val_dataset.pkl"), BATCH_SIZE)
    test_ds_tf = load_dataset_from_pickle(os.path.join(caminho_base, "test_dataset.pkl"), BATCH_SIZE)

    # Filter test dataset if required
    if filter_test_labels is not None:
        def filter_fn(x, y):
            # Compare each scalar label to allowed_labels vector
            allowed_labels = tf.constant(filter_test_labels, dtype=y.dtype)
            is_allowed = tf.reduce_any(tf.equal(y, allowed_labels))
            return is_allowed
        
        # Unbatch -> Filter individual examples -> Rebatch
        test_ds_tf = test_ds_tf.unbatch().filter(filter_fn).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Convert datasets to PyTorch DataLoader
    train_ds = converter_dataset_para_pytorch(train_ds_tf, input_shape, BATCH_SIZE)
    val_ds = converter_dataset_para_pytorch(val_ds_tf, input_shape, BATCH_SIZE)
    test_ds = converter_dataset_para_pytorch(test_ds_tf, input_shape, BATCH_SIZE)
    
    return train_ds, val_ds, test_ds