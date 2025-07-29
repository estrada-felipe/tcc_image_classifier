
import os, argparse, torch, random, copy, torch.nn as nn, numpy as np

from torchsummary import summary
from keras.preprocessing.image import ImageDataGenerator
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset_gen_load import gerar_dataset, rodar_dataset_torch
from src.utils_plot import print_dataset_shape, visualizar_amostras_dataset
from src.utils_model import EarlyStopping, prune, salvar_modelo_completo
from src.gen_models import ModelFactory_Tiny

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# SEED = 12345

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

gerar_dtset = True
BATCH_SIZE = 32                         #16
IMAGE_SIZE_H, IMAGE_SIZE_W = 96,96      # 128,128
CHANNELS = 3
EPOCHS = 100
MODELO_DEFAULT = 'EfficientNet'
input_shape = (IMAGE_SIZE_H, IMAGE_SIZE_W, CHANNELS)

data_augmentation_config = {
    'rotation_range': 10,          # Rotaçao
    'width_shift_range': 0.07,     # Deslocamento horizontal
    'height_shift_range': 0.07,    # Deslocamento vertical
    'shear_range': 0.1,            # Distorçao angular
    'zoom_range': 0.15,            # Zoom aumentado para ±15%
    'horizontal_flip': True,       # Mantém o flip horizontal
    'fill_mode': 'nearest'         # Preenchimento
}

augmentation_datagen = ImageDataGenerator(**data_augmentation_config)
standard_datagen = ImageDataGenerator()

def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento do modelo de detecção de catarata")
    parser.add_argument('--modelo', type=str, default=MODELO_DEFAULT, help="Escolha o modelo: EfficientNet, ResNet, Vision Transformer, ViTlite, CVT, CCT, CoatNet")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Número de épocas")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Tamanho do batch")
    return parser.parse_args()


def main(args, modelos):
    history = {'train_loss': [],'train_accuracy': [],'val_loss': [],'val_accuracy': []}
    selected_classes2 = ['1_normal', '2_cataract']
    selected_classes = ['1_normal', '2_cataract','2_glaucoma', '3_retina_disease']
    original_dir = os.path.join("dataset")
    os.makedirs('modelos', exist_ok=True)

    classes = '4_Classes' if len(selected_classes) == 4 else '2_Classes'
    if args.modelo == modelos[0] and gerar_dtset == True:
        gerar_dataset(args, original_dir, selected_classes, augmentation_datagen, standard_datagen, input_shape)    
    # train_ds, val_ds, test_ds = rodar_dataset_torch(input_shape, BATCH_SIZE, caminho_base = os.path.join(classes,'x','datasets_utilizados'))
    train_ds, val_ds, test_ds = rodar_dataset_torch(input_shape, BATCH_SIZE, caminho_base = os.path.join('datasets_utilizados'))

    factory = ModelFactory_Tiny(len(selected_classes), input_shape, pretrained = False)
    model = factory.create_model(args.modelo)   

    # Configuração do dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6) #5e-5 ou 1e-5 tambem
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-6)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, verbose=True)
    if args.modelo.lower() in [
        'vision transformer',   "vitlite",  "cvt",  "cct",        
        'resnet10vit',  'resnet10vit2', 'resnet18vit',
        'resnet18vit2', 'resnet6vit',   'resnet6vit']:
        early_stopping = EarlyStopping(patience=20, min_delta=0.001)
    else:
        early_stopping = EarlyStopping(patience=10, min_delta=0.01)

    config = {
        'batch_size': BATCH_SIZE,
        'image_size_h': IMAGE_SIZE_H,
        'image_size_w': IMAGE_SIZE_W,
        'channels': CHANNELS,
        'input_shape': input_shape,
        'num_classes': len(selected_classes),
        'EPOCHS': args.epochs,
        'architecture': args.modelo,
        'optimizer': 'AdamW',
        'learning_rate': optimizer.param_groups[0]['lr'],
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_factor': 0.5,
        'scheduler_patience': 3,
        'loss_function': 'CrossEntropyLoss',
        'embed_dim': getattr(model, 'embed_dim', None),
        'depth': getattr(model, 'depth', None),
        'num_heads': getattr(model, 'num_heads', None),
        'kernel_size': getattr(model, 'kernel_size', None),
        'num_conv_layers': getattr(model, 'num_conv_layers', None),
        'seq_pool': getattr(model, 'seq_pool', False),
        'dropout': getattr(model, 'dropout', None),
    } 

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        print(f"\n--- Época [{epoch+1}/{args.epochs}] ---")

        for batch_idx, (images, labels) in enumerate(train_ds):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_loss = loss.item()
            batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)

            # Exibir métricas a cada batch
            print(f"Batch [{batch_idx+1}/{len(train_ds)}] - Loss: {batch_loss:.4f}, Accuracy: {batch_acc:.2f}%")

        # Métricas da época
        epoch_loss = running_loss / len(train_ds)
        epoch_acc = 100 * correct / total
        print(f"\n>>> Época [{epoch+1}/{args.epochs}] Finalizada - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Validação ao final de cada época
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_ds:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss_avg = val_loss / len(val_ds)
        val_acc = 100 * val_correct / val_total
        print(f"Validação - Loss: {val_loss / len(val_ds):.4f}, Accuracy: {val_acc:.2f}%")


        modelo_epoch_path = os.path.join('modelos', f'modelo_{args.modelo.lower()}_epoca_{epoch+1}.pth')
        modelo_anterior_path = os.path.join('modelos', f'modelo_{args.modelo.lower()}_epoca_{epoch}.pth')
        if epoch > 0 and os.path.exists(modelo_anterior_path):
            os.remove(modelo_anterior_path)
            print(f"🗑️ Modelo da época {epoch} removido: {modelo_anterior_path}")
        torch.save(model.state_dict(), modelo_epoch_path)
        print(f"📦 Modelo salvo ao final da época {epoch+1}: {modelo_epoch_path}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"🔍 Learning Rate atual: {current_lr:.10f}")
        scheduler.step(val_loss)
        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Treinamento interrompido por EarlyStopping.")
            break

        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

    modelo_ultima_epoca_path = os.path.join('modelos', f'modelo_{args.modelo.lower()}_epoca_{epoch+1}.pth')
    if os.path.exists(modelo_ultima_epoca_path):
        os.remove(modelo_ultima_epoca_path)
        print(f"🗑️ Modelo da última época removido: {modelo_ultima_epoca_path}")

    salvar_modelo_completo(args, model, optimizer, args.modelo.lower(), epoch, history, config)




if __name__ == "__main__":  
    try: args = parse_args()
    except SystemExit:
        class Args:
            modelo = MODELO_DEFAULT
            epochs = EPOCHS
            batch_size = BATCH_SIZE
        args = Args()

    modelos = ['EfficientNet', 'ResNet18', 'ResNet10','ResNet8',
        'ResNet6','cvt','cct','ResNet18ViT','ResNet10ViT','ResNet8ViT','ResNet6ViT']

    # for modelo in modelos:
    #     print(f"\nTreinando modelo: {modelo}\n")
    #     args.modelo = modelo
    #     main(args, modelos)

    main(args, modelos)
