import os, pickle, json, torch, torch.nn as nn, torch.nn.utils.prune as prune
from torchsummary import summary


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


from src.gen_models import ModelFactory_Tiny
from contextlib import redirect_stdout


class EarlyStopping:
    """Para interromper o treinamento quando a validação não melhora após 'patience' épocas."""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

         
def aplicar_pruning(model, amount=0.3):
    """
    Aplica pruning (poda) nas camadas convolucionais e lineares.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Remove a máscara após o pruning
    print(f"Pruning aplicado com {amount*100}% de pesos podados.")
    return model


def inspect_models(ModelFactory, arch, num_classes, input_shape):
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    factory = ModelFactory(num_classes=num_classes, input_shape=input_shape, pretrained=False)
    IMAGE_SIZE_H, IMAGE_SIZE_W, CHANNELS = input_shape
       
    print(f"\n{'='*60}\n🔍 Arquitetura: {arch}\n{'='*60}")
    model = factory.create_model(arch)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")

    try:
        summary(model, input_size=(CHANNELS, IMAGE_SIZE_H, IMAGE_SIZE_W))
    except Exception as e:
        print(f"Não foi possível gerar o resumo do modelo: {e}")


def salvar_modelo_completo(args, model, optimizer, modelo_nome, epoch, history, config):
    modelo_dir = os.path.join('modelos', modelo_nome)
    os.makedirs(modelo_dir, exist_ok=True)

    resumo_modelo_path = os.path.join(modelo_dir, 'resumo_modelo.txt')

    with open(resumo_modelo_path, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            inspect_models(ModelFactory_Tiny, config['architecture'], config['num_classes'], config['input_shape'])


    # Caminho para o modelo
    modelo_final_path = os.path.join(modelo_dir, f'modelo_final_epoca_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config
    }, modelo_final_path)

    print(f"Modelo completo salvo: {modelo_final_path}")

    # Salvar histórico separadamente (opcional)
    with open(os.path.join(modelo_dir, 'historico.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print("Histórico de treinamento salvo.")

    os.makedirs('modelos/historico', exist_ok=True)
    with open(f'modelos/historico/historico_{args.modelo.lower()}.pkl', 'wb') as f:
        pickle.dump(history, f)

    # Salvar configurações em JSON
    with open(os.path.join(modelo_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    print("Configurações do modelo salvas.")


def listar_modelos(diretorio="modelos", excluir=None):
    # Define os valores padrão
    if excluir is None:
        excluir = []
    
    # Adiciona os valores padrão obrigatórios
    excluir.extend(["Anteriores", "historico"])
    
    # Remove duplicatas, caso haja
    excluir = list(set(excluir))
    
    # Filtra as subpastas
    subpastas = [
        nome for nome in os.listdir(diretorio)
        if os.path.isdir(os.path.join(diretorio, nome)) and nome not in excluir
    ]
    return subpastas


# Função para carregar o histórico ajustado ao novo formato
def load_history(classe, model_name):
    history_path = os.path.join(classe, "modelos", model_name, "historico.pkl")
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    return history

import time
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_metrics(conf_matrix):
    num_classes = conf_matrix.shape[0]
    sensitivities = []
    specificities = []
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        tn = np.sum(conf_matrix) - (tp + fn + fp)
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    return avg_sensitivity, avg_specificity




def plot_confusion_matrix(true_labels, predicted_labels, class_names, model_name):
    """
    Plota a matriz de confusão.
    
    Args:
        true_labels (list): Lista de labels reais.
        predicted_labels (list): Lista de labels preditos pelo modelo.
        class_names (list): Lista de nomes das classes.
        model_name (str): Nome do modelo avaliado.
    """
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.show()



def avaliar_modelo_no_teste(model, test_ds, modl, class_names, plot_conf=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    inference_times = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_ds:
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            inference_time = end_time - start_time
            inference_times.append(inference_time)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    avg_sensitivity, avg_specificity = compute_metrics(conf_matrix)

    avg_loss = test_loss / len(test_ds)
    accuracy = 100 * correct / total
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    print(f"--- Treinamento {modl} ---")
    print(f"Teste - Perda: {avg_loss:.4f}, Precisão: {accuracy:.2f}%, Sensibilidade: {avg_sensitivity:.4f}, Especificidade: {avg_specificity:.4f}, Tempo Medio Inferencia (s): {avg_inference_time:.6f}s")

    if plot_conf == True:plot_confusion_matrix(all_labels, all_preds, class_names, modl)

def print_history_values(histories, model_name):
    print(f"--- {model_name} History ---")
    for key, values in histories.items():
        print(f"{key}: {values}")


def print_last_history_values(histories, model_name):
    print(f"--- Últimos valores do histórico para {model_name} ---")
    for key, values in histories.items():
        if values:  # Verifica se a lista de valores não está vazia
            last_value = values[-1]  # Pega o último valor da lista
            print(f"{key}: {last_value}")
        else:
            print(f"{key}: Nenhum valor disponível")




def adjust_pos_embed(pos_embed_checkpoint, current_pos_embed):
    """
    Ajusta os embeddings posicionais do checkpoint para o tamanho esperado pelo modelo atual.
    """
    if pos_embed_checkpoint.shape != current_pos_embed.shape:
        print(f"Ajustando os embeddings posicionais de {pos_embed_checkpoint.shape} para {current_pos_embed.shape}")
        pos_embed_checkpoint = nn.Parameter(
            torch.nn.functional.interpolate(
                pos_embed_checkpoint.unsqueeze(0),  # Adiciona dimensão do batch
                size=current_pos_embed.shape[1],    # Ajusta o número de tokens esperado
                mode='linear',                     # Interpolação linear
                align_corners=False
            )
        ).squeeze(0)  # Remove a dimensão do batch após a interpolação
    return pos_embed_checkpoint


def carregar_modelo(modelo_path):
    checkpoint = torch.load(modelo_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    config = checkpoint['config']
    
    factory = ModelFactory_Tiny(config['num_classes'], config['input_shape'], pretrained=False)
    model = factory.create_model(config['architecture'])
    
    checkpoint_state = checkpoint['model_state_dict']
    current_state = model.state_dict()

    if 'pos_embed' in checkpoint_state and 'pos_embed' in current_state:
        checkpoint_state['pos_embed'] = adjust_pos_embed(checkpoint_state['pos_embed'], current_state['pos_embed'])

    
    model.load_state_dict(checkpoint_state, strict=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Modelo carregado com sucesso da arquitetura: {config['architecture']}")
    return model, optimizer, config


def carregar_modelo_mais_recente(modelo_nome, modelo_dir, classe = ''):


    # Verifica se o diretório existe
    if not os.path.exists(modelo_dir):
        print(f"O diretório do modelo '{modelo_nome}' não foi encontrado.")
        return None, None, None

    # Lista todos os arquivos .pth e filtra pelos modelos salvos
    modelos_salvos = [f for f in os.listdir(modelo_dir) if f.endswith('.pth') and 'modelo_final' in f]

    if not modelos_salvos:
        print(f"Nenhum modelo salvo encontrado para '{modelo_nome}'.")
        return None, None, None

    # Ordena pelos números de época (extraído do nome do arquivo)
    modelos_salvos.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)

    # Seleciona o modelo com a maior época (mais recente)
    modelo_mais_recente = modelos_salvos[0]
    modelo_path = os.path.join(modelo_dir, modelo_mais_recente)

    # Carrega o modelo
    modelo, otimizador, config = carregar_modelo(modelo_path)
    print(f"Modelo mais recente carregado: {modelo_mais_recente}")
    return modelo, otimizador, config