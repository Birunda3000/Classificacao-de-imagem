# -*- coding: utf-8 -*-
"""
Módulo para funções de visualização do projeto.

Inclui funções para plotar o histórico de treino, matriz de confusão,
filtros da rede e mapas de ativação.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from typing import List, Any

# Importa as configurações do projeto
try:
    from . import config
except ImportError:
    import config

def plot_history(history: tf.keras.callbacks.History):
    """
    Plota a acurácia e a perda do treino e da validação.

    Args:
        history (tf.keras.callbacks.History): O objeto retornado por model.fit().
    Returns:
        matplotlib.figure.Figure: O objeto da figura do Matplotlib.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    fig = plt.figure(figsize=(16, 6)) # Captura a figura explícitamente
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Acurácia de Treino')
    plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
    plt.legend(loc='lower right')
    plt.title('Acurácia de Treino e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Perda de Treino')
    plt.plot(epochs_range, val_loss, label='Perda de Validação')
    plt.legend(loc='upper right')
    plt.title('Perda de Treino e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    
    plt.suptitle('Histórico de Treino', fontsize=16, y=1.02)
    
    # REMOVA ESTA LINHA:
    # plt.show()
    
    return fig  # Retorna a figura criada, não gcf()


def plot_confusion_matrix(
    y_true_labels: np.ndarray, 
    y_pred_labels: np.ndarray,
    title: str = "Matriz de Confusão Normalizada", # Adiciona o argumento de título
    figsize: tuple = (12, 10) # Adiciona um argumento para o tamanho da figura (opcional)
) -> plt.Figure: # Adiciona o tipo de retorno
    """
    Gera e retorna uma figura da matriz de confusão normalizada.

    Args:
        y_true_labels (np.ndarray): Array com os rótulos verdadeiros.
        y_pred_labels (np.ndarray): Array com os rótulos previstos.
        title (str): O título do gráfico.
        figsize (tuple): Tamanho da figura (largura, altura).

    Returns:
        matplotlib.figure.Figure: O objeto da figura do Matplotlib.
    """
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normaliza por linha

    fig, ax = plt.subplots(figsize=figsize) # Cria a figura e os eixos explicitamente
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt=".2%", 
        cmap='Blues', 
        xticklabels=config.CLASS_NAMES, 
        yticklabels=config.CLASS_NAMES,
        ax=ax # Passa os eixos para o heatmap
    )
    ax.set_title(title, fontsize=16) # Usa o título passado como argumento
    ax.set_ylabel('Rótulo Verdadeiro')
    ax.set_xlabel('Rótulo Previsto')
    plt.tight_layout() # Ajusta o layout para evitar sobreposição

    # REMOVA ESTA LINHA:
    # plt.show()

    return fig # RETORNE O OBJETO DA FIGURA


def plot_model_filters(model: Model, layer_name: str, max_filters: int = 16) -> plt.Figure:
    """
    Visualiza os filtros (kernels) de uma camada convolucional específica e retorna a figura.

    Args:
        model (Model): O modelo Keras treinado.
        layer_name (str): O nome da camada convolucional a ser visualizada.
        max_filters (int): Número máximo de filtros a serem exibidos.
    Returns:
        matplotlib.figure.Figure: O objeto da figura do Matplotlib.
    """
    try:
        layer = model.get_layer(name=layer_name)
        filters, biases = layer.get_weights()

        # Normaliza os filtros para o intervalo [0, 1] para visualização
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        num_filters = min(filters.shape[3], max_filters)
        
        # Calcula o grid de plotagem
        n_cols = 8
        n_rows = num_filters // n_cols + (1 if num_filters % n_cols else 0)

        fig = plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5)) # Captura a figura explicitamente
        for i in range(num_filters):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            # Plota o i-ésimo filtro do kernel
            ax.imshow(filters[:, :, :, i])
        
        fig.suptitle(f"Filtros da Camada '{layer_name}'", fontsize=16, y=1.02)
        plt.tight_layout() # Ajusta o layout
        
        # REMOVA ESTA LINHA:
        # plt.show()
        
        return fig # Retorna a figura criada

    except Exception as e:
        print(f"Não foi possível plotar os filtros para a camada '{layer_name}': {e}")
        return None # Retorna None em caso de erro


def plot_feature_maps(model: Model, layer_names: List[str], image: np.ndarray) -> List[plt.Figure]:
    """
    Visualiza os mapas de características (ativações) para uma imagem de entrada
    e retorna uma lista de figuras.

    Args:
        model (Model): O modelo Keras treinado.
        layer_names (List[str]): Lista com os nomes das camadas a serem visualizadas.
        image (np.ndarray): A imagem de entrada (deve ter o shape esperado pelo modelo, ex: (1, 32, 32, 3)).
    Returns:
        List[matplotlib.figure.Figure]: Uma lista de objetos de figura do Matplotlib.
    """
    # Cria um sub-modelo que retorna as ativações das camadas desejadas
    activation_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.inputs, outputs=activation_outputs)
    
    # Obtém as ativações
    activations = activation_model.predict(image)
    
    figures = [] # Lista para armazenar as figuras

    for layer_name, layer_activation in zip(layer_names, activations):
        num_features = layer_activation.shape[-1]
        
        n_cols = 8
        n_rows = num_features // n_cols + (1 if num_features % n_cols else 0)

        fig = plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5)) # Captura a figura explicitamente
        for i in range(num_features):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            # Plota o i-ésimo mapa de características
            ax.imshow(layer_activation[0, :, :, i], cmap='viridis')

        fig.suptitle(f"Mapas de Características da Camada '{layer_name}'", fontsize=16, y=1.02)
        plt.tight_layout() # Ajusta o layout
        
        # REMOVA ESTA LINHA:
        # plt.show()
        
        figures.append(fig) # Adiciona a figura à lista
        
    return figures # Retorna a lista de figuras