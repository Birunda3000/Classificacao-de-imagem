# -*- coding: utf-8 -*-
"""
Script de Avaliação de Modelo Pós-Treino.

Este script é utilizado para carregar um modelo já treinado e avaliar a sua
performance no conjunto de dados de teste.

COMO USAR:
----------
Execute este script a partir do terminal, na pasta raiz do projeto.

1. Para avaliar a execução de treino mais recente (uso mais comum):
   -------------------------------------------------
   python evaluate.py
   -------------------------------------------------
   O script irá procurar automaticamente na pasta 'output/' pelo último treino
   realizado e avaliar o modelo salvo lá.

2. Para avaliar uma execução de treino específica:
   -----------------------------------------------------------------------------------------------
   python evaluate.py --run_dir output/NOME_DA_PASTA_DA_EXECUCAO
   -----------------------------------------------------------------------------------------------
   Substitua 'NOME_DA_PASTA_DA_EXECUCAO' pelo nome completo da pasta do treino
   que você deseja avaliar (ex: 'CIFAR10_CNN_acc_val=0.8765_20250722_103000').

O QUE O SCRIPT FAZ:
-------------------
- Carrega o modelo .h5 da pasta da execução.
- Carrega o conjunto de dados de teste usando o data_loader.
- Imprime a perda (loss) e a acurácia (accuracy) no conjunto de teste.
- Gera e exibe um relatório visual, como a matriz de confusão.

Script de Avaliação de Modelo Pós-Treino (Versão 2.0)

Gera um relatório de performance completo para os conjuntos de dados
de treino, teste e combinado, incluindo:
- Acurácia, Perda
- Relatório de Classificação (Precisão, Recall/Sensibilidade, F1-Score)
- Matriz de Confusão
"""
import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
import io # Para capturar a saída do print
import sys # Para redirecionar stdout
import matplotlib.pyplot as plt # Para manipular figuras do matplotlib

# Importa os nossos módulos
from src import config
from src import data_loader
from src import visualization

def get_latest_run_dir(output_dir: Path) -> Path:
    """
    Encontra o diretório da execução de treino mais recente que NÃO seja temporário.
    Procura por pastas que foram renomeadas após um treino bem-sucedido.
    """
    latest_run_dir = None
    latest_timestamp = None

    try:
        # Itera sobre os itens no diretório de saída
        for item in output_dir.iterdir():
            # Verifica se é um diretório e se o nome NÃO começa com '_tmp_'
            if item.is_dir() and not item.name.startswith('_tmp_'):
                try:
                    # Tenta extrair o timestamp do final do nome da pasta.
                    # Exemplo de nome: cifar10_CNN___val_acc=0.8550___2025-07-21_17h13m34s
                    parts = item.name.split('___')
                    
                    # Verifica se o nome tem a estrutura esperada para extrair o timestamp
                    if len(parts) >= 3:
                        timestamp_str = parts[-1] # A última parte deve ser o timestamp
                        
                        # Converte a string do timestamp para um objeto datetime para comparação
                        current_run_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%Hh%Mm%Ss")
                        
                        # Compara com o timestamp mais recente encontrado até agora
                        if latest_timestamp is None or current_run_timestamp > latest_timestamp:
                            latest_timestamp = current_run_timestamp
                            latest_run_dir = item
                except ValueError:
                    # Se o nome da pasta não puder ser parseado como um timestamp,
                    # significa que não segue o padrão de renomeação de sucesso, então ignora.
                    continue
    except FileNotFoundError:
        print(f"ERRO: Diretório de saída '{output_dir}' não encontrado.")
        exit()
    
    if latest_run_dir is None:
        print(f"ERRO: Nenhum diretório de execução finalizado encontrado em '{output_dir}'.")
        print("Certifique-se de que um treino foi concluído com sucesso e o diretório foi renomeado.")
        exit() # Termina o script se nenhum diretório válido for encontrado
        
    return latest_run_dir

def generate_performance_report(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset, 
    dataset_name: str,
    output_dir: Path # Adicionar este parâmetro para salvar imagens
) -> str: # Agora retorna a string do relatório
    """
    Gera um relatório de performance completo para um determinado dataset,
    salva a matriz de confusão como imagem e retorna o texto do relatório.
    """
    # Cria um buffer para capturar a saída do print
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    print("\n" + "-"*60)
    print(f"RELATÓRIO DE PERFORMANCE - CONJUNTO DE DADOS: {dataset_name.upper()}")
    print("-"*60)

    # Passo 1: Avaliação geral com model.evaluate()
    loss, accuracy = model.evaluate(dataset, verbose=0)
    print(f"\n  - Acurácia Geral (evaluate): {accuracy:.4f}")
    print(f"  - Perda Geral (evaluate): {loss:.4f}")

    # Passo 2: Previsões para métricas detalhadas
    print("\nGerando previsões para o relatório detalhado...")
    y_pred_probs = model.predict(dataset, verbose=0)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    
    # IMPORTANTE: Coletar todos os rótulos verdadeiros do dataset
    # Isso é necessário porque dataset.take(1) pega apenas 1 batch.
    # Devemos iterar sobre TODO o dataset para coletar todos os rótulos.
    y_true_labels_list = []
    for _, labels_one_hot in dataset:
        y_true_labels_list.append(np.argmax(labels_one_hot.numpy(), axis=1))
    y_true_labels = np.concatenate(y_true_labels_list, axis=0)


    # Passo 3: Relatório de Classificação (Precisão, Recall, F1-Score)
    print("\n  - Relatório de Classificação Detalhado:")
    report = classification_report(
        y_true_labels, 
        y_pred_labels, 
        target_names=config.CLASS_NAMES
    )
    print(report)

    # Passo 4: Matriz de Confusão
    print("Gerando Matriz de Confusão...")
    fig = visualization.plot_confusion_matrix(
        y_true_labels, 
        y_pred_labels,
        title=f"Matriz de Confusão - {dataset_name}"
    )
    
    # Salvar a figura em vez de exibir
    matrix_filename = f"confusion_matrix_{dataset_name.lower().replace(' ', '_')}.png"
    matrix_path = output_dir / matrix_filename
    fig.savefig(matrix_path)
    plt.close(fig) # Fechar a figura para liberar memória
    print(f"Matriz de Confusão salva em: {matrix_path}")

    # Restaura a saída padrão e retorna o texto capturado
    sys.stdout = old_stdout
    return redirected_output.getvalue()


def evaluate(run_dir: Path):
    """
    Orquestra o processo de avaliação completo, salvando relatórios em texto e imagens.
    """
    # Buffer para coletar todo o output do console
    full_report_text = io.StringIO()
    
    # Redireciona a saída do print para o buffer de texto
    old_stdout = sys.stdout
    sys.stdout = full_report_text

    print("="*60)
    print(f"INICIANDO AVALIAÇÃO PARA A EXECUÇÃO: {run_dir.name}")
    print("="*60)

    # --- Carregar o Modelo ---
    print("\n[ETAPA 1/3] Carregando o modelo treinado...")
    model_path = run_dir / "model.h5"
    if not model_path.exists():
        print(f"ERRO: Ficheiro do modelo não encontrado em '{model_path}'"); return
    model = tf.keras.models.load_model(model_path)

    # --- Carregar os Datasets ---
    print("\n[ETAPA 2/3] Carregando datasets de treino e teste...")

    train_ds_pipeline, num_train_images = data_loader.create_full_pipeline(config.PATH_TRAIN, config.Training.BATCH_SIZE, is_training=False)
    test_ds_pipeline, num_test_images = data_loader.create_full_pipeline(config.PATH_TEST, config.Training.BATCH_SIZE, is_training=False)
    
    # Cria um dataset combinado
    combined_ds = train_ds_pipeline.concatenate(test_ds_pipeline)

    # --- Gerar Relatórios ---
    print("\n[ETAPA 3/3] Gerando relatórios de performance...")
    
    # Coleta e salva o relatório de teste
    test_report_text = generate_performance_report(model, test_ds_pipeline, "Dados de Teste", run_dir)
    full_report_text.write(test_report_text) # Adiciona ao relatório completo
    
    # Coleta e salva o relatório de treino
    train_report_text = generate_performance_report(model, train_ds_pipeline, "Dados de Treino", run_dir)
    full_report_text.write(train_report_text) # Adiciona ao relatório completo
    
    # Coleta e salva o relatório combinado
    combined_report_text = generate_performance_report(model, combined_ds, "Dados Combinados (Treino + Teste)", run_dir)
    full_report_text.write(combined_report_text) # Adiciona ao relatório completo

    print("\n" + "="*60)
    print("AVALIAÇÃO CONCLUÍDA")
    print("="*60)

    # Restaura a saída padrão para o console
    sys.stdout = old_stdout
    
    # Salva o relatório completo em um arquivo de texto
    report_filename = run_dir / "evaluation_report.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(full_report_text.getvalue())
    print(f"\nRelatório de avaliação completo salvo em: {report_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Avaliação de Modelo.")
    parser.add_argument(
        '--run_dir', 
        type=str, 
        help="(Opcional) Caminho para a pasta da execução a ser avaliada."
    )
    args = parser.parse_args()

    if args.run_dir:
        run_to_evaluate = Path(args.run_dir)
    else:
        print("Nenhum diretório especificado. A procurar pela execução mais recente...")
        run_to_evaluate = get_latest_run_dir(config.Paths.OUTPUT_DIR)
    
    evaluate(run_to_evaluate)