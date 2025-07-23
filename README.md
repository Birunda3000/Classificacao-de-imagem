# Projeto de Classificação de Imagem com TensorFlow e Docker

Este repositório contém uma estrutura modular e robusta para projetos de classificação de imagem, utilizando TensorFlow e Keras. O ambiente de desenvolvimento é totalmente containerizado com Docker e otimizado para o uso de GPUs NVIDIA, garantindo reprodutibilidade e performance.

## ✨ Funcionalidades Principais

- [cite_start]**Ambiente Containerizado:** Utiliza Docker e VS Code Dev Containers para uma configuração de ambiente rápida, consistente e isolada.
- [cite_start]**Estrutura Modular:** O código é organizado em módulos com responsabilidades únicas (`config`, `data_loader`, `model_architectures`, `reporting`, etc.), seguindo as melhores práticas de MLOps.
- [cite_start]**Configuração Centralizada:** Todas as variáveis e hiperparâmetros do projeto são geridos a partir de um único ficheiro (`src/config.py`), facilitando a experimentação.
- [cite_start]**Pipeline de Dados Eficiente:** Carregamento de dados otimizado com `tf.data`, permitindo trabalhar com datasets grandes que não cabem na memória e acelerando o treino com pré-carregamento (`prefetch`).
- [cite_start]**Scripts Dedicados:** Scripts de alto nível para orquestrar o treino (`train.py`) e a avaliação (`evaluate.py`).
- [cite_start]**Geração Automática de Artefactos:** Cada treino gera uma pasta de resultados única, contendo o modelo treinado, um gráfico do histórico e um relatório detalhado com métricas e tempos de execução.
- [cite_start]**Modo de Depuração:** Permite executar o pipeline completo com um pequeno subconjunto de dados para testes e depuração rápidos.
- [cite_start]**Suporte a Otimizações Avançadas:** O ambiente está configurado para usar acelerações como Mixed Precision e XLA (JIT Compilation).

## 📂 Estrutura do Projeto

A estrutura do projeto foi desenhada para ser intuitiva e escalável.
folder/
├── .devcontainer/         # Configuração do VS Code para o ambiente Docker
├── data/                  # Pasta para armazenar os datasets (ex: cifar10/train, cifar10/test)
├── env_check/             # Scripts para diagnóstico e validação do ambiente
├── output/                # Pasta onde todos os resultados (modelos, relatórios) são salvos
├── src/                   # Todo o código-fonte modularizado do projeto
│   ├── augmentation.py
│   ├── callbacks.py
│   ├── config.py          # O "painel de controle" do projeto
│   ├── data_loader.py
│   ├── model_architectures.py
│   ├── reporting.py
│   └── visualization.py
├── old/                   # Notebooks e scripts antigos para referência
├── Dockerfile             # "Receita" para construir a imagem Docker do ambiente
├── docker-compose.yml     # Orquestra os serviços Docker
├── train.py               # Script principal para iniciar o treino de um modelo
├── evaluate.py            # Script para avaliar um modelo já treinado
└── requirements.txt       # Lista de dependências Python

## 🚀 Como Começar

### Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Visual Studio Code](https://code.visualstudio.com/)
- Extensão [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) para o VS Code
- (Para utilizadores de GPU) Drivers NVIDIA e [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Passos para a Configuração

1. **Clone o Repositório:**
   ```bash
   git clone [URL_DO_SEU_REPOSITORIO]
   cd [NOME_DO_REPOSITORIO]
   ```
2. **Abra no VS Code:**
   Abra a pasta do projeto no Visual Studio Code.
3. **Inicie o Ambiente no Container:**
   O VS Code irá detetar a pasta `.devcontainer` e sugerir "Reopen in Container". Clique nesse botão. O VS Code irá construir a imagem Docker e iniciar o ambiente. Este processo pode demorar alguns minutos na primeira vez.
4. **(Opcional) Verifique o Ambiente:**
   Abra um terminal no VS Code (`Ctrl+Shift+ñ` ou `Terminal > New Terminal`) e execute o script de diagnóstico para garantir que tudo está a funcionar, incluindo a GPU e as otimizações.
   ```bash
   python env_check/env_check.py
   ```

## workflows  Fluxo de Trabalho

### 1. Configurar uma Execução de Treino

- Abra o ficheiro `src/config.py`.
- Ajuste os parâmetros na secção de configurações principais. Os mais comuns são:
  - `Debug.SUBSET_SIZE_PER_CLASS`: Defina como um número (ex: `100`) para um teste rápido ou como `None` para o treino completo.
  - `Training.EPOCHS`: Defina o número de épocas para o treino.
  - `Model.IMG_SIZE`: Ajuste o tamanho das imagens, se necessário.

### 2. Executar o Treino

- No terminal do VS Code, execute o script de treino:
  ```bash
  python train.py
  ```
- Acompanhe o progresso no terminal. Ao final, uma nova pasta será criada em `output/` com o nome formatado `[NOME_PROJETO]___val_acc=[ACURACIA]___[TIMESTAMP]`.

### 3. Avaliar um Modelo Treinado

- Após o treino, execute o script de avaliação.
- **Para avaliar o último modelo treinado:**

  ```bash
  python evaluate.py
  ```
- **Para avaliar uma execução específica e mais antiga:**

  ```bash
  python evaluate.py --run_dir output/[NOME_DA_PASTA_DA_EXECUCAO]
  ```
- O script irá imprimir um relatório detalhado com métricas (Acurácia, Precisão, Recall, F1-Score) e exibir as matrizes de confusão para os conjuntos de treino, teste e combinado.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o ficheiro [LICENSE](LICENSE) para mais detalhes.
