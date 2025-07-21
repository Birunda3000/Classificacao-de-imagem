# Imagem base da NVIDIA para TensorFlow. Oferece otimizações para GPUs NVIDIA.
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Instala bibliotecas do sistema Linux necessárias para o OpenCV.
# O '--no-install-recommends' evita pacotes extras não essenciais.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0

# Define o diretório de trabalho padrão dentro do container para o seu projeto.
# Todos os comandos e o VS Code operarão a partir daqui.
WORKDIR /tf/TCC_GAN

# Copia o arquivo de dependências Python para o container.
COPY requirements.txt .

# Instala todas as bibliotecas Python listadas em requirements.txt.
# '--no-cache-dir' otimiza o espaço em disco.
RUN pip install --no-cache-dir -r requirements.txt

# Garante que o ambiente Python e as ferramentas auxiliares estão na PATH do usuário 'root'.
# Isso é crucial para o VS Code e suas extensões encontrarem tudo que precisam.
ENV PATH="/usr/local/bin:/usr/bin:${PATH}"

# Comando padrão que o container executará ao iniciar.
# Um loop infinito simples que mantém o container rodando para o VS Code poder anexar.
CMD ["/bin/bash", "-c", "while true; do sleep infinity; done"]