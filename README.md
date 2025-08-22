📘 LIA - Libras com Inteligência Artificial

O LIA é um aplicativo interativo que ensina Língua Brasileira de Sinais (Libras) utilizando reconhecimento de gestos em tempo real.
Ele combina MediaPipe, OpenCV, TensorFlow e Tkinter para capturar gestos com a câmera, reconhecer sinais e gamificar o aprendizado em níveis e seções temáticas.

🚀 Funcionalidades

🎮 Aplicativo interativo em interface gráfica (Tkinter).

✋ Reconhecimento de gestos usando MediaPipe + LSTM.

📷 Captura de gestos com a câmera.

📝 Coleta de novos gestos e armazenamento em CSV.

🤖 Treinamento de modelos de reconhecimento personalizados.

📊 Feedback em tempo real com confiança e suavização de predições.


⚙️ Instalação

Clone este repositório:

git clone https://github.com/ElaineFesa/Projeto_Simac.git

Instale as dependências:

Python 3.11

TensorFlow 2.19.0

MediaPipe 0.10.21

OpenCV

Tkinter (incluso no Python)

NumPy, Pandas, Scikit-learn, Joblib

▶️ Como Usar
1. Coletar novos gestos

Execute:

python coletar_gestos.py


Digite o nome do gesto.

Mostre o gesto para a câmera.

Pressione Espaço para gravar (mínimo 10 frames).

Pressione ESC para cancelar.

Os dados serão salvos em dados/gestos_libras.csv.

2. Treinar o modelo
python treinar_modelo_gestos.py


Treina um modelo LSTM baseado nos gestos coletados.

Gera os arquivos:

modelos/modelo_gestos.h5 (rede neural treinada).

modelos/rotulador_gestos.pkl (rótulos dos gestos).

3. Testar reconhecimento
python reconhecer_gestos.py


Inicia a captura da câmera.

Exibe os gestos reconhecidos em tempo real.

4. Rodar o aplicativo
python main.py


Interface gráfica abre em tela cheia.

Escolha a seção.

Complete os níveis mostrando os gestos corretos para a câmera.

Avance desbloqueando novas seções.

👩‍💻 Autoria

Desenvolvido por Elaíne Gomes e Joyce da Costa, 2025.
Projeto acadêmico para o SIMAC.
