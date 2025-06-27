# Projeto_Simac
# 🤟 Reconhecimento de Letras em Libras com Python

Este projeto utiliza visão computacional e aprendizado de máquina para **reconhecer letras do alfabeto em Libras (Língua Brasileira de Sinais)**, utilizando **MediaPipe** para rastreamento de mão e **Random Forest** para classificação dos sinais.

> Ideal para fins educacionais, demonstrações de IA e inclusão digital.

---

## 📸 Demonstração

O sistema é dividido em três etapas principais:

1. **Coleta de Dados** (`coletar_letras.py`)  
   Captura os gestos da mão e associa à letra pressionada no teclado, gerando um dataset.

2. **Treinamento do Modelo** (`treinar_modelo.py`)  
   Treina um classificador Random Forest com os dados coletados e salva o modelo.

3. **Reconhecimento em Tempo Real** (`reconhecer_letras.py`)  
   Usa o modelo treinado para prever, em tempo real, qual letra está sendo mostrada com a mão.

---

## 📂 Estrutura dos Arquivos

```
📁 dados/
  └── letras_libras.csv         # Arquivo CSV com os dados coletados
coletar_letras.py              # Script para coletar os dados
treinar_modelo.py              # Script para treinar o modelo
reconhecer_letras.py           # Script para reconhecer as letras em tempo real
modelo_letras_libras.pkl       # (Gerado após o treinamento)
```
## 🚀 Como Executar

### Pré-requisitos

Instale os pacotes necessários:

```bash
pip install opencv-python mediapipe scikit-learn pandas joblib
```

### 1. Coletar dados com webcam

```bash
python coletar_letras.py
```

- Mostre uma letra em Libras com a mão para a câmera.
- Pressione a tecla correspondente no teclado (ex: `A`, `B`, `C`...).
- Repita para várias letras e exemplos.
- Pressione `ESC` para sair.

### 2. Treinar o modelo

```bash
python treinar_modelo.py
```

- O modelo será treinado e salvo como `modelo_letras_libras.pkl`.

### 3. Reconhecer letras em tempo real

```bash
python reconhecer_letras.py
```

- A webcam será ativada e o sistema exibirá a letra reconhecida na tela.
- Pressione `ESC` para sair.

## 🎯 Resultados Esperados

- A acurácia será exibida após o treinamento.
- Durante o reconhecimento, a letra detectada será mostrada em tempo real sobre o vídeo.

## 🛠️ Tecnologias Utilizadas

- [OpenCV](https://opencv.org/)
- [MediaPipe Hands](https://google.github.io/mediapipe/)
- [Scikit-Learn](https://scikit-learn.org/)
- [Python 3](https://www.python.org/)

## 📌 Observações

- O modelo depende da qualidade dos dados coletados.
- É recomendado coletar múltiplas amostras de cada letra e variar ângulos e posições.
- Atualmente, o sistema reconhece **apenas uma mão por vez**.

👨‍💻 Autores
Este projeto foi desenvolvido por:
- Elaíne Gomes
- Joyce Peres

(🧠 Este projeto contou com o apoio de inteligência artificial generativa para otimizar a escrita de código e estruturação do projeto)

