# Projeto_Simac
# 🤟 Reconhecimento de Letras em Libras com Python e MediaPipe

Este projeto utiliza visão computacional e aprendizado de máquina para **reconhecer letras do alfabeto em Libras (Língua Brasileira de Sinais)**, utilizando **MediaPipe** para rastreamento de mão e **Random Forest** para classificação dos sinais.

> Ideal para fins educacionais, demonstrações de IA e inclusão digital.

---

## 📸 Demonstração

| Coleta de dados | Reconhecimento |
|-----------------|----------------|
| ![coleta](https://imgur.com/Xexemplo1.gif) | ![reconhecimento](https://imgur.com/Xexemplo2.gif) |

---

## 📦 Tecnologias Utilizadas

- [Python 3.11+](https://www.python.org/)
- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Joblib](https://joblib.readthedocs.io/)

---

## 🚀 Como Executar o Projeto

### 1. Clone o repositório:
```bash
git clone https://github.com/lalaDevil/Projeto_Simac.git
cd Projeto_Simac
```
### 2. Instale as dependências:
- pip install opencv-python mediapipe scikit-learn pandas joblib
- pip install opencv-python
- pip install mediapipe opencv-python

🧪 Etapas do Projeto
1. Coletar dados
Use a webcam para capturar posições da mão e rotular com a tecla da letra correspondente (A–Z):
python app/coletar_letras.py

2. Treinar o modelo
Com os dados salvos, treine o modelo de reconhecimento com:
python app/treinar_modelo.py

3. Reconhecer letras em tempo real
Execute a detecção e predição com o modelo treinado:
python app/reconhecer_letras.py

📁 Estrutura do Projeto
libras_alfabeto/
├── dados/                     ← Dados coletados (.csv)
├── app/
│   ├── coletar_letras.py      ← Coleta das posições da mão
│   ├── treinar_modelo.py      ← Treinamento do modelo
│   └── reconhecer_letras.py   ← Execução do reconhecimento
├── modelo_letras_libras.pkl   ← Modelo treinado (gerado)
├── requirements.txt           ← Dependências
└── README.md

👨‍💻 Autor
Desenvolvido por Elaíne Gomes e Joyce Peres


