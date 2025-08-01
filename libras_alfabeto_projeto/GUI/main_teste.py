import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import threading
import random
import numpy as np
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model
from collections import deque
import subprocess
import sys
import os

class AplicativoLibras:
    def __init__(self, root):
        self.root = root
        self.root.title("Aprenda Libras - Sistema Completo")
        self.root.geometry("1200x800")
        
        # Configuração de caminhos - Adaptada para nova estrutura
        self.base_dir = Path(__file__).parent.parent.parent  # Vai para PROJETO_SIMAC
        self.dados_dir = self.base_dir / "dados"
        self.entrada_dir = self.base_dir / "entrada"
        self.modelos_dir = self.base_dir / "modelos"
        self.saida_dir = self.base_dir / "saída"
        
        # Caminhos dentro do pacote libras_alfabeto_projeto
        self.app_dir = self.base_dir / "libras_alfabeto_projeto" / "app"
        self.coleta_dir = self.app_dir / "coleta"
        self.reconhecer_dir = self.app_dir / "reconhecer"
        self.treinamento_dir = self.app_dir / "treinamento"
        self.gui_dir = self.base_dir / "libras_alfabeto_projeto" / "GUI"
        self.utilitarios_dir = self.base_dir / "libras_alfabeto_projeto" / "utilitarios"

        # Verificar e criar diretórios necessários
        self.modelos_dir.mkdir(exist_ok=True)
        self.dados_dir.mkdir(exist_ok=True)
        self.saida_dir.mkdir(exist_ok=True)

        # Configurações do MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Estado do aplicativo
        self.cap = None
        self.running = False
        self.nivel_atual = 1
        self.pontuacao = 0
        self.gesto_alvo = None
        self.buffer_gestos = deque(maxlen=30)
        self.historico_predicoes = deque(maxlen=15)
        
        # Carregar modelos
        self.modelo_gestos = None
        self.le_gestos = None
        self.modelo_letras = None
        self.carregar_modelos()

        # Gestos por nível (letras no nível 1, gestos nos demais)
        self.gestos_por_nivel = {
            1: ["A", "B", "C", "D", "E"],
            2: ["OLÁ", "OBRIGADO", "AJUDA"],
            3: ["ABAIXO", "ACIMA", "ADENTRO"],
            4: ["FAMÍLIA", "AMIGO", "ESCOLA"],
            5: ["EU TE AMO", "COMO VOCÊ ESTÁ"]
        }

        # Interface
        self.criar_menu()
        self.criar_barra_status()
        self.criar_area_principal()
        self.criar_painel_controle()

    def carregar_modelos(self):
        """Carrega os modelos com verificação detalhada"""
        try:
            print("\n=== VERIFICAÇÃO DE MODELOS ===")
            
            # Modelo de gestos
            modelo_gestos_path = self.modelos_dir / "modelo_gestos.h5"
            rotulador_path = self.modelos_dir / "rotulador_gestos.pkl"
            
            print(f"Procurando em: {self.modelos_dir}")
            
            if modelo_gestos_path.exists() and rotulador_path.exists():
                print("Carregando modelo de gestos...")
                self.modelo_gestos = load_model(modelo_gestos_path)
                self.le_gestos = joblib.load(rotulador_path)
                print(f"✅ Modelo de gestos carregado. Classes: {list(self.le_gestos.classes_)}")
            else:
                print("❌ Arquivos do modelo de gestos não encontrados:")
                if not modelo_gestos_path.exists():
                    print(f"- Faltando: {modelo_gestos_path}")
                if not rotulador_path.exists():
                    print(f"- Faltando: {rotulador_path}")
            
            # Modelo de letras - agora na raiz do projeto
            modelo_letras_path = self.base_dir / "modelo_letras_libras.pkl"
            if modelo_letras_path.exists():
                self.modelo_letras = joblib.load(modelo_letras_path)
                print("✅ Modelo de letras carregado")
            else:
                print(f"❌ Modelo de letras não encontrado: {modelo_letras_path}")
                
        except Exception as e:
            error_msg = f"ERRO ao carregar modelos:\n{str(e)}"
            print(error_msg)
            messagebox.showerror("Erro", error_msg)

    def criar_menu(self):
        menubar = tk.Menu(self.root)
        
        # Menu Níveis
        menu_niveis = tk.Menu(menubar, tearoff=0)
        for i in range(1, 6):
            menu_niveis.add_command(
                label=f"Nível {i}", 
                command=lambda n=i: self.iniciar_nivel(n)
            )
        menubar.add_cascade(label="Níveis", menu=menu_niveis)
        
        # Menu Coleta
        menu_coleta = tk.Menu(menubar, tearoff=0)
        menu_coleta.add_command(label="Coletar Gestos", command=self.executar_coletar_gestos)
        menu_coleta.add_command(label="Coletar Letras", command=self.executar_coletar_letras)
        menubar.add_cascade(label="Coleta", menu=menu_coleta)
        
        # Menu Treinamento
        menu_treinamento = tk.Menu(menubar, tearoff=0)
        menu_treinamento.add_command(label="Treinar Modelo Gestos", command=self.executar_treinar_gestos)
        menu_treinamento.add_command(label="Treinar Modelo Letras", command=self.executar_treinar_letras)
        menubar.add_cascade(label="Treinamento", menu=menu_treinamento)
        
        # Menu Utilitários
        menu_util = tk.Menu(menubar, tearoff=0)
        menu_util.add_command(label="Legendar Vídeo", command=self.executar_legendar_video)
        menubar.add_cascade(label="Utilitários", menu=menu_util)
        
        # Menu Ajuda
        menu_ajuda = tk.Menu(menubar, tearoff=0)
        menu_ajuda.add_command(label="Sobre", command=self.mostrar_sobre)
        menu_ajuda.add_command(label="Verificar Modelos", command=self.verificar_modelos)
        menubar.add_cascade(label="Ajuda", menu=menu_ajuda)
        
        self.root.config(menu=menubar)

    def verificar_modelos(self):
        """Janela de verificação de modelos"""
        info = "=== STATUS DOS MODELOS ===\n\n"
        
        # Verificar modelo de gestos
        modelo_gestos_path = self.modelos_dir / "modelo_gestos.h5"
        rotulador_path = self.modelos_dir / "rotulador_gestos.pkl"
        
        gestos_ok = modelo_gestos_path.exists() and rotulador_path.exists()
        info += f"Modelo de Gestos: {'✅ Carregado' if gestos_ok else '❌ Faltando arquivos'}\n"
        info += f"- {modelo_gestos_path.name}: {'Encontrado' if modelo_gestos_path.exists() else 'Não encontrado'}\n"
        info += f"- {rotulador_path.name}: {'Encontrado' if rotulador_path.exists() else 'Não encontrado'}\n\n"
        
        # Verificar modelo de letras
        modelo_letras_path = self.base_dir / "modelo_letras_libras.pkl"
        letras_ok = modelo_letras_path.exists()
        info += f"Modelo de Letras: {'✅ Carregado' if letras_ok else '❌ Não encontrado'}\n"
        info += f"- {modelo_letras_path.name}: {'Encontrado' if modelo_letras_path.exists() else 'Não encontrado'}\n\n"
        
        info += "Soluções:\n"
        info += "1. Execute 'Treinar Modelo Gestos' no menu Treinamento\n"
        info += "2. Execute 'Treinar Modelo Letras' no menu Treinamento\n"
        info += f"Modelos de gestos devem estar em: {self.modelos_dir}\n"
        info += f"Modelo de letras deve estar em: {self.base_dir}"
        
        messagebox.showinfo("Verificação de Modelos", info)

    def criar_barra_status(self):
        self.status_var = tk.StringVar()
        self.status_var.set(f"Pronto | Nível: {self.nivel_atual} | Pontos: {self.pontuacao}")
        
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def criar_area_principal(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Painel do gesto alvo
        left_frame = ttk.LabelFrame(main_frame, text="Gesto Alvo")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.gesto_alvo_label = ttk.Label(
            left_frame,
            text="Selecione um nível para começar",
            font=('Helvetica', 24, 'bold'),
            foreground='#2E86C1',
            anchor=tk.CENTER
        )
        self.gesto_alvo_label.pack(expand=True, fill=tk.BOTH)
        
        # Painel de feedback
        self.feedback_frame = ttk.LabelFrame(left_frame, text="Feedback")
        self.feedback_frame.pack(fill=tk.X, pady=10)
        
        self.feedback_label = ttk.Label(
            self.feedback_frame,
            text="",
            font=('Helvetica', 14),
            anchor=tk.CENTER
        )
        self.feedback_label.pack(pady=10)
        
        # Painel da câmera
        right_frame = ttk.LabelFrame(main_frame, text="Sua Câmera")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(right_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def criar_painel_controle(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_camera = ttk.Button(
            control_frame,
            text="Iniciar Câmera",
            command=self.toggle_camera
        )
        self.btn_camera.pack(side=tk.LEFT, padx=5)
        
        self.btn_verificar = ttk.Button(
            control_frame,
            text="Verificar Gesto",
            command=self.verificar_gesto,
            state=tk.DISABLED
        )
        self.btn_verificar.pack(side=tk.LEFT, padx=5)
        
        self.btn_proximo = ttk.Button(
            control_frame,
            text="Próximo Nível",
            command=self.proximo_nivel,
            state=tk.DISABLED
        )
        self.btn_proximo.pack(side=tk.LEFT, padx=5)

    def iniciar_nivel(self, nivel):
        """Inicia um novo nível de aprendizado"""
        self.nivel_atual = nivel
        self.gesto_alvo = random.choice(self.gestos_por_nivel[nivel])
        self.gesto_alvo_label.config(text=self.gesto_alvo)
        self.feedback_label.config(text="Mostre o gesto para a câmera", foreground='black')
        self.btn_verificar.config(state=tk.NORMAL)
        self.btn_proximo.config(state=tk.DISABLED)
        self.atualizar_status(f"Nível {nivel} | Gesto: {self.gesto_alvo}")

    def toggle_camera(self):
        """Liga/desliga a câmera"""
        if self.cap is None:
            self.iniciar_camera()
        else:
            self.parar_camera()

    def iniciar_camera(self):
        """Inicia a captura de vídeo com detecção de landmarks"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível acessar a câmera")
            return
        
        self.running = True
        self.btn_camera.config(text="Parar Câmera")
        
        self.video_thread = threading.Thread(target=self.atualizar_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def atualizar_video(self):
        """Atualiza o frame da câmera com landmarks visíveis"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Desenhar landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Processar landmarks para reconhecimento
                self.processar_landmarks(results)
            
            # Converter para Tkinter
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = img
            self.video_label.config(image=img)
            
            self.root.update()

    def processar_landmarks(self, results):
        """Processa os landmarks para reconhecimento"""
        landmarks = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        
        # Padroniza para 2 mãos (42 landmarks)
        landmarks = landmarks[:42]
        if len(landmarks) < 42:
            landmarks.extend([[0,0,0]] * (42 - len(landmarks)))
        
        self.buffer_gestos.append(np.array(landmarks).flatten())

    def verificar_gesto(self):
        """Verifica se o gesto/letra corresponde ao alvo"""
        # Determinar qual modelo usar baseado no nível
        if self.nivel_atual == 1:  # Nível de letras
            if not hasattr(self, 'modelo_letras') or self.modelo_letras is None:
             messagebox.showerror("Erro", 
                    "Modelo de letras não carregado!\n\n"
                    "Por favor:\n"
                    "1. Acesse o menu 'Treinamento > Treinar Modelo Letras'\n"
                    "2. Verifique se o arquivo está em:\n"
                    f"   {self.base_dir}/modelo_letras_libras.pkl")
            return
        
            modelo = self.modelo_letras
            buffer_size = 10  # Menos frames para reconhecimento de letras
        
            # Processamento específico para letras (apenas primeira mão)
            if len(self.buffer_gestos) < buffer_size:
                self.feedback_label.config(
                    text="Continue mostrando a letra... (coletando frames)",
                    foreground='orange'
                )
                return
            
            try:
                # Pegar apenas os primeiros 21 landmarks (primeira mão)
                landmarks_primeira_mao = np.array(self.buffer_gestos[-1])[:63]  # 21 landmarks * 3 coordenadas
                entrada = landmarks_primeira_mao.reshape(1, -1)
            
                pred = modelo.predict(entrada)[0]
                gesto_reconhecido = pred
                confianca = 1.0  # Modelo de letras pode não retornar confiança
            
                if gesto_reconhecido == self.gesto_alvo:
                    self.pontuacao += 10 * self.nivel_atual
                    self.feedback_label.config(
                        text=f"✅ Correto! {gesto_reconhecido}",
                        foreground='green'
                    )
                    self.btn_proximo.config(state=tk.NORMAL)
                else:
                    self.feedback_label.config(
                        text=f"❌ Tente novamente. Reconhecido: {gesto_reconhecido}",
                        foreground='red'
                    )
                
                self.atualizar_status()
            
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao reconhecer letra: {str(e)}")
            
            return  # Sai do método após processar letras
    
        # Processamento para gestos (níveis 2-5) - mantém o original
        if not hasattr(self, 'modelo_gestos') or self.modelo_gestos is None:
            messagebox.showerror("Erro", 
                "Modelo de gestos não carregado!\n\n"
                "Por favor:\n"
                "1. Acesse o menu 'Treinamento > Treinar Modelo Gestos'\n"
                "2. Verifique se os arquivos estão em:\n"
                f"   {self.modelos_dir}/modelo_gestos.h5\n"
                f"   {self.modelos_dir}/rotulador_gestos.pkl")
            return
    
        if len(self.buffer_gestos) < 30:
            self.feedback_label.config(
                text="Continue mostrando o gesto... (coletando frames)",
                foreground='orange'
            )
            return
    
        try:
            entrada = np.array(self.buffer_gestos).reshape(1, 30, 126)
            preds = self.modelo_gestos.predict(entrada, verbose=0)[0]
            classe_idx = np.argmax(preds)
            confianca = preds[classe_idx]
            gesto_reconhecido = self.le_gestos.classes_[classe_idx]
        
            # Suavização com histórico
            self.historico_predicoes.append(gesto_reconhecido)
            contagem = {}
            for g in self.historico_predicoes:
                contagem[g] = contagem.get(g, 0) + 1
            gesto_final = max(contagem.items(), key=lambda x: x[1])[0]
        
            if confianca > 0.7:
                if gesto_final == self.gesto_alvo:
                    self.pontuacao += 10 * self.nivel_atual
                    self.feedback_label.config(
                        text=f"✅ Correto! {gesto_final} ({confianca:.0%} confiança)",
                        foreground='green'
                    )
                    self.btn_proximo.config(state=tk.NORMAL)
                else:
                    self.feedback_label.config(
                        text=f"❌ Tente novamente. Reconhecido: {gesto_final}",
                        foreground='red'
                    )
            else:
                self.feedback_label.config(
                    text="Gesto não reconhecido claramente. Tente novamente",
                    foreground='orange'
                )
        
            self.atualizar_status()
        
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao reconhecer gesto: {str(e)}")

    def proximo_nivel(self):
        """Avança para o próximo nível"""
        if self.nivel_atual < 5:
            self.nivel_atual += 1
            self.buffer_gestos.clear()
            self.historico_predicoes.clear()
            self.iniciar_nivel(self.nivel_atual)
        else:
            messagebox.showinfo(
                "Parabéns!",
                f"Você completou todos os níveis!\nPontuação final: {self.pontuacao}"
            )
            self.btn_proximo.config(state=tk.DISABLED)

    def executar_coletar_gestos(self):
        """Executa o coletor de gestos"""
        self.parar_camera()
        try:
            script_path = self.coleta_dir / "coletar_gestos.py"
            subprocess.Popen([sys.executable, str(script_path)])
            messagebox.showinfo("Aviso", "Coletor de gestos iniciado em outra janela")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao executar coletar_gestos.py: {str(e)}")

    def executar_coletar_letras(self):
        """Executa o coletor de letras"""
        self.parar_camera()
        try:
            script_path = self.coleta_dir / "coletar_letras.py"
            subprocess.Popen([sys.executable, str(script_path)])
            messagebox.showinfo("Aviso", "Coletor de letras iniciado em outra janela")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao executar coletar_letras.py: {str(e)}")

    def executar_treinar_gestos(self):
        """Executa o treinamento de gestos"""
        self.parar_camera()
        try:
            script_path = self.treinamento_dir / "treinar_modelo_gestos.py"
            subprocess.Popen([sys.executable, str(script_path)])
            messagebox.showinfo("Aviso", 
                "Treinamento de gestos iniciado em outra janela\n\n"
                "Os modelos serão salvos em:\n"
                f"{self.modelos_dir}/modelo_gestos.h5\n"
                f"{self.modelos_dir}/rotulador_gestos.pkl")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao executar treinar_modelo_gestos.py: {str(e)}")

    def executar_treinar_letras(self):
        """Executa o treinamento de letras"""
        self.parar_camera()
        try:
            script_path = self.treinamento_dir / "treinar_modelo.py"
            subprocess.Popen([sys.executable, str(script_path)])
            messagebox.showinfo("Aviso", 
                "Treinamento de letras iniciado em outra janela\n\n"
                "O modelo será salvo em:\n"
                f"{self.base_dir}/modelo_letras_libras.pkl")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao executar treinar_modelo.py: {str(e)}")

    def executar_legendar_video(self):
        """Executa o gerador de legendas"""
        self.parar_camera()
        try:
            script_path = self.utilitarios_dir / "gerar_legenda_video.py"
            video_input = self.entrada_dir / "video_para_legendar.mp4"  # Exemplo
            subprocess.Popen([sys.executable, str(script_path), str(video_input)])
            messagebox.showinfo("Aviso", 
                "Gerador de legendas iniciado em outra janela\n\n"
                "O vídeo legendado será salvo em:\n"
                f"{self.saida_dir}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao executar gerar_legenda_video.py: {str(e)}")

    def parar_camera(self):
        """Para a captura de vídeo"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_camera.config(text="Iniciar Câmera")
        self.btn_verificar.config(state=tk.DISABLED)
        self.video_label.config(image='')

    def atualizar_status(self, mensagem=None):
        """Atualiza a barra de status"""
        if mensagem:
            self.status_var.set(mensagem)
        else:
            self.status_var.set(f"Nível {self.nivel_atual} | Pontos: {self.pontuacao} | Gesto: {self.gesto_alvo}")

    def mostrar_sobre(self):
        """Mostra informações sobre o aplicativo"""
        sobre = f"""
        Aprenda Libras - Sistema Educacional
        
        Versão 2.0
        Sistema completo para aprendizado de Libras
        
        Estrutura do Projeto:
        PROJETO_SIMAC/
        ├── dados/            (arquivos CSV)
        ├── entrada/          (video para legendar)
        ├── libras_alfabeto_projeto/          
        ├   ├── app/
        ├   │       ├── coleta/       (coletar_gestos.py, coletar_letras.py)
        ├   │       ├── reconhecer/   (reconhecer_gestos.py, reconhecer_letras.py)
        ├   │       └── treinamento/  (treinar_modelo_gestos.py, treinar_modelo.py)  
        ├   ├── GUI/ (main_teste.py)
        ├   └── utilitarios/      (gerar_legenda_video.py)
        ├── modelos/          (modelo_gestos.h5, rotulador_gestos.pkl)
        ├── saída/            (video e txt depois de legendado)
        └── (modelo_letras_libras.pkl)
        
        Diretório atual: {self.base_dir}
        
        © 2023 Projeto de Acessibilidade
        """
        messagebox.showinfo("Sobre", sobre)

    def sair(self):
        """Encerra o aplicativo corretamente"""
        self.parar_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicativoLibras(root)
    root.protocol("WM_DELETE_WINDOW", app.sair)
    root.mainloop()