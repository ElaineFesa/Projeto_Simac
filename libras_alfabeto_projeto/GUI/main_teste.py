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
from collections import deque, defaultdict
import os
import sys
import subprocess

class AplicativoLibras:
    def __init__(self, root):
        # Configuração do tema
        self.COR_PRIMARIA = "#6A0DAD"  # Roxo principal
        self.COR_SECUNDARIA = "#FFD700"  # Amarelo ouro
        self.COR_TERCIARIA = "#4B0082"  # Roxo escuro
        self.COR_TEXTO_CLARO = "#FFFFFF"  # Branco
        self.COR_TEXTO_ESCURO = "#333333"  # Cinza escuro
        self.COR_FUNDO = "#F5F5F5"  # Cinza claro
        self.COR_SUCESSO = "#2E8B57"  # Verde
        self.COR_ERRO = "#DC143C"  # Vermelho

        self.root = root
        self.root.title("Aprenda Libras - Sistema Completo")
        self.root.geometry("1200x800")
        self.root.configure(bg=self.COR_FUNDO)
        
        # Configuração de estilos
        self.configurar_estilos()
        
        # Configuração de caminhos
        self.base_dir = Path(__file__).parent.parent.parent
        self.dados_dir = self.base_dir / "dados"
        self.modelos_dir = self.base_dir / "modelos"
        self.coleta_dir = self.base_dir / "libras_alfabeto_projeto" / "app" / "coleta"
        self.reconhecer_dir = self.base_dir / "libras_alfabeto_projeto" / "app" / "reconhecer"
        self.treinamento_dir = self.base_dir / "libras_alfabeto_projeto" / "app" / "treinamento"
        self.utilitarios_dir = self.base_dir / "libras_alfabeto_projeto" / "utilitarios"
        self.saida_dir = self.base_dir / "saída"

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
        self.frames_sem_maos = 0
        self.RESET_THRESHOLD = 10
        self.ultimo_gesto_reconhecido = None
        
        # Carregar modelos
        self.modelo_gestos = None
        self.le_gestos = None
        self.modelo_letras = None
        self.carregar_modelos()

        # Gestos por nível
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

    def configurar_estilos(self):
        """Configura todos os estilos visuais"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        style.configure('TFrame', background=self.COR_FUNDO)
        
        # Labels
        style.configure('TLabel', 
                      background=self.COR_FUNDO,
                      foreground=self.COR_TEXTO_ESCURO,
                      font=('Helvetica', 10))
        
        # Botões
        style.configure('TButton',
                      background=self.COR_PRIMARIA,
                      foreground=self.COR_TEXTO_CLARO,
                      font=('Helvetica', 10, 'bold'),
                      padding=8,
                      bordercolor=self.COR_PRIMARIA)
        style.map('TButton',
                background=[('active', self.COR_TERCIARIA),
                          ('disabled', '#CCCCCC')],
                foreground=[('active', self.COR_SECUNDARIA),
                          ('disabled', '#666666')])
        
        # LabelFrame
        style.configure('TLabelframe',
                      background=self.COR_FUNDO,
                      bordercolor=self.COR_PRIMARIA)
        style.configure('TLabelframe.Label',
                      background=self.COR_FUNDO,
                      foreground=self.COR_PRIMARIA,
                      font=('Helvetica', 10, 'bold'))
        
        # Barra de status
        style.configure('Status.TLabel',
                      background=self.COR_PRIMARIA,
                      foreground=self.COR_TEXTO_CLARO,
                      font=('Helvetica', 10),
                      padding=5)

    def carregar_modelos(self):
        """Carrega os modelos com verificação detalhada"""
        try:
            # Modelo de gestos
            modelo_gestos_path = self.modelos_dir / "modelo_gestos.h5"
            rotulador_path = self.modelos_dir / "rotulador_gestos.pkl"
            
            if modelo_gestos_path.exists() and rotulador_path.exists():
                self.modelo_gestos = load_model(modelo_gestos_path)
                self.le_gestos = joblib.load(rotulador_path)
            
            # Modelo de letras
            modelo_letras_path = self.base_dir / "modelo_letras_libras.pkl"
            if modelo_letras_path.exists():
                self.modelo_letras = joblib.load(modelo_letras_path)
                
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar modelos: {str(e)}")

    def criar_menu(self):
        menubar = tk.Menu(self.root,
                         bg=self.COR_FUNDO,
                         fg=self.COR_TEXTO_ESCURO,
                         activebackground=self.COR_PRIMARIA,
                         activeforeground=self.COR_SECUNDARIA,
                         tearoff=0)
        
        # Menu Níveis
        menu_niveis = tk.Menu(menubar, tearoff=0,
                            bg=self.COR_FUNDO,
                            fg=self.COR_TEXTO_ESCURO,
                            activebackground=self.COR_PRIMARIA,
                            activeforeground=self.COR_SECUNDARIA)
        
        for i in range(1, 6):
            menu_niveis.add_command(
                label=f"Nível {i}", 
                command=lambda n=i: self.iniciar_nivel(n)
            )
        menubar.add_cascade(label="Níveis", menu=menu_niveis)
        
        # Menu Coleta
        menu_coleta = tk.Menu(menubar, tearoff=0,
                            bg=self.COR_FUNDO,
                            fg=self.COR_TEXTO_ESCURO,
                            activebackground=self.COR_PRIMARIA,
                            activeforeground=self.COR_SECUNDARIA)
        
        menu_coleta.add_command(label="Coletar Gestos", command=self.executar_coletar_gestos)
        menu_coleta.add_command(label="Coletar Letras", command=self.executar_coletar_letras)
        menubar.add_cascade(label="Coleta", menu=menu_coleta)
        
        # Menu Treinamento
        menu_treinamento = tk.Menu(menubar, tearoff=0,
                                 bg=self.COR_FUNDO,
                                 fg=self.COR_TEXTO_ESCURO,
                                 activebackground=self.COR_PRIMARIA,
                                 activeforeground=self.COR_SECUNDARIA)
        
        menu_treinamento.add_command(label="Treinar Modelo Gestos", command=self.executar_treinar_gestos)
        menu_treinamento.add_command(label="Treinar Modelo Letras", command=self.executar_treinar_letras)
        menubar.add_cascade(label="Treinamento", menu=menu_treinamento)
        
        # Menu Utilitários
        menu_util = tk.Menu(menubar, tearoff=0,
                          bg=self.COR_FUNDO,
                          fg=self.COR_TEXTO_ESCURO,
                          activebackground=self.COR_PRIMARIA,
                          activeforeground=self.COR_SECUNDARIA)
        
        menu_util.add_command(label="Legendar Vídeo", command=self.executar_legendar_video)
        menubar.add_cascade(label="Utilitários", menu=menu_util)
        
        # Menu Ajuda
        menu_ajuda = tk.Menu(menubar, tearoff=0,
                           bg=self.COR_FUNDO,
                           fg=self.COR_TEXTO_ESCURO,
                           activebackground=self.COR_PRIMARIA,
                           activeforeground=self.COR_SECUNDARIA)
        
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
            anchor=tk.W,
            style='Status.TLabel'
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def criar_area_principal(self):
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Painel do gesto alvo
        left_frame = ttk.LabelFrame(
            main_frame,
            text="Gesto Alvo",
            style='TLabelframe'
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.gesto_alvo_label = ttk.Label(
            left_frame,
            text="Selecione um nível para começar",
            font=('Helvetica', 24, 'bold'),
            foreground=self.COR_PRIMARIA,
            anchor=tk.CENTER,
            style='TLabel'
        )
        self.gesto_alvo_label.pack(expand=True, fill=tk.BOTH)
        
        # Painel de feedback
        self.feedback_frame = ttk.LabelFrame(
            left_frame,
            text="Feedback",
            style='TLabelframe'
        )
        self.feedback_frame.pack(fill=tk.X, pady=10)
        
        self.feedback_label = ttk.Label(
            self.feedback_frame,
            text="",
            font=('Helvetica', 14),
            anchor=tk.CENTER,
            style='TLabel'
        )
        self.feedback_label.pack(pady=10)
        
        # Painel da câmera
        right_frame = ttk.LabelFrame(
            main_frame,
            text="Sua Câmera",
            style='TLabelframe'
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(right_frame, style='TLabel')
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def criar_painel_controle(self):
        control_frame = ttk.Frame(self.root, style='TFrame')
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_camera = ttk.Button(
            control_frame,
            text="Iniciar Câmera",
            command=self.toggle_camera,
            style='TButton'
        )
        self.btn_camera.pack(side=tk.LEFT, padx=5)
        
        self.btn_proximo = ttk.Button(
            control_frame,
            text="Próximo Nível",
            command=self.proximo_nivel,
            state=tk.DISABLED,
            style='TButton'
        )
        self.btn_proximo.pack(side=tk.LEFT, padx=5)

    def iniciar_nivel(self, nivel):
        """Inicia um novo nível de aprendizado"""
        self.nivel_atual = nivel
        self.gesto_alvo = random.choice(self.gestos_por_nivel[nivel])
        self.gesto_alvo_label.config(text=self.gesto_alvo)
        self.feedback_label.config(text="Mostre o gesto para a câmera", foreground=self.COR_TEXTO_ESCURO)
        self.btn_proximo.config(state=tk.DISABLED)
        self.atualizar_status(f"Nível {nivel} | Gesto: {self.gesto_alvo}")
        self.buffer_gestos.clear()
        self.historico_predicoes.clear()
        self.ultimo_gesto_reconhecido = None

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
        
        return np.array(landmarks).flatten()

    def reconhecer_gesto_automatico(self):
        """Reconhece o gesto automaticamente quando o buffer estiver cheio"""
        if len(self.buffer_gestos) == 30:  # Buffer cheio
            if self.nivel_atual == 1:  # Letras
                self.reconhecer_letra()
            else:  # Gestos
                self.reconhecer_gesto()

    def reconhecer_letra(self):
        """Reconhecimento automático para letras (nível 1)"""
        try:
            # Pegar apenas os primeiros 63 valores (primeira mão)
            landmarks_primeira_mao = np.array(self.buffer_gestos[-1])[:63]
            entrada = landmarks_primeira_mao.reshape(1, -1)
            
            pred = self.modelo_letras.predict(entrada)[0]
            gesto_reconhecido = pred
            
            if gesto_reconhecido == self.gesto_alvo:
                self.pontuacao += 10 * self.nivel_atual
                self.feedback_label.config(
                    text=f"✅ Correto! {gesto_reconhecido}",
                    foreground=self.COR_SUCESSO
                )
                self.btn_proximo.config(state=tk.NORMAL)
                self.buffer_gestos.clear()  # Resetar buffer após reconhecimento
            elif gesto_reconhecido != self.ultimo_gesto_reconhecido:
                self.feedback_label.config(
                    text=f"Reconhecido: {gesto_reconhecido} (Mostre: {self.gesto_alvo})",
                    foreground=self.COR_SECUNDARIA
                )
            
            self.ultimo_gesto_reconhecido = gesto_reconhecido
            self.atualizar_status()
            
        except Exception as e:
            print(f"Erro ao reconhecer letra: {str(e)}")

    def reconhecer_gesto(self):
        """Reconhecimento automático para gestos (níveis 2-5)"""
        try:
            entrada = np.array(self.buffer_gestos).reshape(1, 30, 126)
            preds = self.modelo_gestos.predict(entrada, verbose=0)[0]
            classe_idx = np.argmax(preds)
            confianca = preds[classe_idx]
            gesto_reconhecido = self.le_gestos.classes_[classe_idx]
            
            # Suavização com histórico
            self.historico_predicoes.append(gesto_reconhecido)
            contagem = defaultdict(int)
            for g in self.historico_predicoes:
                contagem[g] += 1
            gesto_final = max(contagem.items(), key=lambda x: x[1])[0]
            
            if confianca > 0.7:
                if gesto_final == self.gesto_alvo:
                    self.pontuacao += 10 * self.nivel_atual
                    self.feedback_label.config(
                        text=f"✅ Correto! {gesto_final} ({confianca:.0%} confiança)",
                        foreground=self.COR_SUCESSO
                    )
                    self.btn_proximo.config(state=tk.NORMAL)
                    self.buffer_gestos.clear()  # Resetar buffer após reconhecimento
                elif gesto_final != self.ultimo_gesto_reconhecido:
                    self.feedback_label.config(
                        text=f"Reconhecido: {gesto_final} (Mostre: {self.gesto_alvo})",
                        foreground=self.COR_SECUNDARIA
                    )
                
                self.ultimo_gesto_reconhecido = gesto_final
                self.atualizar_status()
                
        except Exception as e:
            print(f"Erro ao reconhecer gesto: {str(e)}")

    def atualizar_video(self):
        """Atualiza o frame da câmera com reconhecimento automático"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Reset se não detectar mãos por muitos frames
            if not results.multi_hand_landmarks:
                self.frames_sem_maos += 1
                if self.frames_sem_maos > self.RESET_THRESHOLD and self.buffer_gestos:
                    self.buffer_gestos.clear()
                    self.feedback_label.config(
                        text="Mãos não detectadas. Mostre o gesto novamente",
                        foreground=self.COR_SECUNDARIA
                    )
            else:
                self.frames_sem_maos = 0
                landmarks = self.processar_landmarks(results)
                self.buffer_gestos.append(landmarks)
                
                # Desenhar landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Reconhecimento automático
            if self.gesto_alvo:  # Só reconhece se um nível estiver ativo
                self.reconhecer_gesto_automatico()
            
            # Exibir buffer status
            cv2.putText(frame, f"Buffer: {len(self.buffer_gestos)}/30", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Converter para Tkinter
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = img
            self.video_label.config(image=img)
            
            self.root.update()

    def parar_camera(self):
        """Para a captura de vídeo"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_camera.config(text="Iniciar Câmera")
        self.video_label.config(image='')

    def atualizar_status(self, mensagem=None):
        """Atualiza a barra de status"""
        if mensagem:
            self.status_var.set(mensagem)
        else:
            self.status_var.set(f"Nível {self.nivel_atual} | Pontos: {self.pontuacao} | Gesto: {self.gesto_alvo}")

    def proximo_nivel(self):
        """Avança para o próximo nível"""
        if self.nivel_atual < 5:
            self.nivel_atual += 1
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
            video_input = self.base_dir / "entrada" / "video_para_legendar.mp4"  # Exemplo
            subprocess.Popen([sys.executable, str(script_path), str(video_input)])
            messagebox.showinfo("Aviso", 
                "Gerador de legendas iniciado em outra janela\n\n"
                "O vídeo legendado será salvo em:\n"
                f"{self.saida_dir}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao executar gerar_legenda_video.py: {str(e)}")

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
    
    # Configuração padrão para messagebox
    root.option_add('*Dialog.msg.font', 'Helvetica 10')
    root.option_add('*Dialog.msg.background', '#F5F5F5')
    root.option_add('*Dialog.msg.foreground', '#333333')
    root.option_add('*Dialog.msg.buttonBackground', '#6A0DAD')
    root.option_add('*Dialog.msg.buttonForeground', '#FFFFFF')
    
    app = AplicativoLibras(root)
    root.protocol("WM_DELETE_WINDOW", app.sair)
    root.mainloop()