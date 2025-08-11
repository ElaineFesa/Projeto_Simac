import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import random
import numpy as np
from collections import deque, defaultdict
import time
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path
import threading
import logging

logging.basicConfig(level=logging.INFO)

# --- Configurações de reconhecimento ---
SEQUENCE_LENGTH = 30
MIN_CONFIDENCE = 0.70      # Confiança mínima para considerar uma predição
CONFIDENCE_MARGIN = 0.15   # Margem entre top1 e top2 para evitar ambiguidade
VOTE_WINDOW = 7            # Quantas predições recentes usar na votação
REQUIRED_VOTE_RATIO = 0.6  # Percentual necessário na votação para aceitar (ex: 0.6 = 60%)
SLIDE_AFTER_DETECT = 8     # Quantos frames descartar (deslizar) após detectar para evitar repetições
MODEL_DIR = Path("modelos")

class AplicativoLibras:
    def __init__(self, root):
        # Configuração de cores e estilo
        self.configurar_cores()

        # Janela principal
        self.root = root
        self.root.title("LIA")
        self.root.state('zoomed')
        self.root.configure(bg=self.COR_FUNDO)

        # MediaPipe (criado apenas uma vez)
        self.configurar_mediapipe()

        # Estado do app
        self.inicializar_estado()

        # Pré-carrega o modelo em background (não bloquear UI)
        threading.Thread(target=self._carregar_modelo_warmup, daemon=True).start()

        # Estrutura de seções (mantive a sua estrutura original)
        self.secoes = {
            "Alfabeto": {
                1: ["A", "E", "I", "O", "U"],
                2: ["B", "C", "D", "G", "I", "L"],
                3: ["M", "N", "O", "P", "Q", "R"],
                4: ["S", "T", "U", "V", "W"],
                5: ["H", "J", "K", "X"],
                6: ["W", "Y", "Z"]
            }
        }

        self.icones_secoes = {"Alfabeto": "🔤"}

        for secao in self.secoes:
            self.niveis_completos[secao] = []

        self.mostrar_tela_inicial()

    # ---------------- UI helpers (sem alteração significativa) ----------------
    def configurar_cores(self):
        self.COR_PRIMARIA = "#6A0DAD"
        self.COR_SECUNDARIA = "#FFD700"
        self.COR_FUNDO = "#F5F5F5"
        self.COR_TEXTO_CLARO = "#FFFFFF"
        self.COR_TEXTO_ESCURO = "#333333"
        self.COR_SUCESSO = "#2E8B57"
        self.COR_ERRO = "#DC143C"
        self.COR_BLOQUEADO = "#CCCCCC"
        self.COR_CARD = "#FFFFFF"
        self.COR_BORDA = "#E0E0E0"

    def configurar_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        if not hasattr(self, 'hands') or self.hands is None:
            # Criar uma única instância do Hands para reduzir overhead
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def inicializar_estado(self):
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.skip_frames = 1
        self.nivel_atual = 1
        self.pontuacao = 0
        self.gesto_alvo = None
        self.buffer_gestos = deque(maxlen=SEQUENCE_LENGTH)
        self.historico_predicoes = deque(maxlen=VOTE_WINDOW)
        self.frames_sem_maos = 0
        self.RESET_THRESHOLD = 10
        self.niveis_completos = {}
        self.secoes_liberadas = ["Alfabeto"]
        self.tempo_inicio = 0
        self.tempo_gasto = 0
        self.ultimo_gesto_reconhecido = None
        self.modelo_gestos = None
        self.le_gestos = None
        self.camera_timeout = 5
        self.predicting = False  # flag para evitar várias predições simultâneas

        # Carregar modelo (pode demorar - fizemos warmup em thread)
        try:
            modelo, le = self.carregar_modelo_gestos()
            if modelo is not None:
                self.modelo_gestos = modelo
                self.le_gestos = le
        except Exception as e:
            logging.warning(f"Modelo não carregado na inicialização: {e}")

    def carregar_modelo_gestos(self):
        try:
            modelo_path = MODEL_DIR / "modelo_gestos.h5"
            rotulador_path = MODEL_DIR / "rotulador_gestos.pkl"
            if not modelo_path.exists() or not rotulador_path.exists():
                logging.warning("Modelo ou rotulador não encontrados em 'modelos/'.")
                return None, None
            modelo = load_model(modelo_path)
            le = joblib.load(rotulador_path)
            logging.info(f"Modelo carregado. Classes: {list(le.classes_)}")
            return modelo, le
        except Exception as e:
            logging.exception("Falha ao carregar modelo:")
            return None, None

    def _carregar_modelo_warmup(self):
        """Carrega modelo e faz um predict de warm-up para reduzir latência na 1ª predição"""
        modelo, le = self.carregar_modelo_gestos()
        if modelo is not None:
            self.modelo_gestos = modelo
            self.le_gestos = le
            # Warm-up com zeros (rodar em background)
            try:
                modelo.predict(np.zeros((1, SEQUENCE_LENGTH, 126)), verbose=0)
                logging.info("Modelo aquecido (warm-up) em background.")
            except Exception as e:
                logging.warning(f"Warm-up falhou: {e}")

    # ----- telas e navegação (mantive sua lógica) -----
    def mostrar_tela_inicial(self):
        self.limpar_tela()
        splash_frame = tk.Frame(self.root, bg=self.COR_PRIMARIA)
        splash_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(splash_frame, text="👋", font=("Helvetica", 100),
                 bg=self.COR_PRIMARIA, fg=self.COR_SECUNDARIA).pack(pady=20)
        tk.Label(splash_frame, text="LIA", font=("Helvetica", 100, "bold"),
                 bg=self.COR_PRIMARIA, fg=self.COR_TEXTO_CLARO).pack(pady=10)
        self.root.after(900, self.mostrar_tela_secoes)

    def mostrar_tela_secoes(self):
        self.limpar_tela()
        self.configurar_estilo_progressbar()
        main_frame = tk.Frame(self.root, bg=self.COR_FUNDO)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.criar_cabecalho_secoes(main_frame)
        container = tk.Frame(main_frame, bg=self.COR_FUNDO)
        container.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(container, bg=self.COR_FUNDO, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COR_FUNDO)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        colunas = max(3, min(4, self.root.winfo_screenwidth() // 300))
        for i, secao in enumerate(self.secoes):
            card = self.criar_card(scrollable_frame, secao)
            card.grid(row=i // colunas, column=i % colunas, padx=10, pady=10, sticky="nsew")
        for col in range(colunas):
            scrollable_frame.columnconfigure(col, weight=1, uniform="group1")
        for row in range((len(self.secoes) + colunas - 1) // colunas):
            scrollable_frame.rowconfigure(row, weight=1)
        self.criar_rodape_secoes(main_frame)

    def configurar_estilo_progressbar(self):
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar", troughcolor=self.COR_FUNDO, background=self.COR_PRIMARIA)

    def criar_cabecalho_secoes(self, parent):
        header_frame = tk.Frame(parent, bg=self.COR_FUNDO)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        tk.Label(header_frame, text="Seções", font=("Helvetica", 26, "bold"),
                 bg=self.COR_FUNDO, fg=self.COR_PRIMARIA).pack(side=tk.LEFT)
        info_frame = tk.Frame(header_frame, bg=self.COR_FUNDO)
        info_frame.pack(side=tk.RIGHT, padx=10)
        tk.Label(info_frame, text=f"🏆 Pontuação: {self.pontuacao}", font=("Helvetica", 18, "bold"),
                 bg=self.COR_FUNDO, fg=self.COR_PRIMARIA).pack(side=tk.LEFT, padx=10)
        total_secoes = len(self.secoes)
        secoes_liberadas = len(self.secoes_liberadas)
        progresso_geral = (secoes_liberadas / total_secoes) * 100 if total_secoes else 0
        progress_frame = tk.Frame(info_frame, bg=self.COR_FUNDO)
        progress_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(progress_frame, text="📊 Progresso Geral:", font=("Helvetica", 18),
                 bg=self.COR_FUNDO, fg=self.COR_TEXTO_ESCURO).pack(side=tk.LEFT)
        progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=150,
                                       mode='determinate', style="Custom.Horizontal.TProgressbar")
        progress_bar['value'] = progresso_geral
        progress_bar.pack(side=tk.LEFT, padx=5)
        tk.Label(progress_frame, text=f"{secoes_liberadas}/{total_secoes}", font=("Helvetica", 14),
                 bg=self.COR_FUNDO, fg=self.COR_TEXTO_ESCURO).pack(side=tk.LEFT)

    def criar_rodape_secoes(self, parent):
        footer_frame = tk.Frame(parent, bg=self.COR_FUNDO)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        tk.Button(footer_frame, text="Sair", font=("Helvetica", 14),
                  bg=self.COR_ERRO, fg=self.COR_TEXTO_CLARO, padx=20, pady=5,
                  command=self.sair).pack(side=tk.RIGHT, padx=10)
        tk.Button(footer_frame, text="Voltar", font=("Helvetica", 14),
                  bg=self.COR_SECUNDARIA, fg=self.COR_TEXTO_ESCURO, padx=20, pady=5,
                  command=self.mostrar_tela_inicial).pack(side=tk.RIGHT)

    def criar_card(self, parent, secao):
        secao_liberada = secao in self.secoes_liberadas
        cor_titulo = self.COR_PRIMARIA if secao_liberada else self.COR_BLOQUEADO
        card_frame = tk.Frame(parent, bg=self.COR_CARD, bd=0, highlightbackground=self.COR_BORDA,
                              highlightthickness=1, padx=15, pady=15)
        header_frame = tk.Frame(card_frame, bg=self.COR_CARD)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(header_frame, text=self.icones_secoes.get(secao, "📁"),
                 font=("Helvetica", 26), bg=self.COR_CARD, fg=cor_titulo).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(header_frame, text=secao, font=("Helvetica", 18, "bold"),
                 bg=self.COR_CARD, fg=cor_titulo).pack(side=tk.LEFT, fill=tk.X, expand=True)
        total_niveis = len(self.secoes[secao])
        niveis_completos = len(self.niveis_completos[secao])
        progresso = (niveis_completos / total_niveis) * 100 if total_niveis > 0 else 0
        progress_frame = tk.Frame(card_frame, bg=self.COR_CARD)
        progress_frame.pack(fill=tk.X, pady=5)
        tk.Label(progress_frame, text="Progresso:", font=("Helvetica", 14), bg=self.COR_CARD,
                 fg=self.COR_TEXTO_ESCURO).pack(side=tk.LEFT, anchor="w")
        progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=100,
                                       mode='determinate', style="Custom.Horizontal.TProgressbar")
        progress_bar['value'] = progresso
        progress_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Label(progress_frame, text=f"{niveis_completos}/{total_niveis}", font=("Helvetica", 14),
                 bg=self.COR_CARD, fg=self.COR_TEXTO_ESCURO).pack(side=tk.LEFT)
        niveis_frame = tk.Frame(card_frame, bg=self.COR_CARD)
        niveis_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        num_niveis = len(self.secoes[secao])
        colunas = 6 if num_niveis > 6 else num_niveis
        for i, nivel in enumerate(self.secoes[secao].keys()):
            nivel_completo = nivel in self.niveis_completos[secao]
            nivel_liberado = (nivel == 1 and secao_liberada) or nivel_completo
            cor_botao = self.COR_PRIMARIA if nivel_liberado else self.COR_BLOQUEADO
            estado = tk.NORMAL if nivel_liberado else tk.DISABLED
            btn_frame = tk.Frame(niveis_frame, bg=self.COR_CARD)
            btn_frame.grid(row=i // colunas, column=i % colunas, padx=3, pady=3)
            btn_nivel = tk.Button(btn_frame, text=str(nivel), font=("Helvetica", 14, "bold"),
                                  width=3, height=1, bg=cor_botao, fg=self.COR_TEXTO_CLARO,
                                  bd=0, state=estado,
                                  command=lambda s=secao, n=nivel: self.iniciar_nivel(s, n))
            btn_nivel.pack()
        return card_frame

    def iniciar_nivel(self, secao, nivel):
        """Inicia um nível com tela de carregamento"""
        self.limpar_tela()
        self.criar_tela_carregamento(secao, nivel)
        self.root.after(100, lambda: self.carregar_nivel_background(secao, nivel, 0))

    def carregar_nivel_background(self, secao, nivel, progresso):
        if progresso <= 33:
            self.loading_stage.config(text="Inicializando câmera...")
            if progresso == 10:
                self.pre_iniciar_camera()
        elif progresso <= 66:
            self.loading_stage.config(text="Carregando modelo de reconhecimento...")
        else:
            self.loading_stage.config(text="Preparando interface...")
        if progresso <= 100:
            self.loading_progress['value'] = progresso
            self.loading_percent.config(text=f"{progresso}%")
            self.root.update_idletasks()
            incremento = 1 if progresso > 90 else 2 if progresso > 50 else 5
            self.root.after(50, lambda: self.carregar_nivel_background(secao, nivel, progresso + incremento))
        else:
            self.loading_frame.destroy()
            self.iniciar_nivel_real(secao, nivel)

    def criar_tela_carregamento(self, secao, nivel):
        self.loading_frame = tk.Frame(self.root, bg=self.COR_FUNDO)
        self.loading_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(self.loading_frame, text=f"Preparando nível {nivel} de {secao}...",
                font=("Helvetica", 18, "bold"), bg=self.COR_FUNDO, fg=self.COR_PRIMARIA).pack(pady=10)
        self.loading_stage = tk.Label(self.loading_frame, text="Inicializando componentes...",
                                    font=("Helvetica", 12), bg=self.COR_FUNDO, fg=self.COR_TEXTO_ESCURO)
        self.loading_stage.pack(pady=5)
        self.loading_spinner = tk.Label(self.loading_frame, text="⏳", font=("Helvetica", 24),
                                      bg=self.COR_FUNDO, fg=self.COR_PRIMARIA)
        self.loading_spinner.pack(pady=10)
        self.loading_progress = ttk.Progressbar(self.loading_frame, orient=tk.HORIZONTAL,
                                              length=300, mode='determinate')
        self.loading_progress.pack(pady=10)
        self.loading_percent = tk.Label(self.loading_frame, text="0%", font=("Helvetica", 14),
                                      bg=self.COR_FUNDO, fg=self.COR_TEXTO_ESCURO)
        self.loading_percent.pack(pady=5)
        self.animar_spinner()

    def animar_spinner(self):
        spinners = ["⏳", "⌛", "⏳", "⌛"]
        if hasattr(self, 'loading_spinner'):
            current = self.loading_spinner.cget("text")
            idx = (spinners.index(current) + 1) % len(spinners) if current in spinners else 0
            self.loading_spinner.config(text=spinners[idx])
            self.root.after(300, self.animar_spinner)

    def pre_iniciar_camera(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                ret, frame = self.cap.read()
                if ret:
                    self.cap.release()
                    self.cap = None
        except Exception as e:
            logging.warning(f"Erro na pré-inicialização da câmera: {e}")

    def iniciar_nivel_real(self, secao, nivel):
        self.secao_atual = secao
        self.nivel_atual = nivel
        self.letras_nivel = self.secoes[secao][nivel]
        self.letra_atual_idx = 0
        self.tempo_inicio = time.time()
        main_frame = tk.Frame(self.root, bg=self.COR_FUNDO)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        self.criar_barra_superior(main_frame)
        content_frame = tk.Frame(main_frame, bg=self.COR_FUNDO)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=10)
        content_frame.columnconfigure(0, weight=18)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        left_frame = self.criar_frame_gesto_alvo(content_frame)
        right_frame = self.criar_frame_camera(content_frame)
        self.criar_controles_inferiores(main_frame)
        self.atualizar_progresso()
        self.iniciar_camera()
        self.proxima_letra()
        self.atualizar_tempo()

    def criar_barra_superior(self, parent):
        top_frame = tk.Frame(parent, bg=self.COR_PRIMARIA)
        top_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0, ipady=10)
        tk.Button(top_frame, text="← Voltar", font=("Helvetica", 12, "bold"),
                 bg=self.COR_PRIMARIA, fg=self.COR_TEXTO_CLARO, bd=0,
                 command=self.mostrar_tela_secoes).pack(side=tk.LEFT, padx=10)
        tk.Label(top_frame, text=f"{self.secao_atual} - Nível {self.nivel_atual}",
                font=("Helvetica", 16, "bold"), bg=self.COR_PRIMARIA, fg=self.COR_TEXTO_CLARO).pack(side=tk.LEFT, expand=True)
        info_frame = tk.Frame(top_frame, bg=self.COR_PRIMARIA)
        info_frame.pack(side=tk.RIGHT, padx=10)
        timer_frame = tk.Frame(info_frame, bg=self.COR_PRIMARIA)
        timer_frame.pack(side=tk.LEFT, padx=(0, 20))
        tk.Label(timer_frame, text="⏱️ ", font=("Helvetica", 12),
                bg=self.COR_PRIMARIA, fg=self.COR_SECUNDARIA).pack(side=tk.LEFT)
        self.tempo_label = tk.Label(timer_frame, text="00:00", font=("Helvetica", 12, "bold"),
                                  bg=self.COR_PRIMARIA, fg=self.COR_TEXTO_CLARO)
        self.tempo_label.pack(side=tk.LEFT)
        pontos_frame = tk.Frame(info_frame, bg=self.COR_PRIMARIA)
        pontos_frame.pack(side=tk.LEFT)
        tk.Label(pontos_frame, text="🏆 ", font=("Helvetica", 12),
                bg=self.COR_PRIMARIA, fg=self.COR_SECUNDARIA).pack(side=tk.LEFT)
        self.pontuacao_label = tk.Label(pontos_frame, text=f"{self.pontuacao}", font=("Helvetica", 12, "bold"),
                                      bg=self.COR_PRIMARIA, fg=self.COR_TEXTO_CLARO)
        self.pontuacao_label.pack(side=tk.LEFT)

    def criar_frame_gesto_alvo(self, parent):
        frame = tk.Frame(parent, bg=self.COR_CARD, bd=1, relief=tk.RAISED,
                        highlightbackground=self.COR_BORDA, highlightthickness=1)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        tk.Label(frame, text="Gesto Alvo", font=("Helvetica", 14, "bold"),
                bg=self.COR_CARD, fg=self.COR_PRIMARIA, pady=10).grid(row=0, column=0, sticky="ew")
        content_frame = tk.Frame(frame, bg=self.COR_CARD)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        content_frame.rowconfigure(1, weight=1)
        self.gesto_alvo_label = tk.Label(content_frame, text="", font=("Helvetica", 72, "bold"),
                                    bg=self.COR_CARD, fg=self.COR_PRIMARIA)
        self.gesto_alvo_label.grid(row=0, column=0, sticky="s")
        self.imagem_letra_label = tk.Label(content_frame, bg=self.COR_CARD)
        self.imagem_letra_label.grid(row=1, column=0, sticky="n", pady=(20, 0))
        return frame

    def criar_frame_camera(self, parent):
        frame = tk.Frame(parent, bg=self.COR_CARD, bd=1, relief=tk.RAISED,
                        highlightbackground=self.COR_BORDA, highlightthickness=1)
        frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        tk.Label(frame, text="Sua Câmera", font=("Helvetica", 14, "bold"),
                bg=self.COR_CARD, fg=self.COR_PRIMARIA, pady=10).grid(row=0, column=0, sticky="ew")
        video_container = tk.Frame(frame, bg="white", padx=0, pady=0)
        video_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.video_label = tk.Label(video_container, bg="white")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        return frame

    def criar_controles_inferiores(self, parent):
        control_frame = tk.Frame(parent, bg=self.COR_FUNDO)
        control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=3)
        control_frame.columnconfigure(2, weight=1)
        self.btn_camera = tk.Button(control_frame, text="⏸️ Parar Câmera",
                                  font=("Helvetica", 12, "bold"), bg=self.COR_PRIMARIA,
                                  fg=self.COR_TEXTO_CLARO, bd=0, padx=15, pady=8,
                                  command=self.toggle_camera)
        self.btn_camera.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.feedback_label = tk.Label(control_frame, text="Mostre o gesto para a câmera",
                                     font=("Helvetica", 14), bg=self.COR_FUNDO,
                                     fg=self.COR_TEXTO_ESCURO)
        self.feedback_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.progress_frame = tk.Frame(control_frame, bg=self.COR_FUNDO)
        self.progress_frame.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.progress_label = tk.Label(self.progress_frame, text="Progresso:",
                                     font=("Helvetica", 12), bg=self.COR_FUNDO,
                                     fg=self.COR_TEXTO_ESCURO)
        self.progress_label.pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL,
                                         length=150, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.progress_text = tk.Label(self.progress_frame, text="0/0",
                                    font=("Helvetica", 12), bg=self.COR_FUNDO,
                                    fg=self.COR_TEXTO_ESCURO)
        self.progress_text.pack(side=tk.LEFT)

    def atualizar_tempo(self):
        if hasattr(self, 'tempo_inicio') and self.tempo_inicio > 0:
            tempo_decorrido = time.time() - self.tempo_inicio
            minutos = int(tempo_decorrido // 60)
            segundos = int(tempo_decorrido % 60)
            self.tempo_label.config(text=f"{minutos:02d}:{segundos:02d}")
        self.root.after(1000, self.atualizar_tempo)

    def atualizar_progresso(self):
        total = len(self.letras_nivel)
        atual = self.letra_atual_idx
        self.progress_bar['maximum'] = total
        self.progress_bar['value'] = atual
        self.progress_text.config(text=f"{atual}/{total}")

    def proxima_letra(self):
        if self.letra_atual_idx < len(self.letras_nivel):
            self.gesto_alvo = self.letras_nivel[self.letra_atual_idx]
            self.gesto_alvo_label.config(text=self.gesto_alvo)
            self.letra_atual_idx += 1
            self.feedback_label.config(text="Mostre o gesto para a câmera", fg=self.COR_TEXTO_ESCURO)
            self.atualizar_progresso()
            self.atualizar_imagem_letra()
        else:
            self.tempo_gasto = time.time() - self.tempo_inicio
            if self.nivel_atual not in self.niveis_completos[self.secao_atual]:
                self.niveis_completos[self.secao_atual].append(self.nivel_atual)
            self.mostrar_tela_parabens()

    def atualizar_imagem_letra(self):
        if not hasattr(self, 'gesto_alvo') or not self.gesto_alvo:
            return
        letra = self.gesto_alvo.lower()
        caminho_imagem = f"libras_alfabeto_projeto/imagens/{letra}.png"
        try:
            img = Image.open(caminho_imagem)
            img = img.resize((200, 200), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            self.imagem_letra_label.config(image=img)
            self.imagem_letra_label.image = img
        except FileNotFoundError:
            self.imagem_letra_label.config(text="Imagem não disponível", font=("Helvetica", 14), fg=self.COR_ERRO)
        except Exception as e:
            logging.warning(f"Erro ao carregar imagem: {e}")
            self.imagem_letra_label.config(text="Erro ao carregar imagem", font=("Helvetica", 14), fg=self.COR_ERRO)

    def toggle_camera(self):
        if self.running:
            self.parar_camera()
            self.btn_camera.config(text="▶️ Iniciar Câmera")
        else:
            self.iniciar_camera()
            self.btn_camera.config(text="⏸️ Parar Câmera")

    def iniciar_camera(self):
        if self.running:
            return
        try:
            start_time = time.time()
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            while not self.cap.isOpened() and (time.time() - start_time) < self.camera_timeout:
                time.sleep(0.05)
            if not self.cap.isOpened():
                messagebox.showerror("Erro", "Timeout ao acessar a câmera")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            self.cap.set(cv2.CAP_PROP_FPS, 20)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.running = True
            self.frame_count = 0
            self.skip_frames = 1
            self.feedback_label.config(text="Câmera iniciada - Ajustando...")
            self.atualizar_frame()
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao iniciar câmera: {e}")
            self.cap = None

    def atualizar_frame(self):
        if not self.running:
            return
        try:
            self.frame_count += 1
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Erro ao capturar frame - tentando reiniciar câmera")
                self.reiniciar_camera()
                return
            if self.frame_count % (self.skip_frames + 1) == 0:
                frame_proc = self.processar_frame(frame)
                self.mostrar_frame(frame_proc)
        except Exception as e:
            logging.exception("Erro no loop da câmera:")
            self.reiniciar_camera()
        finally:
            if self.running:
                self.root.after(30, self.atualizar_frame)

    def reiniciar_camera(self):
        self.parar_camera()
        time.sleep(0.5)
        self.iniciar_camera()

    def processar_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            self.frames_sem_maos = 0
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            landmarks = self.processar_landmarks(results)
            self.buffer_gestos.append(landmarks)
            if len(self.buffer_gestos) == SEQUENCE_LENGTH and not self.predicting:
                entrada = np.array(self.buffer_gestos).reshape(1, SEQUENCE_LENGTH, 126)
                self.predicting = True
                threading.Thread(target=self._predict_thread, args=(entrada,), daemon=True).start()
        else:
            self.frames_sem_maos += 1
            if self.frames_sem_maos > self.RESET_THRESHOLD and self.buffer_gestos:
                self.buffer_gestos.clear()
                self.feedback_label.config(text="Mãos não detectadas", fg=self.COR_ERRO)
        return frame_rgb

    def _predict_thread(self, entrada):
        try:
            if not self.modelo_gestos:
                result = None
            else:
                preds = self.modelo_gestos.predict(entrada, verbose=0)[0]
                result = preds
        except Exception as e:
            logging.exception("Erro durante predição:")
            result = None
        self.root.after(0, lambda: self._handle_prediction_result(result))

    def _handle_prediction_result(self, preds):
        try:
            if preds is None:
                self.reconhecer_gesto_simulado()
                self.predicting = False
                return

            top_idx = int(np.argmax(preds))
            top_conf = float(preds[top_idx])
            sorted_idxs = np.argsort(preds)
            second_idx = int(sorted_idxs[-2])
            second_conf = float(preds[second_idx])
            margin = top_conf - second_conf
            gesto_pred = str(self.le_gestos.classes_[top_idx])

            logging.info(f"Predição: {gesto_pred} conf={top_conf:.3f} margin={margin:.3f}")

            if top_conf >= MIN_CONFIDENCE and margin >= CONFIDENCE_MARGIN:
                self.historico_predicoes.append(gesto_pred)
                contagem = defaultdict(int)
                for g in self.historico_predicoes:
                    contagem[g] += 1
                gesto_mais_comum, qtd = max(contagem.items(), key=lambda x: x[1])
                ratio = qtd / len(self.historico_predicoes)
                logging.info(f"Votação: {gesto_mais_comum} ({qtd}/{len(self.historico_predicoes)}) ratio={ratio:.2f}")

                if ratio >= REQUIRED_VOTE_RATIO and gesto_mais_comum == self.gesto_alvo:
                    self.pontuacao += 10 * self.nivel_atual
                    self.pontuacao_label.config(text=f"{self.pontuacao}")
                    self.feedback_label.config(text=f"✅ Correto! {gesto_mais_comum}", fg=self.COR_SUCESSO)
                    self.root.after(1000, self.proxima_letra)
                    self.historico_predicoes.clear()
                    for _ in range(SLIDE_AFTER_DETECT):
                        if self.buffer_gestos:
                            self.buffer_gestos.popleft()
                else:
                    if gesto_pred != self.ultimo_gesto_reconhecido:
                        self.feedback_label.config(text=f"Reconhecido: {gesto_pred} (Mostre: {self.gesto_alvo})", fg=self.COR_SECUNDARIA)
            else:
                if self.le_gestos is not None:
                    if str(self.le_gestos.classes_[top_idx]) != self.ultimo_gesto_reconhecido:
                        self.feedback_label.config(text=f"Tente novamente: {str(self.le_gestos.classes_[top_idx])}", fg=self.COR_ERRO)
        except Exception as e:
            logging.exception("Erro ao processar resultado de predição:")
            self.reconhecer_gesto_simulado()
        finally:
            try:
                self.ultimo_gesto_reconhecido = (str(self.le_gestos.classes_[int(np.argmax(preds))]) if preds is not None and self.le_gestos is not None else None)
            except Exception:
                self.ultimo_gesto_reconhecido = None
            self.predicting = False

    def reconhecer_gesto_simulado(self):
        if random.random() < 0.3:
            self.pontuacao += 10
            self.pontuacao_label.config(text=f"{self.pontuacao}")
            self.feedback_label.config(text=f"✅ Correto! {self.gesto_alvo}", fg=self.COR_SUCESSO)
            self.root.after(1000, self.proxima_letra)
            self.buffer_gestos.clear()
        else:
            self.feedback_label.config(text=f"Tente novamente: {self.gesto_alvo}", fg=self.COR_ERRO)

    def processar_landmarks(self, results):
        landmarks = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        landmarks = landmarks[:42]
        if len(landmarks) < 42:
            landmarks.extend([[0, 0, 0]] * (42 - len(landmarks)))
        return np.array(landmarks).flatten()

    def mostrar_frame(self, frame):
        img = Image.fromarray(frame)
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        if label_width <= 1 or label_height <= 1:
            label_width = 800
            label_height = 600
        img_ratio = img.width / img.height
        label_ratio = label_width / label_height
        if label_ratio > img_ratio:
            new_height = label_height
            new_width = int(new_height * img_ratio)
        else:
            new_width = label_width
            new_height = int(new_width / img_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = img
        self.video_label.config(image=img)

    def parar_camera(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def limpar_tela(self):
        self.parar_camera()
        for widget in self.root.winfo_children():
            widget.destroy()

    def sair(self):
        self.parar_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicativoLibras(root)
    root.protocol("WM_DELETE_WINDOW", app.sair)
    root.mainloop()
