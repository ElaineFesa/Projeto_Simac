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

class AplicativoLibras:
    def __init__(self, root):
        # Configuração do tema
        self.COR_PRIMARIA = "#6A0DAD"  # Roxo
        self.COR_SECUNDARIA = "#FFD700"  # Dourado
        self.COR_FUNDO = "#F5F5F5"  # Cinza claro
        self.COR_TEXTO_CLARO = "#FFFFFF"  # Branco
        self.COR_TEXTO_ESCURO = "#333333"  # Cinza escuro
        self.COR_SUCESSO = "#2E8B57"  # Verde para sucesso
        self.COR_ERRO = "#DC143C"  # Vermelho para erro
        self.COR_BLOQUEADO = "#CCCCCC"  # Cinza
        self.COR_CARD = "#FFFFFF"  # Branco para cards
        self.COR_BORDA = "#E0E0E0"  # Cinza claro para bordas
        self.COR_SOMBRA = "#DDDDDD"  # Cor para sombra

        # Configurações de tamanho
        self.root = root
        self.root.title("LIA")
        self.root.state('zoomed')  # Iniciar maximizado
        self.root.configure(bg=self.COR_FUNDO)

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
        self.frame_count = 0
        self.skip_frames = 2
        self.nivel_atual = 1
        self.pontuacao = 0
        self.gesto_alvo = None
        self.buffer_gestos = deque(maxlen=30)
        self.historico_predicoes = deque(maxlen=15)  # Para suavização
        self.frames_sem_maos = 0
        self.RESET_THRESHOLD = 10
        self.niveis_completos = {}
        self.secoes_liberadas = ["Alfabeto"]
        self.tempo_inicio = 0
        self.tempo_gasto = 0
        self.ultimo_gesto_reconhecido = None

        # Carregar modelo de gestos
        self.modelo_gestos, self.le_gestos = self.carregar_modelo_gestos()

        # Estrutura de seções e níveis
        self.secoes = {
            "Alfabeto": {
                1: ["A", "E", "I", "O", "U"],
                2: ["B", "C", "D", "G", "I", "L"],
                3: ["M", "N", "O", "P", "Q", "R"],
                4: ["S", "T", "U", "V", "W"],
                5: ["H", "J", "K", "X"],
                6: ["W", "Y", "Z"]
            },
            "Saudações": {
                1: ["OI", "TCHAU", "TUDO BEM"],
                2: ["POR FAVOR", "OBRIGADO", "DESCULPA"],
                3: ["BOM DIA", "BOA TARDE", "BOA NOITE"]
            },
            "Família": {
                1: ["PAI", "MÃE", "IRMÃO", "IRMÃ"],
                2: ["AVÔ", "AVÓ", "PADRASTO", "MADRASTA"],
                3: ["TIO", "TIA", "PRIMO", "PRIMA"],
                4: ["CUNHADA", "CUNHADO", "SOGRO", "SOGRA"],
                5: ["NAMORADO", "NAMORADA", "NOIVO", "NOIVA"],
                6: ["ESPOSO", "ESPOSA", "FILHO", "FILHA"]
            },
            "Alimentos": {
                1: ["MAÇÃ", "LARANJA", "UVA", "MELANCIA"],
                2: ["LIMÃO", "MELÃO", "TOMATE", "ABACAXI"]
            },
            "Cores": {
                1: ["AZUL", "AMARELO", "VERDE", "VERMELHO"],
                2: ["ROSA", "ROXO", "LARANJA", "BRANCO", "PRETO"]
            },
            "Animais": {
                1: ["CÃO", "GATO", "PEIXE", "CAVALO"],
                2: ["MACACO", "LEÃO", "BALEIA", "PÁSSARO"],
                3: ["FORMIGA", "ABELHA", "BORBOLETA"]
            },
            "Pronomes": {
                1: ["EU", "VOCÊ", "ELES", "NÓS"]
            }
        }

        # Ícones para cada seção
        self.icones_secoes = {
            "Alfabeto": "🔤",
            "Saudações": "👋",
            "Família": "👨‍👩‍👧‍👦",
            "Alimentos": "🍎",
            "Cores": "🎨",
            "Animais": "🐶",
            "Pronomes": "📍"
        }

        for secao in self.secoes:
            self.niveis_completos[secao] = []

        self.mostrar_tela_inicial()

    def carregar_modelo_gestos(self):
        """Carrega o modelo de gestos e o rotulador"""
        try:
            modelos_dir = Path("modelos")
            modelo_path = modelos_dir / "modelo_gestos.h5"
            rotulador_path = modelos_dir / "rotulador_gestos.pkl"
            
            if not modelo_path.exists() or not rotulador_path.exists():
                messagebox.showerror("Erro", 
                    "Modelo de gestos não encontrado!\n\n"
                    "Verifique se os arquivos estão em:\n"
                    f"{modelo_path}\n{rotulador_path}")
                return None, None
            
            modelo = load_model(modelo_path)
            le = joblib.load(rotulador_path)
            print(f"Modelo carregado. Classes: {list(le.classes_)}")
            return modelo, le
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar modelo: {str(e)}")
            return None, None

    def criar_card(self, parent, secao):
        """Cria um card estilizado para cada seção"""
        secao_liberada = secao in self.secoes_liberadas
        cor_titulo = self.COR_PRIMARIA if secao_liberada else self.COR_BLOQUEADO
        
        # Frame principal do card
        card_frame = tk.Frame(
            parent,
            bg=self.COR_CARD,
            bd=0,
            highlightbackground=self.COR_BORDA,
            highlightthickness=1,
            padx=15,
            pady=15
        )
        
        # Cabeçalho do card
        header_frame = tk.Frame(card_frame, bg=self.COR_CARD)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Ícone da seção
        icone = tk.Label(
            header_frame,
            text=self.icones_secoes.get(secao, "📁"),
            font=("Helvetica", 26),
            bg=self.COR_CARD,
            fg=cor_titulo
        )
        icone.pack(side=tk.LEFT, padx=(0, 10))
        
        # Título da seção
        titulo = tk.Label(
            header_frame,
            text=secao,
            font=("Helvetica", 18, "bold"),
            bg=self.COR_CARD,
            fg=cor_titulo
        )
        titulo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Barra de progresso
        total_niveis = len(self.secoes[secao])
        niveis_completos = len(self.niveis_completos[secao])
        progresso = (niveis_completos / total_niveis) * 100 if total_niveis > 0 else 0
        
        progress_frame = tk.Frame(card_frame, bg=self.COR_CARD)
        progress_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            progress_frame,
            text="Progresso:",
            font=("Helvetica", 14),
            bg=self.COR_CARD,
            fg=self.COR_TEXTO_ESCURO
        ).pack(side=tk.LEFT, anchor="w")
        
        progress_bar = ttk.Progressbar(
            progress_frame,
            orient=tk.HORIZONTAL,
            length=100,
            mode='determinate',
            style="Custom.Horizontal.TProgressbar"
        )
        progress_bar['value'] = progresso
        progress_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        tk.Label(
            progress_frame,
            text=f"{niveis_completos}/{total_niveis}",
            font=("Helvetica", 14),
            bg=self.COR_CARD,
            fg=self.COR_TEXTO_ESCURO
        ).pack(side=tk.LEFT)
        
        # Botões de nível em grade
        niveis_frame = tk.Frame(card_frame, bg=self.COR_CARD)
        niveis_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Calcular número de colunas com base no número de níveis
        num_niveis = len(self.secoes[secao])
        colunas = 6 if num_niveis > 6 else num_niveis
        
        for i, nivel in enumerate(self.secoes[secao].keys()):
            nivel_completo = nivel in self.niveis_completos[secao]
            nivel_liberado = (nivel == 1 and secao_liberada) or nivel_completo
            cor_botao = self.COR_PRIMARIA if nivel_liberado else self.COR_BLOQUEADO
            estado = tk.NORMAL if nivel_liberado else tk.DISABLED
            
            btn_frame = tk.Frame(niveis_frame, bg=self.COR_CARD)
            btn_frame.grid(row=i // colunas, column=i % colunas, padx=3, pady=3)
            
            btn_nivel = tk.Button(
                btn_frame,
                text=str(nivel),
                font=("Helvetica", 14, "bold"),
                width=3,
                height=1,
                bg=cor_botao,
                fg=self.COR_TEXTO_CLARO,
                bd=0,
                state=estado,
                command=lambda s=secao, n=nivel: self.iniciar_nivel(s, n)
            )
            
            if nivel_completo:
                btn_nivel.config(relief=tk.SUNKEN)
            
            btn_nivel.pack()
            
        return card_frame

    def mostrar_tela_inicial(self):
        """Tela de splash com logo"""
        self.limpar_tela()
        
        splash_frame = tk.Frame(self.root, bg=self.COR_PRIMARIA)
        splash_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(
            splash_frame, 
            text="👋",  
            font=("Helvetica", 100),
            bg=self.COR_PRIMARIA,
            fg=self.COR_SECUNDARIA
        ).pack(pady=20)
        
        tk.Label(
            splash_frame,
            text="LIA",
            font=("Helvetica", 100, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO
        ).pack(pady=10)
        
        self.root.after(1500, self.mostrar_tela_secoes)

    def mostrar_tela_secoes(self):
        """Tela com seções e níveis - Layout otimizado para ocupar espaço"""
        self.limpar_tela()
        
        # Configurar estilos
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar", 
                         troughcolor=self.COR_FUNDO, 
                         background=self.COR_PRIMARIA,
                         bordercolor=self.COR_BORDA,
                         lightcolor=self.COR_PRIMARIA,
                         darkcolor=self.COR_PRIMARIA)
        
        # Frame principal com expansão
        main_frame = tk.Frame(self.root, bg=self.COR_FUNDO)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Cabeçalho com informações
        header_frame = tk.Frame(main_frame, bg=self.COR_FUNDO)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Título
        tk.Label(
            header_frame,
            text="Seções",
            font=("Helvetica", 26, "bold"),
            bg=self.COR_FUNDO,
            fg=self.COR_PRIMARIA
        ).pack(side=tk.LEFT)
        
        # Informações à direita
        info_frame = tk.Frame(header_frame, bg=self.COR_FUNDO)
        info_frame.pack(side=tk.RIGHT, padx=10)
        
        # Pontuação
        tk.Label(
            info_frame,
            text=f"🏆 Pontuação: {self.pontuacao}",
            font=("Helvetica", 18, "bold"),
            bg=self.COR_FUNDO,
            fg=self.COR_PRIMARIA
        ).pack(side=tk.LEFT, padx=10)
        
        # Progresso geral
        total_secoes = len(self.secoes)
        secoes_liberadas = len(self.secoes_liberadas)
        progresso_geral = (secoes_liberadas / total_secoes) * 100
        
        progress_frame = tk.Frame(info_frame, bg=self.COR_FUNDO)
        progress_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            progress_frame,
            text="📊 Progresso Geral:",
            font=("Helvetica", 18),
            bg=self.COR_FUNDO,
            fg=self.COR_TEXTO_ESCURO
        ).pack(side=tk.LEFT)
        
        progress_bar = ttk.Progressbar(
            progress_frame,
            orient=tk.HORIZONTAL,
            length=150,
            mode='determinate',
            style="Custom.Horizontal.TProgressbar"
        )
        progress_bar['value'] = progresso_geral
        progress_bar.pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            progress_frame,
            text=f"{secoes_liberadas}/{total_secoes}",
            font=("Helvetica", 14),
            bg=self.COR_FUNDO,
            fg=self.COR_TEXTO_ESCURO
        ).pack(side=tk.LEFT)
        
        # Container para os cards com rolagem
        container = tk.Frame(main_frame, bg=self.COR_FUNDO)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas com rolagem vertical
        canvas = tk.Canvas(container, bg=self.COR_FUNDO, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COR_FUNDO)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Organizar seções em grade responsiva
        colunas = max(3, min(4, self.root.winfo_screenwidth() // 300))
        
        for i, secao in enumerate(self.secoes):
            card = self.criar_card(scrollable_frame, secao)
            card.grid(
                row=i // colunas, 
                column=i % colunas, 
                padx=10, 
                pady=10, 
                sticky="nsew"
            )
        
        # Configurar expansão uniforme
        for col in range(colunas):
            scrollable_frame.columnconfigure(col, weight=1, uniform="group1")
        
        for row in range((len(self.secoes) + colunas - 1) // colunas):
            scrollable_frame.rowconfigure(row, weight=1)
        
        # Rodapé
        footer_frame = tk.Frame(main_frame, bg=self.COR_FUNDO)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        btn_sair = tk.Button(
            footer_frame,
            text="Sair",
            font=("Helvetica", 14),
            bg=self.COR_ERRO,
            fg=self.COR_TEXTO_CLARO,
            padx=20,
            pady=5,
            command=self.sair
        )
        btn_sair.pack(side=tk.RIGHT, padx=10)
        
        btn_inicio = tk.Button(
            footer_frame,
            text="Voltar",
            font=("Helvetica", 14),
            bg=self.COR_SECUNDARIA,
            fg=self.COR_TEXTO_ESCURO,
            padx=20,
            pady=5,
            command=self.mostrar_tela_inicial
        )
        btn_inicio.pack(side=tk.RIGHT)

    def mostrar_tela_parabens(self):
        """Tela de parabéns ao completar nível com tempo gasto"""
        self.limpar_tela()
        
        parabens_frame = tk.Frame(self.root, bg=self.COR_FUNDO)
        parabens_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        btn_fechar = tk.Button(
            self.root,
            text="✕",
            font=("Helvetica", 16),
            bg=self.COR_FUNDO,
            fg=self.COR_ERRO,
            bd=0,
            command=self.mostrar_tela_secoes
        )
        btn_fechar.place(relx=0.95, rely=0.05, anchor="ne")
        
        tk.Label(
            parabens_frame,
            text="🎉 Parabéns! 🎉",
            font=("Helvetica", 24, "bold"),
            bg=self.COR_FUNDO,
            fg=self.COR_PRIMARIA
        ).pack(pady=20)
        
        tk.Label(
            parabens_frame,
            text=f"Você completou o nível {self.nivel_atual} da seção {self.secao_atual}!",
            font=("Helvetica", 16),
            bg=self.COR_FUNDO,
            fg=self.COR_TEXTO_ESCURO
        ).pack(pady=10)
        
        # Exibir tempo gasto formatado
        minutos = int(self.tempo_gasto // 60)
        segundos = int(self.tempo_gasto % 60)
        tempo_formatado = f"{minutos} min {segundos} seg" if minutos > 0 else f"{segundos} segundos"
        
        tk.Label(
            parabens_frame,
            text=f"⏱️ Tempo gasto: {tempo_formatado}",
            font=("Helvetica", 14),
            bg=self.COR_FUNDO,
            fg=self.COR_SECUNDARIA
        ).pack(pady=5)
        
        tk.Label(
            parabens_frame,
            text=f"🏆 Pontuação atual: {self.pontuacao}",
            font=("Helvetica", 14),
            bg=self.COR_FUNDO,
            fg=self.COR_SECUNDARIA
        ).pack(pady=5)
        
        botoes_frame = tk.Frame(parabens_frame, bg=self.COR_FUNDO)
        botoes_frame.pack(pady=30)
        
        if self.nivel_atual < len(self.secoes[self.secao_atual]):
            tk.Button(
                botoes_frame,
                text=f"Próximo Nível ({self.nivel_atual + 1})",
                font=("Helvetica", 14, "bold"),
                bg=self.COR_PRIMARIA,
                fg=self.COR_TEXTO_CLARO,
                padx=20,
                pady=10,
                command=self.ir_para_proximo_nivel
            ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            botoes_frame,
            text="Voltar às Seções",
            font=("Helvetica", 14),
            bg=self.COR_SECUNDARIA,
            fg=self.COR_TEXTO_ESCURO,
            padx=20,
            pady=10,
            command=self.mostrar_tela_secoes
        ).pack(side=tk.LEFT, padx=10)

    def ir_para_proximo_nivel(self):
        """Avança para o próximo nível"""
        proximo_nivel = self.nivel_atual + 1
        self.iniciar_nivel(self.secao_atual, proximo_nivel)

    def iniciar_nivel(self, secao, nivel):
        """Inicia um nível específico com design aprimorado e ocupação de espaço"""
        self.limpar_tela()
        self.secao_atual = secao
        self.nivel_atual = nivel
        self.letras_nivel = self.secoes[secao][nivel]
        self.letra_atual_idx = 0
        self.tempo_inicio = time.time()
        
        # Frame principal com grid expansível
        main_frame = tk.Frame(self.root, bg=self.COR_FUNDO)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configurar grid (2 linhas: 1 para cabeçalho, 2 para conteúdo)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Barra superior
        top_frame = tk.Frame(main_frame, bg=self.COR_PRIMARIA)
        top_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0, ipady=10)
        
        # Botão voltar
        btn_voltar = tk.Button(
            top_frame,
            text="← Voltar",
            font=("Helvetica", 12, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO,
            bd=0,
            command=self.mostrar_tela_secoes
        )
        btn_voltar.pack(side=tk.LEFT, padx=10)
        
        # Título
        lbl_titulo = tk.Label(
            top_frame,
            text=f"{self.secao_atual} - Nível {self.nivel_atual}",
            font=("Helvetica", 16, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO
        )
        lbl_titulo.pack(side=tk.LEFT, expand=True)
        
        # Informações (tempo e pontuação)
        info_frame = tk.Frame(top_frame, bg=self.COR_PRIMARIA)
        info_frame.pack(side=tk.RIGHT, padx=10)
        
        # Timer
        timer_frame = tk.Frame(info_frame, bg=self.COR_PRIMARIA)
        timer_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(
            timer_frame,
            text="⏱️ ",
            font=("Helvetica", 12),
            bg=self.COR_PRIMARIA,
            fg=self.COR_SECUNDARIA
        ).pack(side=tk.LEFT)
        
        self.tempo_label = tk.Label(
            timer_frame,
            text="00:00",
            font=("Helvetica", 12, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO
        )
        self.tempo_label.pack(side=tk.LEFT)
        
        # Pontuação
        pontos_frame = tk.Frame(info_frame, bg=self.COR_PRIMARIA)
        pontos_frame.pack(side=tk.LEFT)
        
        tk.Label(
            pontos_frame,
            text="🏆 ",
            font=("Helvetica", 12),
            bg=self.COR_PRIMARIA,
            fg=self.COR_SECUNDARIA
        ).pack(side=tk.LEFT)
        
        self.pontuacao_label = tk.Label(
            pontos_frame,
            text=f"{self.pontuacao}",
            font=("Helvetica", 12, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO
        )
        self.pontuacao_label.pack(side=tk.LEFT)
        
        # Frame do conteúdo principal (gesto alvo + câmera)
        content_frame = tk.Frame(main_frame, bg=self.COR_FUNDO)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=10)
        
        # Configurar grid para conteúdo (1 linha, 2 colunas)
        content_frame.columnconfigure(0, weight=18)  # Gesto alvo
        content_frame.columnconfigure(1, weight=1)  # Câmera (3x mais espaço)
        content_frame.rowconfigure(0, weight=1)
        
        # Frame do gesto alvo
        left_frame = tk.Frame(
            content_frame,
            bg=self.COR_CARD,
            bd=1,
            relief=tk.RAISED,
            highlightbackground=self.COR_BORDA,
            highlightthickness=1
        )
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        
        # Configurar expansão do frame do gesto alvo
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        
        tk.Label(
            left_frame,
            text="Gesto Alvo",
            font=("Helvetica", 14, "bold"),
            bg=self.COR_CARD,
            fg=self.COR_PRIMARIA,
            pady=10
        ).grid(row=0, column=0, sticky="ew")
        
        self.gesto_alvo_label = tk.Label(
            left_frame,
            text="",
            font=("Helvetica", 72, "bold"),
            bg=self.COR_CARD,
            fg=self.COR_PRIMARIA,
            pady=20
        )
        self.gesto_alvo_label.grid(row=1, column=0, sticky="nsew")
        
        # Frame da câmera (agora com mais espaço)
        right_frame = tk.Frame(
            content_frame,
            bg=self.COR_CARD,
            bd=1,
            relief=tk.RAISED,
            highlightbackground=self.COR_BORDA,
            highlightthickness=1
        )
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        
        # Configurar expansão do frame da câmera
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        tk.Label(
            right_frame,
            text="Sua Câmera",
            font=("Helvetica", 14, "bold"),
            bg=self.COR_CARD,
            fg=self.COR_PRIMARIA,
            pady=10
        ).grid(row=0, column=0, sticky="ew")
        
        # Container para o vídeo com bordas internas
        video_container = tk.Frame(right_frame, bg="black", padx=0, pady=0)
        video_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.video_label = tk.Label(video_container, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Controles inferiores
        control_frame = tk.Frame(main_frame, bg=self.COR_FUNDO)
        control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
        
        # Configurar grid para controles
        control_frame.columnconfigure(0, weight=1)  # Botão câmera
        control_frame.columnconfigure(1, weight=3)  # Feedback
        control_frame.columnconfigure(2, weight=1)  # Progresso
        
        self.btn_camera = tk.Button(
            control_frame,
            text="⏸️ Parar Câmera",
            font=("Helvetica", 12, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO,
            bd=0,
            padx=15,
            pady=8,
            command=self.toggle_camera
        )
        self.btn_camera.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.feedback_label = tk.Label(
            control_frame,
            text="Mostre o gesto para a câmera",
            font=("Helvetica", 14),
            bg=self.COR_FUNDO,
            fg=self.COR_TEXTO_ESCURO
        )
        self.feedback_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Barra de progresso
        self.progress_frame = tk.Frame(control_frame, bg=self.COR_FUNDO)
        self.progress_frame.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="Progresso:",
            font=("Helvetica", 12),
            bg=self.COR_FUNDO,
            fg=self.COR_TEXTO_ESCURO
        )
        self.progress_label.pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            orient=tk.HORIZONTAL,
            length=150,
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        
        self.progress_text = tk.Label(
            self.progress_frame,
            text="0/0",
            font=("Helvetica", 12),
            bg=self.COR_FUNDO,
            fg=self.COR_TEXTO_ESCURO
        )
        self.progress_text.pack(side=tk.LEFT)
        
        self.atualizar_progresso()
        self.iniciar_camera()
        self.proxima_letra()
        self.atualizar_tempo()

    def atualizar_tempo(self):
        """Atualiza o timer na interface"""
        if hasattr(self, 'tempo_inicio') and self.tempo_inicio > 0:
            tempo_decorrido = time.time() - self.tempo_inicio
            minutos = int(tempo_decorrido // 60)
            segundos = int(tempo_decorrido % 60)
            self.tempo_label.config(text=f"{minutos:02d}:{segundos:02d}")
        
        self.root.after(1000, self.atualizar_tempo)

    def atualizar_progresso(self):
        """Atualiza a barra de progresso"""
        total = len(self.letras_nivel)
        atual = self.letra_atual_idx
        self.progress_bar['maximum'] = total
        self.progress_bar['value'] = atual
        self.progress_text.config(text=f"{atual}/{total}")

    def proxima_letra(self):
        """Avança para a próxima letra"""
        if self.letra_atual_idx < len(self.letras_nivel):
            self.gesto_alvo = self.letras_nivel[self.letra_atual_idx]
            self.gesto_alvo_label.config(text=self.gesto_alvo)
            self.letra_atual_idx += 1
            self.feedback_label.config(text="Mostre o gesto para a câmera", fg=self.COR_TEXTO_ESCURO)
            self.atualizar_progresso()
        else:
            self.tempo_gasto = time.time() - self.tempo_inicio
            
            if self.nivel_atual not in self.niveis_completos[self.secao_atual]:
                self.niveis_completos[self.secao_atual].append(self.nivel_atual)
            
            if self.nivel_atual == len(self.secoes[self.secao_atual]):
                secoes = list(self.secoes.keys())
                index_atual = secoes.index(self.secao_atual)
                if index_atual < len(secoes) - 1 and self.secao_atual not in self.secoes_liberadas:
                    proxima_secao = secoes[index_atual + 1]
                    self.secoes_liberadas.append(proxima_secao)
            
            self.mostrar_tela_parabens()

    def toggle_camera(self):
        """Liga/desliga a câmera"""
        if self.running:
            self.parar_camera()
            self.btn_camera.config(text="▶️ Iniciar Câmera")
        else:
            self.iniciar_camera()
            self.btn_camera.config(text="⏸️ Parar Câmera")

    def iniciar_camera(self):
        """Inicia a câmera"""
        if self.running:
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível acessar a câmera")
            return
        
        # Configurar resolução máxima suportada
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        self.frame_count = 0
        self.atualizar_frame()

    def atualizar_frame(self):
        """Atualiza o frame da câmera"""
        if self.running:
            self.frame_count += 1
            ret, frame = self.cap.read()
            
            if ret and self.frame_count % (self.skip_frames + 1) == 0:
                frame = self.processar_frame(frame)
                self.mostrar_frame(frame)
            
            self.root.after(30, self.atualizar_frame)

    def processar_frame(self, frame):
        """Processa o frame para detecção de mãos"""
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
            
            # Reconhece o gesto quando o buffer estiver cheio
            if len(self.buffer_gestos) == 30:
                self.reconhecer_gesto()
        else:
            self.frames_sem_maos += 1
            if self.frames_sem_maos > self.RESET_THRESHOLD and self.buffer_gestos:
                self.buffer_gestos.clear()
                self.feedback_label.config(text="Mãos não detectadas", fg=self.COR_ERRO)
        
        return frame_rgb

    def reconhecer_gesto(self):
        """Reconhece o gesto usando o modelo carregado"""
        if not self.modelo_gestos or not self.le_gestos or not self.gesto_alvo:
            # Se não tiver modelo, usa reconhecimento simulado
            self.reconhecer_gesto_simulado()
            return
        
        try:
            # Prepara os dados para o modelo (30 frames, 126 features cada)
            entrada = np.array(self.buffer_gestos).reshape(1, 30, 126)
            
            # Faz a predição
            preds = self.modelo_gestos.predict(entrada, verbose=0)[0]
            classe_idx = np.argmax(preds)
            confianca = preds[classe_idx]
            gesto_reconhecido = self.le_gestos.classes_[classe_idx]
            
            # Atualiza histórico para suavização
            self.historico_predicoes.append(gesto_reconhecido)
            
            # Determina o gesto mais frequente no histórico
            contagem = defaultdict(int)
            for g in self.historico_predicoes:
                contagem[g] += 1
            gesto_final = max(contagem.items(), key=lambda x: x[1])[0]
            
            # Verifica se acertou o gesto alvo
            if confianca > 0.7 and gesto_final == self.gesto_alvo:
                self.pontuacao += 10 * self.nivel_atual
                self.pontuacao_label.config(text=f"{self.pontuacao}")
                self.feedback_label.config(
                    text=f"✅ Correto! {gesto_final} ({confianca:.0%} confiança)",
                    foreground=self.COR_SUCESSO
                )
                self.root.after(1500, self.proxima_letra)
                self.buffer_gestos.clear()
                self.historico_predicoes.clear()
            elif gesto_reconhecido != self.ultimo_gesto_reconhecido:
                self.feedback_label.config(
                    text=f"Reconhecido: {gesto_reconhecido} (Mostre: {self.gesto_alvo})",
                    foreground=self.COR_SECUNDARIA
                )
            
            self.ultimo_gesto_reconhecido = gesto_reconhecido
            
        except Exception as e:
            print(f"Erro ao reconhecer gesto: {str(e)}")
            self.feedback_label.config(
                text="Erro no reconhecimento. Tente novamente",
                foreground=self.COR_ERRO
            )
            self.reconhecer_gesto_simulado()

    def reconhecer_gesto_simulado(self):
        """Simula reconhecimento de gestos (usado quando não há modelo)"""
        if random.random() < 0.3:  # 30% de chance de acerto
            self.pontuacao += 10
            self.pontuacao_label.config(text=f"{self.pontuacao}")
            self.feedback_label.config(text=f"✅ Correto! {self.gesto_alvo}", fg=self.COR_SUCESSO)
            self.root.after(1500, self.proxima_letra)
            self.buffer_gestos.clear()
        else:
            self.feedback_label.config(text=f"Tente novamente: {self.gesto_alvo}", fg=self.COR_ERRO)

    def processar_landmarks(self, results):
        """Processa os landmarks das mãos"""
        landmarks = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        
        landmarks = landmarks[:42]  # Limita a 2 mãos
        if len(landmarks) < 42:
            landmarks.extend([[0, 0, 0]] * (42 - len(landmarks)))
        
        return np.array(landmarks).flatten()

    def mostrar_frame(self, frame):
        """Mostra o frame na interface"""
        # Redimensionar a imagem para caber no espaço disponível
        img = Image.fromarray(frame)
        
        # Obter dimensões do label de vídeo
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        
        # Se o label ainda não tem dimensões definidas, usar um padrão
        if label_width <= 1 or label_height <= 1:
            label_width = 800
            label_height = 600
        
        # Manter aspect ratio
        img_ratio = img.width / img.height
        label_ratio = label_width / label_height
        
        if label_ratio > img_ratio:
            # Ajustar pela altura
            new_height = label_height
            new_width = int(new_height * img_ratio)
        else:
            # Ajustar pela largura
            new_width = label_width
            new_height = int(new_width / img_ratio)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = img
        self.video_label.config(image=img)

    def parar_camera(self):
        """Para a câmera"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = None

    def limpar_tela(self):
        """Limpa todos os widgets"""
        self.parar_camera()
        for widget in self.root.winfo_children():
            widget.destroy()

    def sair(self):
        """Fecha o aplicativo"""
        self.parar_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicativoLibras(root)
    root.protocol("WM_DELETE_WINDOW", app.sair)
    root.mainloop()