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

class ReconhecedorGestos:
    def __init__(self):
        # Configurações
        self.MODEL_PATH = Path('modelos/modelo_gestos.h5')
        self.LABEL_PATH = Path('modelos/rotulador_gestos.pkl')
        self.SEQUENCE_LENGTH = 30
        self.MIN_CONFIDENCE = 0.7
        self.RESET_THRESHOLD = 10

        # Carregar modelo
        self.model = load_model(self.MODEL_PATH)
        self.le = joblib.load(self.LABEL_PATH)
        
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Variáveis de estado
        self.buffer = deque(maxlen=self.SEQUENCE_LENGTH)
        self.frames_sem_maos = 0
        self.historico_predicoes = deque(maxlen=15)
        self.ultimo_gesto = None
        self.cap = None
        self.running = False

    def processar_landmarks(self, results):
        """Processa landmarks e retorna array padronizado"""
        landmarks = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        
        # Padroniza para 2 mãos (42 landmarks)
        landmarks = landmarks[:42]
        if len(landmarks) < 42:
            landmarks.extend([[0, 0, 0]] * (42 - len(landmarks)))
        
        return np.array(landmarks).flatten()

    def iniciar_captura(self):
        """Inicia a captura da câmera em uma janela separada"""
        self.cap = cv2.VideoCapture(0)
        self.running = True
        print("Captura iniciada. Pressione ESC para sair.")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Lógica de reconhecimento
            if not results.multi_hand_landmarks:
                self.frames_sem_maos += 1
                if self.frames_sem_maos > self.RESET_THRESHOLD and self.buffer:
                    self.buffer.clear()
                    print("▶️ Buffer resetado (mãos não detectadas)")
            else:
                self.frames_sem_maos = 0
                landmarks = self.processar_landmarks(results)
                self.buffer.append(landmarks)

                # Desenha landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                    )

            # Reconhecimento quando buffer cheio
            if len(self.buffer) == self.SEQUENCE_LENGTH:
                self.reconhecer_gesto()

            cv2.imshow("Reconhecimento de Gestos", frame)
            if cv2.waitKey(1) == 27:  # ESC para sair
                self.parar_captura()

    def reconhecer_gesto(self):
        """Reconhece o gesto e atualiza ultimo_gesto"""
        entrada = np.array(self.buffer).reshape(1, self.SEQUENCE_LENGTH, 126)
        preds = self.model.predict(entrada, verbose=0)[0]
        classe_idx = np.argmax(preds)
        confianca = preds[classe_idx]

        if confianca >= self.MIN_CONFIDENCE:
            gesto_atual = self.le.classes_[classe_idx]
            self.historico_predicoes.append(gesto_atual)

            # Suavização por votação majoritária
            contagem = defaultdict(int)
            for g in self.historico_predicoes:
                contagem[g] += 1
            gesto_final = max(contagem.items(), key=lambda x: x[1])[0]

            if gesto_final != self.ultimo_gesto:
                self.ultimo_gesto = gesto_final
                print(f"Gesto reconhecido: {gesto_final} ({confianca:.0%})")
                self.buffer.clear()
                return gesto_final
        return None

    def parar_captura(self):
        """Para a captura da câmera"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

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

        # Inicializar reconhecedor de gestos
        self.reconhecedor = ReconhecedorGestos()
        self.reconhecedor_thread = None

        # Estado do aplicativo
        self.nivel_atual = 1
        self.pontuacao = 0
        self.gesto_alvo = None
        self.frames_sem_maos = 0
        self.niveis_completos = {}
        self.secoes_liberadas = ["Alfabeto"]
        self.tempo_inicio = 0
        self.tempo_gasto = 0

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

    def iniciar_reconhecimento(self):
        """Inicia o reconhecimento em uma thread separada"""
        if not self.reconhecedor_thread or not self.reconhecedor_thread.is_alive():
            self.reconhecedor_thread = threading.Thread(
                target=self.reconhecedor.iniciar_captura,
                daemon=True
            )
            self.reconhecedor_thread.start()
            messagebox.showinfo("Info", "Janela de reconhecimento iniciada!")

    def parar_reconhecimento(self):
        """Para o reconhecimento"""
        self.reconhecedor.parar_captura()
        if self.reconhecedor_thread:
            self.reconhecedor_thread.join()

    def verificar_gesto(self):
        """Verifica periodicamente se um gesto foi reconhecido"""
        gesto = self.reconhecedor.ultimo_gesto
        if gesto and gesto == self.gesto_alvo:
            self.pontuacao += 10 * self.nivel_atual
            self.pontuacao_label.config(text=f"{self.pontuacao}")
            self.feedback_label.config(
                text=f"✅ Correto! {gesto}",
                foreground=self.COR_SUCESSO
            )
            self.root.after(1500, self.proxima_letra)
            self.reconhecedor.ultimo_gesto = None  # Resetar para evitar repetição
        self.root.after(1000, self.verificar_gesto)  # Verifica a cada 1 segundo

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
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Configurar grid (3 linhas: cabeçalho, conteúdo, controles)
        main_frame.rowconfigure(1, weight=1)  # Linha do conteúdo principal
        main_frame.columnconfigure(0, weight=1)
        
        # Barra superior - cabeçalho do nível
        header_frame = tk.Frame(
            main_frame,
            bg=self.COR_PRIMARIA,
            height=60,
            bd=0,
            highlightthickness=0,
            relief=tk.FLAT
        )
        header_frame.grid(row=0, column=0, sticky="ew", columnspan=2, pady=(0, 20))
        header_frame.grid_propagate(False)  # Mantém a altura fixa
        
        # Conteúdo do cabeçalho
        inner_header = tk.Frame(header_frame, bg=self.COR_PRIMARIA)
        inner_header.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # Botão voltar
        btn_voltar = tk.Button(
            inner_header,
            text="←",
            font=("Helvetica", 16, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO,
            bd=0,
            activebackground=self.COR_PRIMARIA,
            activeforeground=self.COR_SECUNDARIA,
            command=self.mostrar_tela_secoes
        )
        btn_voltar.pack(side=tk.LEFT, padx=(0, 15))
        
        # Título do nível
        lbl_titulo = tk.Label(
            inner_header,
            text=f"{secao.upper()} - NÍVEL {nivel}",
            font=("Helvetica", 18, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO
        )
        lbl_titulo.pack(side=tk.LEFT, expand=True)
        
        # Informações (tempo e pontuação)
        info_frame = tk.Frame(inner_header, bg=self.COR_PRIMARIA)
        info_frame.pack(side=tk.RIGHT)
        
        # Timer
        timer_frame = tk.Frame(info_frame, bg=self.COR_PRIMARIA)
        timer_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(
            timer_frame,
            text="⏱️",
            font=("Helvetica", 14),
            bg=self.COR_PRIMARIA,
            fg=self.COR_SECUNDARIA
        ).pack(side=tk.LEFT)
        
        self.tempo_label = tk.Label(
            timer_frame,
            text="00:00",
            font=("Helvetica", 14, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO
        )
        self.tempo_label.pack(side=tk.LEFT)
        
        # Pontuação
        pontos_frame = tk.Frame(info_frame, bg=self.COR_PRIMARIA)
        pontos_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(
            pontos_frame,
            text="🏆",
            font=("Helvetica", 14),
            bg=self.COR_PRIMARIA,
            fg=self.COR_SECUNDARIA
        ).pack(side=tk.LEFT)
        
        self.pontuacao_label = tk.Label(
            pontos_frame,
            text=f"{self.pontuacao}",
            font=("Helvetica", 14, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO
        )
        self.pontuacao_label.pack(side=tk.LEFT)
        
        # Frame do conteúdo principal (gesto alvo + espaço para câmera)
        content_frame = tk.Frame(main_frame, bg=self.COR_FUNDO)
        content_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 20))
        
        # Configurar grid para conteúdo (1 linha, 2 colunas com pesos iguais)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Frame do gesto alvo - card estilizado
        left_frame = tk.Frame(
            content_frame,
            bg=self.COR_CARD,
            bd=0,
            highlightbackground=self.COR_BORDA,
            highlightthickness=1,
            relief=tk.RAISED
        )
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        
        # Configurar expansão do frame do gesto alvo
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        
        # Cabeçalho do card
        tk.Label(
            left_frame,
            text="GESTO ALVO",
            font=("Helvetica", 14, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO,
            pady=10
        ).grid(row=0, column=0, sticky="ew")
        
        # Área do gesto (centralizada)
        gesto_container = tk.Frame(left_frame, bg=self.COR_CARD)
        gesto_container.grid(row=1, column=0, sticky="nsew")
        gesto_container.columnconfigure(0, weight=1)
        gesto_container.rowconfigure(0, weight=1)
        
        self.gesto_alvo_label = tk.Label(
            gesto_container,
            text="",
            font=("Helvetica", 72, "bold"),
            bg=self.COR_CARD,
            fg=self.COR_PRIMARIA,
            pady=20
        )
        self.gesto_alvo_label.grid(row=0, column=0)
        
        # Frame da câmera - card estilizado
        right_frame = tk.Frame(
            content_frame,
            bg=self.COR_CARD,
            bd=0,
            highlightbackground=self.COR_BORDA,
            highlightthickness=1,
            relief=tk.RAISED
        )
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        
        # Configurar expansão do frame da câmera
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # Cabeçalho do card
        tk.Label(
            right_frame,
            text="RECONHECIMENTO",
            font=("Helvetica", 14, "bold"),
            bg=self.COR_PRIMARIA,
            fg=self.COR_TEXTO_CLARO,
            pady=10
        ).grid(row=0, column=0, sticky="ew")
        
        # Área de conteúdo (centralizada)
        camera_container = tk.Frame(right_frame, bg=self.COR_CARD)
        camera_container.grid(row=1, column=0, sticky="nsew")
        camera_container.columnconfigure(0, weight=1)
        camera_container.rowconfigure(0, weight=1)
        
        # Botão para iniciar reconhecimento (estilizado)
        btn_style = {
            "font": ("Helvetica", 16, "bold"),
            "bg": self.COR_PRIMARIA,
            "fg": self.COR_TEXTO_CLARO,
            "activebackground": self.COR_SECUNDARIA,
            "activeforeground": self.COR_TEXTO_ESCURO,
            "bd": 0,
            "padx": 30,
            "pady": 15,
            "relief": tk.FLAT,
            "highlightthickness": 0
        }
        
        btn_iniciar_reconhecimento = tk.Button(
            camera_container,
            text="INICIAR RECONHECIMENTO",
            **btn_style,
            command=self.iniciar_reconhecimento
        )
        btn_iniciar_reconhecimento.grid(row=0, column=0)
        
        # Adicionar ícone ao botão
        btn_iniciar_reconhecimento.config(compound=tk.LEFT, image='')  # Pode adicionar um ícone aqui
        
        # Controles inferiores (feedback e progresso)
        footer_frame = tk.Frame(main_frame, bg=self.COR_FUNDO)
        footer_frame.grid(row=2, column=0, sticky="ew")
        
        # Configurar grid para controles (2 colunas)
        footer_frame.columnconfigure(0, weight=3)  # Feedback
        footer_frame.columnconfigure(1, weight=1)  # Progresso
        
        # Feedback (mensagens para o usuário)
        feedback_container = tk.Frame(footer_frame, bg=self.COR_FUNDO)
        feedback_container.grid(row=0, column=0, sticky="w")
        
        self.feedback_label = tk.Label(
            feedback_container,
            text="Mostre o gesto alvo para a câmera",
            font=("Helvetica", 14),
            bg=self.COR_FUNDO,
            fg=self.COR_TEXTO_ESCURO,
            pady=10
        )
        self.feedback_label.pack(anchor="w")
        
        # Barra de progresso (estilizada)
        progress_container = tk.Frame(footer_frame, bg=self.COR_FUNDO)
        progress_container.grid(row=0, column=1, sticky="e")
        
        progress_inner = tk.Frame(progress_container, bg=self.COR_FUNDO)
        progress_inner.pack()
        
        self.progress_label = tk.Label(
            progress_inner,
            text="Progresso:",
            font=("Helvetica", 12),
            bg=self.COR_FUNDO,
            fg=self.COR_TEXTO_ESCURO
        )
        self.progress_label.pack(side=tk.LEFT)
        
        # Barra de progresso personalizada
        style = ttk.Style()
        style.configure("Nivel.Horizontal.TProgressbar", 
                    troughcolor=self.COR_BORDA,
                    background=self.COR_PRIMARIA,
                    bordercolor=self.COR_BORDA,
                    lightcolor=self.COR_PRIMARIA,
                    darkcolor=self.COR_PRIMARIA)
        
        self.progress_bar = ttk.Progressbar(
            progress_inner,
            orient=tk.HORIZONTAL,
            length=150,
            mode='determinate',
            style="Nivel.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        
        self.progress_text = tk.Label(
            progress_inner,
            text="0/0",
            font=("Helvetica", 12),
            bg=self.COR_FUNDO,
            fg=self.COR_TEXTO_ESCURO
        )
        self.progress_text.pack(side=tk.LEFT)
        
        # Inicializar componentes
        self.atualizar_progresso()
        self.proxima_letra()
        self.atualizar_tempo()
        self.verificar_gesto()  # Inicia a verificação de gestos

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

    def limpar_tela(self):
        """Limpa todos os widgets"""
        self.parar_reconhecimento()
        for widget in self.root.winfo_children():
            widget.destroy()

    def sair(self):
        """Fecha o aplicativo"""
        self.parar_reconhecimento()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicativoLibras(root)
    root.protocol("WM_DELETE_WINDOW", app.sair)
    root.mainloop()