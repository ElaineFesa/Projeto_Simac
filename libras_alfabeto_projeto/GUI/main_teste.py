import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import random
import numpy as np
from collections import deque, defaultdict

class AplicativoLibras:
    def __init__(self, root):
        # Configuração do tema
        self.COR_PRIMARIA = "#6A0DAD"
        self.COR_SECUNDARIA = "#FFD700"
        self.COR_TERCIARIA = "#4B0082"
        self.COR_TEXTO_CLARO = "#FFFFFF"
        self.COR_TEXTO_ESCURO = "#333333"
        self.COR_FUNDO = "#F5F5F5"
        self.COR_SUCESSO = "#2E8B57"
        self.COR_ERRO = "#DC143C"

        self.root = root
        self.root.title("Aprenda Libras - Jogo Educacional")
        self.root.geometry("1200x800")
        self.root.configure(bg=self.COR_FUNDO)
        
        # Configuração de estilos
        self.configurar_estilos()
        
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
        self.current_frame = None
        self.nivel_atual = 1
        self.pontuacao = 0
        self.gesto_alvo = None
        self.buffer_gestos = deque(maxlen=30)
        self.historico_predicoes = deque(maxlen=15)
        self.frames_sem_maos = 0
        self.RESET_THRESHOLD = 10
        self.ultimo_gesto_reconhecido = None
        
        # Modelo simplificado para exemplo
        self.modelo_letras = self.carregar_modelo_simulado()

        # Gestos por nível (modo jogo)
        self.gestos_por_nivel = {
            1: ["A", "B", "C", "D", "E"],
            2: ["F", "G", "H", "I", "J"],
            3: ["K", "L", "M", "N", "O"],
            4: ["P", "Q", "R", "S", "T"],
            5: ["U", "V", "W", "X", "Y", "Z"]
        }

        # Interface
        self.criar_menu()
        self.criar_barra_status()
        self.criar_area_principal()
        self.criar_painel_controle()

    def carregar_modelo_simulado(self):
        """Modelo simulado para demonstração"""
        class ModeloSimulado:
            def predict(self, X):
                # Simula reconhecimento aleatório (substitua por seu modelo real)
                return [random.choice(["A", "B", "C", "D", "E"])]
        return ModeloSimulado()

    def configurar_estilos(self):
        """Configura todos os estilos visuais"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=self.COR_FUNDO)
        style.configure('TLabel', background=self.COR_FUNDO, foreground=self.COR_TEXTO_ESCURO, font=('Helvetica', 10))
        style.configure('TButton', background=self.COR_PRIMARIA, foreground=self.COR_TEXTO_CLARO, 
                       font=('Helvetica', 10, 'bold'), padding=8, bordercolor=self.COR_PRIMARIA)
        style.map('TButton', background=[('active', self.COR_TERCIARIA), ('disabled', '#CCCCCC')],
                foreground=[('active', self.COR_SECUNDARIA), ('disabled', '#666666')])
        style.configure('TLabelframe', background=self.COR_FUNDO, bordercolor=self.COR_PRIMARIA)
        style.configure('TLabelframe.Label', background=self.COR_FUNDO, foreground=self.COR_PRIMARIA,
                      font=('Helvetica', 10, 'bold'))
        style.configure('Status.TLabel', background=self.COR_PRIMARIA, foreground=self.COR_TEXTO_CLARO,
                      font=('Helvetica', 10), padding=5)

    def criar_menu(self):
        menubar = tk.Menu(self.root, bg=self.COR_FUNDO, fg=self.COR_TEXTO_ESCURO,
                         activebackground=self.COR_PRIMARIA, activeforeground=self.COR_SECUNDARIA)
        
        # Menu Níveis
        menu_niveis = tk.Menu(menubar, tearoff=0)
        for i in range(1, 6):
            menu_niveis.add_command(label=f"Nível {i}", command=lambda n=i: self.iniciar_nivel(n))
        menubar.add_cascade(label="Níveis", menu=menu_niveis)
        
        # Menu Ajuda
        menu_ajuda = tk.Menu(menubar, tearoff=0)
        menu_ajuda.add_command(label="Como Jogar", command=self.mostrar_ajuda)
        menu_ajuda.add_command(label="Sobre", command=self.mostrar_sobre)
        menubar.add_cascade(label="Ajuda", menu=menu_ajuda)
        
        self.root.config(menu=menubar)

    def criar_barra_status(self):
        self.status_var = tk.StringVar()
        self.status_var.set(f"Pronto | Nível: {self.nivel_atual} | Pontos: {self.pontuacao}")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, style='Status.TLabel')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def criar_area_principal(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Painel do gesto alvo
        left_frame = ttk.LabelFrame(main_frame, text="Gesto Alvo")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.gesto_alvo_label = ttk.Label(left_frame, text="Selecione um nível para começar",
                                        font=('Helvetica', 24, 'bold'), foreground=self.COR_PRIMARIA)
        self.gesto_alvo_label.pack(expand=True, fill=tk.BOTH)
        
        # Painel de feedback
        self.feedback_frame = ttk.LabelFrame(left_frame, text="Feedback")
        self.feedback_frame.pack(fill=tk.X, pady=10)
        self.feedback_label = ttk.Label(self.feedback_frame, text="", font=('Helvetica', 14))
        self.feedback_label.pack(pady=10)
        
        # Painel da câmera (otimizado)
        right_frame = ttk.LabelFrame(main_frame, text="Sua Câmera")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = ttk.Label(right_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def criar_painel_controle(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_camera = ttk.Button(control_frame, text="Iniciar Câmera", command=self.toggle_camera)
        self.btn_camera.pack(side=tk.LEFT, padx=5)
        
        self.btn_proximo = ttk.Button(control_frame, text="Próximo Nível", command=self.proximo_nivel, state=tk.DISABLED)
        self.btn_proximo.pack(side=tk.LEFT, padx=5)

    def toggle_camera(self):
        """Liga/desliga a câmera sem threading"""
        if not self.running:
            self.iniciar_camera()
        else:
            self.parar_camera()

    def iniciar_camera(self):
        """Inicia a captura de vídeo usando after()"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível acessar a câmera")
            return
        
        # Configurações para melhor performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        self.btn_camera.config(text="Parar Câmera")
        self.atualizar_frame()  # Inicia o loop principal

    def atualizar_frame(self):
        """Loop principal de atualização de frames"""
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = self.processar_frame(frame)
                self.mostrar_frame(frame)
            
            # Agenda a próxima atualização (30ms = ~33 FPS)
            self.root.after(30, self.atualizar_frame)

    def processar_frame(self, frame):
        """Processa o frame e detecta gestos"""
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
            
            # Processamento para reconhecimento
            landmarks = self.processar_landmarks(results)
            self.buffer_gestos.append(landmarks)
            self.reconhecer_gesto_automatico()
        else:
            self.frames_sem_maos += 1
            if self.frames_sem_maos > self.RESET_THRESHOLD and self.buffer_gestos:
                self.buffer_gestos.clear()
                self.feedback_label.config(text="Mãos não detectadas", foreground=self.COR_ERRO)
        
        return frame_rgb

    def mostrar_frame(self, frame):
        """Exibe o frame na interface"""
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = img  # Mantém referência
        self.video_label.config(image=img)

    def parar_camera(self):
        """Para a captura de vídeo corretamente"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = None
        self.btn_camera.config(text="Iniciar Câmera")
        self.video_label.config(image='')

    def processar_landmarks(self, results):
        """Processa os landmarks para reconhecimento"""
        landmarks = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        
        landmarks = landmarks[:42]
        if len(landmarks) < 42:
            landmarks.extend([[0,0,0]] * (42 - len(landmarks)))
        
        return np.array(landmarks).flatten()

    def reconhecer_gesto_automatico(self):
        """Reconhece o gesto automaticamente quando o buffer estiver cheio"""
        if len(self.buffer_gestos) == 30 and self.gesto_alvo:
            self.reconhecer_letra()

    def reconhecer_letra(self):
        """Reconhecimento para letras"""
        try:
            landmarks = np.array(self.buffer_gestos[-1])[:63]
            pred = self.modelo_letras.predict(landmarks.reshape(1, -1))[0]
            
            if pred == self.gesto_alvo:
                self.pontuacao += 10 * self.nivel_atual
                self.feedback_label.config(text=f"✅ Correto! {pred}", foreground=self.COR_SUCESSO)
                self.btn_proximo.config(state=tk.NORMAL)
                self.buffer_gestos.clear()
            elif pred != self.ultimo_gesto_reconhecido:
                self.feedback_label.config(text=f"Reconhecido: {pred} (Mostre: {self.gesto_alvo})", foreground=self.COR_SECUNDARIA)
            
            self.ultimo_gesto_reconhecido = pred
            self.atualizar_status()
        except Exception as e:
            print(f"Erro ao reconhecer letra: {str(e)}")

    def iniciar_nivel(self, nivel):
        """Inicia um novo nível do jogo"""
        self.nivel_atual = nivel
        self.gesto_alvo = random.choice(self.gestos_por_nivel[nivel])
        self.gesto_alvo_label.config(text=self.gesto_alvo)
        self.feedback_label.config(text="Mostre o gesto para a câmera", foreground=self.COR_TEXTO_ESCURO)
        self.btn_proximo.config(state=tk.DISABLED)
        self.atualizar_status(f"Nível {nivel} | Gesto: {self.gesto_alvo}")

    def proximo_nivel(self):
        """Avança para o próximo nível"""
        if self.nivel_atual < 5:
            self.nivel_atual += 1
            self.iniciar_nivel(self.nivel_atual)
        else:
            messagebox.showinfo("Parabéns!", f"Você completou todos os níveis!\nPontuação final: {self.pontuacao}")
            self.btn_proximo.config(state=tk.DISABLED)

    def mostrar_ajuda(self):
        """Mostra instruções do jogo"""
        ajuda_texto = """
        COMO JOGAR:
        
        1. Selecione um nível no menu
        2. Clique em 'Iniciar Câmera'
        3. Mostre o gesto correspondente à letra exibida
        4. Acertando, você avança para a próxima letra
        5. Complete todas as letras para passar de nível
        
        DICAS:
        - Mantenha as mãos visíveis na câmera
        - Faça os gestos claramente
        - Acerte para ganhar mais pontos!
        """
        messagebox.showinfo("Como Jogar", ajuda_texto)

    def mostrar_sobre(self):
        """Mostra informações sobre o aplicativo"""
        sobre_texto = """
        Aprenda Libras - Jogo Educacional
        
        Versão 2.0
        Jogo interativo para aprendizado de Libras
        
        Desenvolvido para:
        - Aprendizado de sinais do alfabeto em LIBRAS
        - Diversão e educação
        - Melhoria da acessibilidade
        
        © 2023 Projeto de Acessibilidade
        """
        messagebox.showinfo("Sobre", sobre_texto)

    def atualizar_status(self, mensagem=None):
        """Atualiza a barra de status"""
        self.status_var.set(mensagem or f"Nível {self.nivel_atual} | Pontos: {self.pontuacao} | Gesto: {self.gesto_alvo}")

    def sair(self):
        """Encerra o aplicativo corretamente"""
        self.parar_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicativoLibras(root)
    root.protocol("WM_DELETE_WINDOW", app.sair)
    root.mainloop()