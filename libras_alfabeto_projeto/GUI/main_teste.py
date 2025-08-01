import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import os
from pathlib import Path
import sys

class AplicativoLibras:
    def __init__(self, root):
        self.root = root
        self.root.title("Aprenda Libras - Níveis Educacionais")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f8ff')
        
        # Variáveis de estado
        self.cap = None
        self.video_thread = None
        self.running = False
        self.nivel_atual = 1
        self.pontuacao = 0
        self.desafio_ativo = False
        self.gesto_alvo = None
        
        # Configurações de caminhos
        self.dados_dir = Path('dados')
        self.modelos_dir = Path('modelos')
        self.dados_dir.mkdir(exist_ok=True)
        self.modelos_dir.mkdir(exist_ok=True)
        
        # Carregar modelos (simulado - na prática você carregaria seus modelos reais)
        self.modelo_letras = None
        self.modelo_gestos = None
        self.carregar_modelos()
        
        # Interface
        self.criar_menu()
        self.criar_barra_status()
        self.criar_area_principal()
        self.criar_painel_controle()
        
    def carregar_modelos(self):
        """Simula o carregamento dos modelos (substitua com sua implementação real)"""
        modelos = {
            'letras': self.modelos_dir / 'modelo_letras_libras.pkl',
            'gestos': self.modelos_dir / 'modelo_gestos.h5'
        }
        
        for nome, caminho in modelos.items():
            if caminho.exists():
                print(f"Modelo {nome} carregado")
            else:
                print(f"Aviso: Modelo {nome} não encontrado em {caminho}")
    
    def criar_menu(self):
        menubar = tk.Menu(self.root)
        
        # Menu Arquivo
        menu_arquivo = tk.Menu(menubar, tearoff=0)
        menu_arquivo.add_command(label="Sair", command=self.sair)
        menubar.add_cascade(label="Arquivo", menu=menu_arquivo)
        
        # Menu Níveis
        menu_niveis = tk.Menu(menubar, tearoff=0)
        for i in range(1, 6):
            menu_niveis.add_command(
                label=f"Nível {i}", 
                command=lambda n=i: self.definir_nivel(n)
            )
        menubar.add_cascade(label="Níveis", menu=menu_niveis)
        
        # Menu Ajuda
        menu_ajuda = tk.Menu(menubar, tearoff=0)
        menu_ajuda.add_command(label="Sobre", command=self.mostrar_sobre)
        menubar.add_cascade(label="Ajuda", menu=menu_ajuda)
        
        self.root.config(menu=menubar)
    
    def criar_barra_status(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Pronto | Nível 1 | Pontuação: 0")
        
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
        
        # Painel de vídeo
        self.video_frame = ttk.LabelFrame(main_frame, text="Visualização")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Painel de informações
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Nível atual
        nivel_frame = ttk.LabelFrame(info_frame, text="Nível Atual")
        nivel_frame.pack(fill=tk.X, pady=5)
        
        self.nivel_label = ttk.Label(
            nivel_frame, 
            text=f"Nível {self.nivel_atual}", 
            font=('Helvetica', 14, 'bold')
        )
        self.nivel_label.pack(pady=10)
        
        # Progresso
        progresso_frame = ttk.LabelFrame(info_frame, text="Progresso")
        progresso_frame.pack(fill=tk.X, pady=5)
        
        self.progresso_var = tk.DoubleVar()
        self.progresso_bar = ttk.Progressbar(
            progresso_frame, 
            variable=self.progresso_var,
            maximum=100
        )
        self.progresso_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.pontuacao_label = ttk.Label(
            progresso_frame, 
            text=f"Pontuação: {self.pontuacao}",
            font=('Helvetica', 12)
        )
        self.pontuacao_label.pack(pady=5)
        
        # Instruções
        instrucoes_frame = ttk.LabelFrame(info_frame, text="Instruções")
        instrucoes_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.instrucoes_text = tk.Text(
            instrucoes_frame, 
            wrap=tk.WORD, 
            height=10,
            state=tk.DISABLED
        )
        scroll = ttk.Scrollbar(instrucoes_frame, command=self.instrucoes_text.yview)
        self.instrucoes_text.config(yscrollcommand=scroll.set)
        
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.instrucoes_text.pack(fill=tk.BOTH, expand=True)
        
        self.atualizar_instrucoes()
    
    def criar_painel_controle(self):
        controle_frame = ttk.Frame(self.root)
        controle_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Botões de controle
        self.btn_camera = ttk.Button(
            controle_frame, 
            text="Iniciar Câmera", 
            command=self.toggle_camera
        )
        self.btn_camera.pack(side=tk.LEFT, padx=5)
        
        self.btn_desafio = ttk.Button(
            controle_frame, 
            text="Iniciar Desafio", 
            command=self.iniciar_desafio,
            state=tk.DISABLED
        )
        self.btn_desafio.pack(side=tk.LEFT, padx=5)
        
        self.btn_reconhecer = ttk.Button(
            controle_frame, 
            text="Reconhecer Gestos", 
            command=self.reconhecer_gestos,
            state=tk.DISABLED
        )
        self.btn_reconhecer.pack(side=tk.LEFT, padx=5)
        
        self.btn_coletar = ttk.Button(
            controle_frame, 
            text="Coletar Dados", 
            command=self.coletar_dados
        )
        self.btn_coletar.pack(side=tk.LEFT, padx=5)
        
        self.btn_treinar = ttk.Button(
            controle_frame, 
            text="Treinar Modelo", 
            command=self.treinar_modelo
        )
        self.btn_treinar.pack(side=tk.LEFT, padx=5)
    
    def atualizar_instrucoes(self):
        """Atualiza as instruções com base no nível atual"""
        instrucoes = {
            1: "Nível 1: Alfabeto em Libras\n\nMostre as letras A, B e C com as mãos.\n\nDica: Comece com a letra A, formando um punho com a mão direita.",
            2: "Nível 2: Palavras Básicas\n\nMostre os gestos para 'Olá', 'Obrigado' e 'Ajuda'.",
            3: "Nível 3: Frases Simples\n\nCombine gestos para formar 'Meu nome é...'",
            4: "Nível 4: Vocabulário Avançado\n\nMostre gestos para objetos comuns como 'casa', 'escola' e 'família'.",
            5: "Nível 5: Conversação\n\nPratique diálogos curtos usando múltiplos gestos em sequência."
        }
        
        texto = instrucoes.get(self.nivel_atual, "Nível não configurado")
        self.instrucoes_text.config(state=tk.NORMAL)
        self.instrucoes_text.delete(1.0, tk.END)
        self.instrucoes_text.insert(tk.END, texto)
        self.instrucoes_text.config(state=tk.DISABLED)
    
    def definir_nivel(self, nivel):
        """Define o nível atual do aplicativo"""
        self.nivel_atual = nivel
        self.nivel_label.config(text=f"Nível {self.nivel_atual}")
        self.atualizar_instrucoes()
        self.atualizar_status()
    
    def toggle_camera(self):
        """Liga/desliga a câmera"""
        if self.cap is None:
            self.iniciar_camera()
        else:
            self.parar_camera()
    
    def iniciar_camera(self):
        """Inicia a captura de vídeo da câmera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível acessar a câmera")
            return
        
        self.running = True
        self.btn_camera.config(text="Parar Câmera")
        self.btn_desafio.config(state=tk.NORMAL)
        self.btn_reconhecer.config(state=tk.NORMAL)
        
        # Inicia thread para atualizar o vídeo
        self.video_thread = threading.Thread(target=self.atualizar_video, daemon=True)
        self.video_thread.start()
    
    def parar_camera(self):
        """Para a captura de vídeo"""
        self.running = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Limpa o frame de vídeo
        self.video_label.config(image='')
        self.btn_camera.config(text="Iniciar Câmera")
        self.btn_desafio.config(state=tk.DISABLED)
        self.btn_reconhecer.config(state=tk.DISABLED)
    
    def atualizar_video(self):
        """Atualiza o frame de vídeo em tempo real"""
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Processamento básico do frame (pode ser expandido)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1)
                
                # Redimensiona para caber no painel
                img = Image.fromarray(frame)
                img.thumbnail((800, 600))
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Atualiza a imagem no label
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            
            # Taxa de atualização (~30fps)
            self.root.update_idletasks()
    
    def iniciar_desafio(self):
        """Inicia um desafio do nível atual"""
        if self.desafio_ativo:
            self.parar_desafio()
            return
        
        # Define o gesto alvo com base no nível
        desafios = {
            1: ["A", "B", "C"],
            2: ["OLÁ", "OBRIGADO", "AJUDA"],
            3: ["MEU", "NOME", "É"],
            4: ["CASA", "ESCOLA", "FAMÍLIA"],
            5: ["OI", "COMO", "VOCÊ", "ESTÁ"]
        }
        
        self.gesto_alvo = desafios.get(self.nivel_atual, ["A"])[0]
        self.desafio_ativo = True
        self.btn_desafio.config(text="Parar Desafio")
        
        # Mostra instrução do desafio
        self.instrucoes_text.config(state=tk.NORMAL)
        self.instrucoes_text.delete(1.0, tk.END)
        self.instrucoes_text.insert(tk.END, f"Desafio: Mostre o gesto para '{self.gesto_alvo}'")
        self.instrucoes_text.config(state=tk.DISABLED)
        
        # Aqui você implementaria a lógica de reconhecimento específica para o desafio
        messagebox.showinfo("Desafio Iniciado", f"Mostre o gesto para: {self.gesto_alvo}")
    
    def parar_desafio(self):
        """Para o desafio atual"""
        self.desafio_ativo = False
        self.gesto_alvo = None
        self.btn_desafio.config(text="Iniciar Desafio")
        self.atualizar_instrucoes()
    
    def reconhecer_gestos(self):
        """Inicia o reconhecimento contínuo de gestos"""
        # Implementação simplificada - na prática você chamaria seu módulo de reconhecimento
        messagebox.showinfo("Reconhecimento", "Reconhecimento de gestos ativado. Mostre um gesto para a câmera.")
    
    def coletar_dados(self):
        """Abre a janela para coleta de dados de gestos"""
        # Implementação simplificada - na prática você chamaria seu módulo de coleta
        escolha = messagebox.askquestion("Coletar Dados", "Deseja coletar dados de letras ou gestos?")
        
        if escolha == 'yes':
            messagebox.showinfo("Coletar Letras", "Modo de coleta de letras ativado. Pressione a tecla correspondente à letra mostrada.")
        else:
            messagebox.showinfo("Coletar Gestos", "Modo de coleta de gestos ativado. Digite o nome do gesto e mostre-o para a câmera.")
    
    def treinar_modelo(self):
        """Inicia o treinamento do modelo"""
        # Implementação simplificada - na prática você chamaria seu módulo de treinamento
        resposta = messagebox.askyesno("Treinar Modelo", "Isso pode levar alguns minutos. Deseja continuar?")
        
        if resposta:
            # Simula o treinamento
            self.progresso_var.set(0)
            self.atualizar_status("Treinando modelo...")
            
            def simular_treinamento():
                for i in range(1, 101):
                    self.progresso_var.set(i)
                    self.root.update_idletasks()
                    self.root.after(50)
                
                self.atualizar_status("Modelo treinado com sucesso!")
                messagebox.showinfo("Sucesso", "Modelo treinado com sucesso!")
            
            threading.Thread(target=simular_treinamento, daemon=True).start()
    
    def atualizar_status(self, mensagem=None):
        """Atualiza a barra de status"""
        if mensagem:
            self.status_var.set(mensagem)
        else:
            self.status_var.set(f"Pronto | Nível {self.nivel_atual} | Pontuação: {self.pontuacao}")
    
    def mostrar_sobre(self):
        """Mostra a janela 'Sobre'"""
        sobre = """
        Aprenda Libras - Níveis Educacionais
        
        Versão 1.0
        Desenvolvido para auxiliar no aprendizado de Língua Brasileira de Sinais (Libras).
        
        Recursos:
        - Níveis progressivos de aprendizado
        - Reconhecimento de gestos em tempo real
        - Desafios interativos
        - Ferramentas para coleta e treinamento
        
        © 2023 Projeto Educacional
        """
        messagebox.showinfo("Sobre", sobre)
    
    def sair(self):
        """Encerra o aplicativo"""
        if messagebox.askokcancel("Sair", "Deseja realmente sair do aplicativo?"):
            self.parar_camera()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicativoLibras(root)
    root.protocol("WM_DELETE_WINDOW", app.sair)
    root.mainloop()