import numpy as np
from collections import deque

class DetectorSonolenciaInteligente:
    def __init__(self):
        self.baseline_piscada = None
        self.historico_piscadas = deque(maxlen=50)  # Últimas 50 piscadas
        self.estado_olhos = "aberto"
        self.tempo_inicio_fechamento = 0
        self.alertas_ativos = False
        self.mensagem_alerta = ""
        
        # Thresholds adaptativos
        self.margem_seguranca = 0.05  # 50ms
        self.fator_alerta = 1.5  # 50% mais longo que baseline
        
    def detectar_piscada(self, ear, timestamp):
        """Detecta piscadas e ajusta baseline automaticamente"""
        if ear < 0.20:  # Olhos fechados
            if self.estado_olhos == "aberto":
                self.estado_olhos = "fechado"
                self.tempo_inicio_fechamento = timestamp
                
        else:  # Olhos abertos
            if self.estado_olhos == "fechado":
                self.estado_olhos = "aberto"
                duracao_piscada = timestamp - self.tempo_inicio_fechamento
                if 0.1 <= duracao_piscada <= 0.4:
                    self.historico_piscadas.append(duracao_piscada)
                    if len(self.historico_piscadas) >= 10:
                        self.atualizar_baseline()
                    self.verificar_alerta_sonolencia(duracao_piscada)
    
    def atualizar_baseline(self):
        nova_baseline = np.median(list(self.historico_piscadas)[-10:])
        if self.baseline_piscada is None:
            self.baseline_piscada = nova_baseline
        else:
            self.baseline_piscada = 0.8 * self.baseline_piscada + 0.2 * nova_baseline
    
    def verificar_alerta_sonolencia(self, duracao_atual):
        if self.baseline_piscada is None:
            return
        # 1. Piscada individual muito longa
        if duracao_atual > (self.baseline_piscada * 2.0):
            self.ativar_alerta("ALERTA: Piscada anormalmente longa!")
        # 2. Tendência de piscadas mais longas (últimas 5 piscadas)
        ultimas_piscadas = list(self.historico_piscadas)[-5:]
        if len(ultimas_piscadas) >= 3:
            media_recente = np.mean(ultimas_piscadas)
            if media_recente > (self.baseline_piscada * self.fator_alerta):
                self.ativar_alerta("ALERTA: Piscadas ficando mais lentas!")
    
    def ativar_alerta(self, mensagem):
        self.alertas_ativos = True
        self.mensagem_alerta = mensagem

    def resetar_alerta(self):
        self.alertas_ativos = False
        self.mensagem_alerta = ""
    