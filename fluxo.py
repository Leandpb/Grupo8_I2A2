import time
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

from crew_face import FacadeCrew
from agentes_cabecalho import AnaliseCabecalhoCrew
from agentes_itens import AnaliseItensCrew
from orquestrador import AnaliseAmbos


class State(BaseModel):
    tipo_consulta: str = ""
    text: str = ""


class FluxoFiscal(Flow[State]):

    @start()
    def start(self):
        start_time = time.time()

        avaliador = FacadeCrew()
        self.state.tipo_consulta = avaliador.executar(self.state.text).strip().lower()

        end_time = time.time()
        print(f"🚦 Classificação: {self.state.tipo_consulta}")
        print(f"⏱️ Tempo de execução: {end_time - start_time:.4f} segundos")

    @router(start)
    def roteamento(self):
        return self.state.tipo_consulta

    @listen("cabecalho")
    def executar_cabecalho(self):
        crew = AnaliseCabecalhoCrew()
        resposta = crew.executar(self.state.text)
        return resposta

    @listen("itens")
    def executar_itens(self):
        crew = AnaliseItensCrew()
        resposta = crew.executar(self.state.text)
        return resposta
    
    @listen("ambos")
    def executar_ambos(self):
        crew = AnaliseAmbos()
        resposta = crew.executar(self.state.text)
        return resposta