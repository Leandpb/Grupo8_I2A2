
import crewai
from fluxo import FluxoFiscal
from orquestrador import AnaliseAmbos
from agent_utils import Utils

def main():

    Utils.verificar_e_descompactar(caminho_pasta="extract", caminho_zip="zip/202401_NFs.zip")
    # Cria e executa a crew com análise + validação
    #crew = AnaliseCabecalhoCrew()
    #resultado = crew.executar(query="Qual é o fornecedor que teve maior montante recebido?")

    crew = AnaliseAmbos()
    resultado = crew.executar(query="Quais produtos constam nas notas fiscais emitidas em janeiro com valor total acima de 50 mil?")

    #crew = AnaliseItensCrew()
    #resultado = crew.executar(query="Qual item teve maior volume entregue (em quantidade)?")

    #crew = FacadeCrew()
    #resultado = crew.executar(text="Qual é o fornecedor que teve maior montante recebido?")

    #print(resultado)
    # 1 - pergunta = "Qual é o fornecedor que teve maior montante recebido?"
    #2 - pergunta = "Qual item teve maior volume entregue (em quantidade)?"
    #3 - pergunta = "Quais produtos constam nas notas fiscais emitidas em janeiro com valor total acima de 50 mil?"
    #4 - pergunta = "Qual o total de notas fiscais emitidas por UF do destinatário?"
    fluxo = FluxoFiscal()
    resultado = fluxo.kickoff(inputs={'text': pergunta})

    print("✅ Resposta final:", resultado)


if __name__ == "__main__":
    main()
