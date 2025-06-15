from crewai import Agent, Task, Crew, Process
import os
from custom_tool_vendas import QueryCSVCabecalho

class AnaliseCabecalhoCrew:
    def __init__(self):
        self.tool_file_path = os.path.join('extract/202401_NFs_Cabecalho.csv')
        self.llm = "gpt-4o-mini"
        self._setup_crew()

    def _setup_crew(self):
        vendas_tool = QueryCSVCabecalho(file_path=self.tool_file_path)
        

        # === Agente 1 ===
        analista = Agent(
            role="Especialista em Consultas Fiscais",
            goal="Interpretar e gerar queries precisas sobre dados fiscais com base em um arquivo CSV contendo cabeçalhos de notas fiscais.",
            backstory=(
            "Você é um analista de dados fiscais com ampla experiência em manipulação de grandes volumes de dados contábeis e tributários. "
            "Seu trabalho consiste em interpretar arquivos estruturados, como planilhas de notas fiscais, e construir consultas lógicas e eficientes "
            "com base nas colunas disponíveis."
            ),
            memory=True,
            verbose=True,
            llm=self.llm
        )

        # === Task 1 ===
        tarefa_analise = Task(
            description=(
                "Você é um especialista em manipulação de dados fiscais e análise de planilhas. "
                "Seu papel é interpretar perguntas feitas por usuários em linguagem natural e, com base nelas, gerar consultas em Python usando a biblioteca pandas. "
                "Essas consultas devem operar sobre um arquivo CSV que contém os cabeçalhos de notas fiscais eletrônicas emitidas no mês de janeiro de 2024.\n\n"

                "Você entende profundamente o significado e a estrutura de cada coluna presente no arquivo e sabe exatamente como consultar, filtrar, agrupar ou calcular valores com base nas perguntas recebidas. "
                "A consulta que você gerar será executada dentro de um ambiente controlado usando a ferramenta `QueryCSV`, que executa o código Python e retorna o conteúdo da variável `resultado` como resposta final ao usuário.\n\n"

                "O CSV contém as seguintes colunas:\n\n"
                "- CHAVE DE ACESSO\n"
                "- MODELO\n"
                "- SÉRIE\n"
                "- NÚMERO\n"
                "- NATUREZA DA OPERAÇÃO\n"
                "- DATA EMISSÃO\n"
                "- EVENTO MAIS RECENTE\n"
                "- DATA/HORA EVENTO MAIS RECENTE\n"
                "- CPF/CNPJ Emitente\n"
                "- RAZÃO SOCIAL EMITENTE\n"
                "- INSCRIÇÃO ESTADUAL EMITENTE\n"
                "- UF EMITENTE\n"
                "- MUNICÍPIO EMITENTE\n"
                "- CNPJ DESTINATÁRIO\n"
                "- NOME DESTINATÁRIO\n"
                "- UF DESTINATÁRIO\n"
                "- INDICADOR IE DESTINATÁRIO\n"
                "- DESTINO DA OPERAÇÃO\n"
                "- CONSUMIDOR FINAL\n"
                "- PRESENÇA DO COMPRADOR\n"
                "- VALOR NOTA FISCAL\n\n"

                "Com base nas colunas do CSV 202401_NFs_Cabecalho.csv escreva um código para essa solicitação:\n\n"
                "{query}"

                "A query deve sempre carregar o CSV com o seguinte código:\n"
                "```python\n"
                "import pandas as pd\n"
                "df = pd.read_csv('extract/202401_NFs_Cabecalho.csv', sep=',', encoding='utf-8')\n"
                "```\n"
                "E deve armazenar o resultado final na variável `resultado`\n"
                "⚠️ Importante: Se o resultado for uma tabela ou agrupamento com muitas linhas, você **deve usar** `.to_string(index=False)` no final da query, para garantir que o conteúdo completo seja exibido, sem truncamento.\n"
            ),
            expected_output="resultado em um texto simples, objetivo, de facil entendimento para o usuário porem explicando o motivo da resposta",
            agent=analista,
            tools=[vendas_tool]
        )


        # Criar a crew com ambas as tarefas e agentes
        self.crew = Crew(
            agents=[analista],
            tasks=[tarefa_analise],
            process=Process.sequential,
            verbose=True
        )

    def executar(self,query: str) -> str:
        result = self.crew.kickoff(inputs={"query": query})
        return result.raw