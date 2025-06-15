from crewai import Agent, Task, Crew, Process
import os
import pydantic
from custom_tool_vendas import QueryCSVCabecalho

class AnaliseAmbos:
    def __init__(self):
        self.tool_file_path_cabecalho = os.path.join('extract/202401_NFs_Cabecalho.csv')
        self.tool_file_path_itens = os.path.join('extract/202401_NFs_Itens.csv')
        self.llm = "gpt-4o-mini"
        self._setup_crew()

    def _setup_crew(self):
        vendas_tool_cabecalho = QueryCSVCabecalho(file_path=self.tool_file_path_cabecalho)
        vendas_tool_itens = QueryCSVCabecalho(file_path=self.tool_file_path_itens)



         # === Agente Coordenador ===
        coordenador_fiscal = Agent(
            role="Coordenador Fiscal",
            goal="Analisar a solicitação do usuário e delegar corretamente as tarefas.",
            backstory=(
                "Você é um coordenador de equipe com profundo conhecimento em dados fiscais e experiência em direcionar corretamente demandas técnicas. "
                "Sua função é entender a consulta inicial e decidir quais agentes especialistas devem processar a requisição, garantindo precisão e eficiência."
            ),
        memory=True,
        verbose=True,
        llm=self.llm
        )

        # === Agente 2 ===
        analista_cabecalho = Agent(
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

        # === Agente 3 ===
        analista_itens = Agent(
            role="Especialista em Consultas de Itens Fiscais",
            goal="Interpretar e gerar queries precisas sobre dados de itens fiscais com base em um arquivo CSV.",
            backstory=(
                "Você é um analista de dados fiscais com vasta experiência em análise de itens de notas fiscais. "
                "Seu trabalho é transformar perguntas em linguagem natural em consultas eficientes, usando pandas, com base nas colunas disponíveis no CSV."
            ),
            memory=True,
            verbose=True,
            llm=self.llm
        )

        # === Task coordenador ===

        tarefa_coordenador = Task(
            description=(
                "Sua missão é interpretar a pergunta do usuário: '{query}'\n\n"

                "Você é o Coordenador Fiscal, responsável por organizar e distribuir corretamente as tarefas para dois analistas especialistas:\n\n"
                "1. **Agente 'Especialista em Consultas Fiscais' (analista_cabecalho)**:\n"
                "   - Trabalha com o arquivo: 'extract/202401_NFs_Cabecalho.csv'\n"
                "   - Colunas disponíveis:\n"
                "     CHAVE DE ACESSO, MODELO, SÉRIE, NÚMERO, NATUREZA DA OPERAÇÃO, DATA EMISSÃO, EVENTO MAIS RECENTE, DATA/HORA EVENTO MAIS RECENTE,\n"
                "     CPF/CNPJ Emitente, RAZÃO SOCIAL EMITENTE, INSCRIÇÃO ESTADUAL EMITENTE, UF EMITENTE, MUNICÍPIO EMITENTE, CNPJ DESTINATÁRIO,\n"
                "     NOME DESTINATÁRIO, UF DESTINATÁRIO, INDICADOR IE DESTINATÁRIO, DESTINO DA OPERAÇÃO, CONSUMIDOR FINAL, PRESENÇA DO COMPRADOR,\n"
                "     VALOR NOTA FISCAL.\n\n"
                "2. **Agente 'Especialista em Consultas de Itens Fiscais' (analista_itens)**:\n"
                "   - Trabalha com o arquivo: 'extract/202401_NFs_Itens.csv'\n"
                "   - Colunas disponíveis:\n"
                "     CHAVE DE ACESSO, MODELO, SÉRIE, NÚMERO, NATUREZA DA OPERAÇÃO, DATA EMISSÃO, CPF/CNPJ Emitente, RAZÃO SOCIAL EMITENTE,\n"
                "     INSCRIÇÃO ESTADUAL EMITENTE, UF EMITENTE, MUNICÍPIO EMITENTE, CNPJ DESTINATÁRIO, NOME DESTINATÁRIO, UF DESTINATÁRIO,\n"
                "     INDICADOR IE DESTINATÁRIO, DESTINO DA OPERAÇÃO, CONSUMIDOR FINAL, PRESENÇA DO COMPRADOR, NÚMERO PRODUTO, DESCRIÇÃO DO PRODUTO/SERVIÇO,\n"
                "     CÓDIGO NCM/SH, NCM/SH (TIPO DE PRODUTO), CFOP, QUANTIDADE, UNIDADE, VALOR UNITÁRIO, VALOR TOTAL.\n\n"

                "🔗 IMPORTANTE:\n"
                "A coluna **CHAVE DE ACESSO** está presente nos dois arquivos e **deve ser utilizada como vínculo principal entre o cabeçalho e os itens**.\n"
                "Ou seja, para consultar os produtos relacionados a uma nota fiscal, é necessário primeiro obter a(s) `CHAVE DE ACESSO` no cabeçalho\n"
                "e depois usá-la(s) para filtrar os itens correspondentes no outro arquivo.\n\n"

                "### Etapas que você deve seguir:\n"
                "1. Divida a pergunta em subtarefas, descrevendo claramente o que precisa ser respondido.\n"
                "2. Para cada subtarefa, defina:\n"
                "   - Qual informação deve ser buscada\n"
                "   - Em qual dos dois arquivos (cabeçalho ou itens)\n"
                "   - Qual agente é o mais adequado para executar a tarefa\n"
                "3. Redija uma **query clara em linguagem natural** para cada subtarefa e delegue ao agente responsável.\n"
                "4. Aguarde o retorno de cada agente e registre os resultados.\n"
                "5. Por fim, redija uma **resposta final objetiva**, combinando as informações recebidas, explicando **como chegou à conclusão**.\n\n"

                "⚠️ Não resolva você mesmo os cálculos ou consultas. Sua função é apenas orquestrar a execução e consolidar os resultados.\n\n"

                "📌 **Instruções específicas para o uso da ferramenta `Delegate work to coworker`:**\n"
                "Ao usar essa ferramenta, certifique-se de passar os argumentos como **strings simples** e **bem formatadas**. O `Action Input` deve conter:\n\n"
                "- `task`: descrição textual da tarefa, como string.\n"
                "- `context`: contexto completo da tarefa, como string.\n"
                "- `coworker`: nome exato do agente, conforme definido no atributo `role`.\n\n"
                "✅ Exemplo correto:\n"
                "```\n"
                "Action: Delegate work to coworker\n"
                "Action Input:\n"
                "{\n"
                "  \"task\": \"Buscar todas as CHAVES DE ACESSO das notas fiscais emitidas em janeiro de 2024 com valor total acima de 50 mil.\",\n"
                "  \"context\": \"Use o arquivo 'extract/202401_NFs_Cabecalho.csv'. Filtre usando pandas onde VALOR NOTA FISCAL > 50000 e DATA EMISSÃO em janeiro. Retorne somente a coluna CHAVE DE ACESSO.\",\n"
                "  \"coworker\": \"Especialista em Consultas Fiscais\"\n"
                "}\n"
                "```\n"
            ),
    
        expected_output=(
            "Um relatório com:\n"
            "- A lista de subtarefas geradas\n"
            "- O agente escolhido para cada uma\n"
            "- A resposta obtida de cada agente\n"
            "- Uma resposta final clara para o usuário, com base nessas respostas"
        ),
        agent=coordenador_fiscal
        )

        # === Task 2 ===
        tarefa_analise_cabecalho = Task(
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

                "Com base nas colunas do CSV 202401_NFs_Cabecalho.csv escreva um código para a solicitação do coordenador\n\n"

                "A query deve sempre carregar o CSV com o seguinte código:\n"
                "```python\n"
                "import pandas as pd\n"
                "df = pd.read_csv('extract/202401_NFs_Cabecalho.csv', sep=',', encoding='utf-8')\n"
                "```\n"
                "E deve armazenar o resultado final na variável `resultado`\n"
                "⚠️ Importante: Se o resultado for uma tabela ou agrupamento com muitas linhas, você **deve usar** `.to_string(index=False)` no final da query, para garantir que o conteúdo completo seja exibido, sem truncamento.\n"
            ),
            expected_output="resultado em um texto simples, objetivo, de facil entendimento para o usuário porem explicando o motivo da resposta",
            agent=analista_cabecalho,
            tools=[vendas_tool_cabecalho]
        )

        # === Task 3 ===
        tarefa_analise_itens = Task(
            description=(
                "Você é um especialista em manipulação de dados fiscais e análise de planilhas. "
                "Seu papel é interpretar perguntas feitas por usuários em linguagem natural e, com base nelas, gerar consultas em Python usando a biblioteca pandas. "
                "Essas consultas devem operar sobre um arquivo CSV que contém os itens de notas fiscais eletrônicas emitidas no mês de janeiro de 2024.\n\n"

                "Você entende profundamente o significado e a estrutura de cada coluna presente no arquivo e sabe exatamente como consultar, filtrar, agrupar ou calcular valores com base nas perguntas recebidas. "
                "A consulta que você gerar será executada dentro de um ambiente controlado usando a ferramenta `QueryCSV`, que executa o código Python e retorna o conteúdo da variável `resultado` como resposta final ao usuário.\n\n"

                "O CSV contém as seguintes colunas:\n\n"
                "- CHAVE DE ACESSO\n"
                "- MODELO\n"
                "- SÉRIE\n"
                "- NÚMERO\n"
                "- NATUREZA DA OPERAÇÃO\n"
                "- DATA EMISSÃO\n"
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
                "- NÚMERO PRODUTO\n"
                "- DESCRIÇÃO DO PRODUTO/SERVIÇO\n"
                "- CÓDIGO NCM/SH\n"
                "- NCM/SH (TIPO DE PRODUTO)\n"
                "- CFOP\n"
                "- QUANTIDADE\n"
                "- UNIDADE\n"
                "- VALOR UNITÁRIO\n"
                "- VALOR TOTAL\n\n"

                "Com base nas colunas do CSV 202401_NFs_Itens.csv escreva um código para a solicitação do coordenador\n\n"

                "A query deve sempre carregar o CSV com o seguinte código:\n"
                "```python\n"
                "import pandas as pd\n"
                "df = pd.read_csv('extract/202401_NFs_Itens.csv', sep=',', encoding='utf-8')\n"
                "```\n"
                "E deve armazenar o resultado final na variável `resultado`\n"
                "⚠️ Importante: Se o resultado for uma tabela ou agrupamento com muitas linhas, você **deve usar** `.to_string(index=False)` no final da query, para garantir que o conteúdo completo seja exibido, sem truncamento.\n"
            ),
            expected_output="resultado em um texto simples, objetivo, de facil entendimento para o usuário porem explicando o motivo da resposta",
            agent=analista_itens,
            tools=[vendas_tool_itens]
        )


        # Criar a crew com ambas as tarefas e agentes
        self.crew = Crew(
            agents=[analista_cabecalho, analista_itens],
            tasks=[tarefa_coordenador,tarefa_analise_cabecalho, tarefa_analise_itens],
            process=Process.hierarchical,
            verbose=True,
            manager_agent=coordenador_fiscal
        )

    def executar(self,query: str) -> str:
        result = self.crew.kickoff(inputs={"query": query})
        return result.raw