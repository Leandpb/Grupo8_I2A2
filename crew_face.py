from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

class FacadeCrew:
    def __init__(self):
        self.agent = None
        self.task = None
        self.crew = None
        self.llm = "gpt-4o-mini"

        self._setup_crew()

    def _setup_crew(self):
        self.agent = Agent(
            role="Classificador de Intenção de Consulta Fiscal",
            goal=(
                "Identificar se a pergunta do usuário se refere ao conteúdo de um CSV de cabeçalho de notas fiscais, de itens, ou de ambos."
            ),
            backstory=(
                "Você é um classificador inteligente que entende perguntas sobre documentos fiscais. "
                "Seu papel é decidir se a pergunta está relacionada ao cabeçalho da nota (como valores totais, datas, emitente, destinatário), "
                "aos itens da nota (como produtos, quantidade, valor unitário, NCM, CFOP), ou se exige dados de ambos para ser respondida."
            ),
            memory=False,
            verbose=True,
            llm=self.llm
        )

        self.task = Task(
            description=(
                "Classifique a pergunta abaixo com base nas informações que ela requer para ser respondida:\n\n"
                "- Se a pergunta estiver relacionada apenas ao **cabeçalho da nota** (ex: valores totais, data de emissão, emitente, destinatário), retorne: `cabecalho`\n"
                "- Se estiver relacionada apenas aos **itens da nota** (ex: produtos, quantidade, valor unitário, NCM, CFOP), retorne: `itens`\n"
                "- Se a pergunta precisar de **dados de ambos os arquivos (cabeçalho e itens)** para ser respondida corretamente, retorne: `ambos`\n\n"
                "⚠️ Retorne **apenas uma palavra**: `cabecalho`, `itens` ou `ambos`. Sem texto adicional.\n\n"
                "<pergunta>\n{text}\n</pergunta>"
            ),
            expected_output="A palavra `cabecalho`, `itens` ou `ambos`, sem aspas e sem texto adicional.",
            agent=self.agent
        )

        self.crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            process=Process.sequential
        )

    def executar(self, text):
        """
        Classifica a pergunta como relacionada a 'cabecalho', 'itens' ou 'ambos'.

        :param text: Pergunta do usuário.
        :return: 'cabecalho', 'itens' ou 'ambos'
        """
        result = self.crew.kickoff(inputs={"text": text})
        return result.raw