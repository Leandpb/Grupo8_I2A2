from crewai.tools import BaseTool
from pathlib import Path

class QueryCSVCabecalho(BaseTool):
    name: str = "Ferramenta de execução de código de consulta a um CSV"
    description: str = "Executa e retorna dados de uma consulta a partir de um CSV"

    file_path: str

    def _run(self, codigo_python: str) -> str:
        contexto = {}
        try:
            path_csv = str(Path(self.file_path).resolve())

            # substitui qualquer ocorrência do nome do arquivo pelo caminho absoluto
            codigo_python = codigo_python.replace(
                "'extract/202401_NFs_Cabecalho.csv'",
                f"r'{path_csv}'"
            ).replace(
                "'202401_NFs_Cabecalho.csv'",
                f"r'{path_csv}'"
            )

            exec(codigo_python, contexto)

            return contexto['resultado']
        except Exception as e:
            return f"[ERRO] Falha ao executar a query: {e}"
        
class QueryCSVitens(BaseTool):
    name: str = "Ferramenta de execução de código de consulta a um CSV"
    description: str = "Executa e retorna dados de uma consulta a partir de um CSV"

    file_path: str

    def _run(self, codigo_python: str) -> str:
        contexto = {}
        try:
            path_csv = str(Path(self.file_path).resolve())

            # substitui qualquer ocorrência do nome do arquivo pelo caminho absoluto
            codigo_python = codigo_python.replace(
                "'extract/202401_NFs_Itens.csv'",
                f"r'{path_csv}'"
            ).replace(
                "'202401_NFs_Itens.csv'",
                f"r'{path_csv}'"
            )

            exec(codigo_python, contexto)

            return contexto['resultado']
        except Exception as e:
            return f"[ERRO] Falha ao executar a query: {e}"