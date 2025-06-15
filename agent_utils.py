import zipfile
from pathlib import Path

class Utils:
    @staticmethod
    def descompactar_arquivo_zip(caminho_zip: str, destino: str = None) -> None:
        """
        Descompacta um arquivo .zip no diretório de destino informado.
        
        :param caminho_zip: Caminho completo para o arquivo .zip
        :param destino: Caminho do diretório onde os arquivos serão extraídos (padrão: mesmo diretório do zip)
        """
        caminho_zip = Path(caminho_zip)

        if not caminho_zip.exists() or not zipfile.is_zipfile(caminho_zip):
            raise FileNotFoundError(f"Arquivo ZIP inválido ou não encontrado: {caminho_zip}")

        destino = Path(destino) if destino else caminho_zip.parent
        destino.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
            zip_ref.extractall(destino)
            print(f"✅ Arquivos extraídos para: {destino.resolve()}")

    @staticmethod
    def verificar_e_descompactar(caminho_pasta: str, caminho_zip: str) -> None:
        """
        Verifica se a pasta está vazia. Se estiver, descompacta o .zip nela.
        """
        pasta = Path(caminho_pasta)

        if not pasta.exists():
            print(f"📁 Pasta não existe, criando: {pasta}")
            pasta.mkdir(parents=True)

        arquivos = list(pasta.iterdir())

        if arquivos:
            print(f"📂 Pasta '{pasta}' já contém arquivos. Nenhuma ação necessária.")
        else:
            print(f"⚠️ Pasta '{pasta}' está vazia. Iniciando descompactação...")
            try:
                Utils.descompactar_arquivo_zip(caminho_zip, caminho_pasta)
            except Exception as e:
                print(f"❌ Falha ao descompactar: {e}")

# Exemplo de uso:
if __name__ == "__main__":
    try:
        Utils.verificar_e_descompactar("extract", "zip/202401_NFs.zip")
    except Exception as e:
        print(f"❌ Erro: {e}")