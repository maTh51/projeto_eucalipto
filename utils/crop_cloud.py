import laspy
import numpy as np
import os

def recortar_nuvem_centro(input_file, output_file, cx, cy, lado_m=100, chunk_size=1_000_000):
    # Calcula os limites do quadrado (Bounding Box 2D)
    # Se o lado é 100m, recuamos 50m para cada lado a partir do centro
    x_min = cx - (lado_m / 2)
    x_max = cx + (lado_m / 2)
    y_min = cy - (lado_m / 2)
    y_max = cy + (lado_m / 2)

    print(f"Iniciando recorte de {lado_m}x{lado_m}m...")
    print(f"Centro: X={cx}, Y={cy}")
    print(f"Limites de X: {x_min:.2f} a {x_max:.2f}")
    print(f"Limites de Y: {y_min:.2f} a {y_max:.2f}")
    
    if not os.path.exists(input_file):
        print(f"Erro: Arquivo '{input_file}' não encontrado.")
        return

    # Etapa 1: Leitura e Escrita em Chunks (Economiza muita RAM)
    with laspy.open(input_file) as infile:
        # Copia o cabeçalho exato do arquivo original (para não perder campos extras, intensidades, etc)
        header = infile.header
        
        with laspy.open(output_file, mode="w", header=header) as outfile:
            pontos_salvos = 0
            
            # Lê a nuvem de 1 em 1 milhão de pontos por vez
            for chunk in infile.chunk_iterator(chunk_size):
                # Cria uma máscara filtrando estritamente em X e Y. 
                # O eixo Z é ignorado no filtro, logo TODOS os pontos em Z estarão presentes.
                mask = (
                    (chunk.x >= x_min) & (chunk.x <= x_max) &
                    (chunk.y >= y_min) & (chunk.y <= y_max)
                )
                
                pontos_filtrados = chunk[mask]
                
                # Se encontrou pontos do recorte neste pedaço, adiciona ao novo arquivo
                if len(pontos_filtrados) > 0:
                    outfile.write_points(pontos_filtrados)
                    pontos_salvos += len(pontos_filtrados)
                    
            print(f"Recorte estrutural concluído. Total de pontos salvos: {pontos_salvos}")

    if pontos_salvos == 0:
        print("AVISO: Nenhum ponto foi encontrado nas coordenadas informadas.")
        print("Verifique se as coordenadas estão no mesmo CRS (ex: UTM) da nuvem original.")
        return

    # Etapa 2: Correção do Bounding Box e Offset
    # Sem isso, o visualizador 3D vai achar que o arquivo ainda tem o tamanho original
    print("Atualizando os limites geométricos (Bounding Box) no cabeçalho...")
    
    las = laspy.read(output_file)

    # Atualiza o offset para a nova origem (ajuda a evitar problemas de precisão em ponto flutuante)
    las.header.offsets = [np.min(las.x), np.min(las.y), np.min(las.z)]
    
    # Atualiza as dimensões máximas e mínimas oficiais
    las.header.mins = [np.min(las.x), np.min(las.y), np.min(las.z)]
    las.header.maxs = [np.max(las.x), np.max(las.y), np.max(las.z)]
    
    las.write(output_file)
        
    print(f"Sucesso! Recorte salvo definitivamente em: {output_file}")


if __name__ == "__main__":
    # COLOQUE O NOME DO SEU ARQUIVO DE ENTRADA AQUI (pode ser .las ou .laz)
    arquivo_input = "/mnt/scratch/matheuspimenta/canoa/cloud2.las" 
    
    # Nome do arquivo que será gerado
    arquivo_output = "talhao_62_25x25.las" 

    # Suas coordenadas de interesse
    cx_alvo = 443096.70
    cy_alvo = 8007612.78

    recortar_nuvem_centro(
        input_file=arquivo_input, 
        output_file=arquivo_output, 
        cx=cx_alvo, 
        cy=cy_alvo, 
        lado_m=25
    )