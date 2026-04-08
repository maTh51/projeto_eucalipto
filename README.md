# projeto_eucalipto

Pipeline para processamento de nuvens de pontos de árvores de eucalipto.

Este repositório integra, de forma reprodutível, os seguintes blocos:

- Isolamento de árvores com **treeiso** ([artemis_treeiso](https://github.com/truebelief/artemis_treeiso))
- Isolamento e segmentação semântica de árvores com **ForestFormer3D / FF3D_inference** (via Docker) —
	[ForestFormer3D](https://github.com/SmartForest-no/ForestFormer3D) e
	[FF3D_inference](https://github.com/bxiang233/FF3D_inference/tree/main/ff3d_forestsens).
- Extração heurística de tronco (quando não há classe semântica de tronco).
- Cálculo de **DAP (DBH)** por métodos geométricos (ensemble de slices com RANSAC).
- Estimativa de **volume de tronco** usando o modelo de cilindro (método padrão),
	além de métodos alternativos (taper, frustum e QSM via PyTLidar/TreeQSM).

---

## Estrutura principal

- `eucalipto/io.py`: leitura/escrita de LAS/LAZ/PLY, normalização, altura da árvore.
- `eucalipto/isolation_ff3d.py`: wrapper para o **FF3D_inference** via Docker, incluindo
	extração direta dos pontos de tronco usando o rótulo semântico de tronco.
- `eucalipto/isolation_treeiso.py`: helpers para separar a nuvem por `treeID` ou `final_segs` e função
	para chamar diretamente o algoritmo original do **treeiso**/artemis_treeiso sobre um diretório.
- `eucalipto/trunk_heuristic.py`: heurística de identificação de tronco (PCA global, distância ao eixo,
	linearidade, scattering, componente conectada). Usado no fluxo com o treeiso.
- `eucalipto/dbh_methods.py`: métodos de cálculo de DAP (RANSAC em fatia única, least squares e
	**ensemble multi‑slice**, que é o padrão recomendado).
- `eucalipto/volume_methods.py`: métodos de volume; atualmente o fluxo de produção usa
	**modelo de cilindro** a partir de DAP + altura, mas também há opções de
	**integração da curva de taper** (r(h)), **frustum** (cone truncado) com
	raios estimados na base e no topo, e um método **QSM** ("qsm") que delega o
	ajuste do modelo a bibliotecas externas (PyTLidar/TreeQSM) via
	``qsm_volume_func``.
- `eucalipto/pipeline_core.py`: funções de alto nível para:
	- rodar isolamento (FF3D ou treeiso),
	- extrair tronco (semântico ou heurístico),
	- calcular DAP e volume para cada árvore.
- `run_ff3d_pipeline.py`: script principal de exemplo usando FF3D como fonte de isolamento + tronco.
- `run_treeiso_pipeline.py`: script de exemplo usando treeiso + heurística de tronco.

---

## Dependências externas esperadas

O projeto assume que, no mesmo nível deste repositório, existem os diretórios (já clonados):

- `FF3D_inference/` — wrapper Docker de inferência do ForestFormer3D
	- GitHub: <https://github.com/bxiang233/FF3D_inference/tree/main/ff3d_forestsens>
- `ForestFormer3D/` — repositório principal da rede FF3D
	- GitHub: <https://github.com/SmartForest-no/ForestFormer3D>
- `artemis_treeiso/` — implementação do algoritmo treeiso
	- GitHub: <https://github.com/truebelief/artemis_treeiso>

Além disso, as bibliotecas Python principais usadas pelo pacote `eucalipto` incluem, entre outras:

- `numpy`, `laspy`, `scikit-learn`, `scipy`, `pyransac3d`, `plyfile`.

O ambiente deve ser configurado com essas dependências (por exemplo, via `pip` ou ambiente conda).

---

## Fluxo padrão (FF3D + DBH ensemble + volume cilindro)

O fluxo considerado mais estável atualmente é:

1. **Isolamento e segmentação de tronco com FF3D**
	 - FF3D (via `FF3D_inference/ff3d_forestsens/run_docker_locally.sh`) recebe uma nuvem de pontos de um talhão
		 e produz rótulos de instância (árvore) e rótulos semânticos (tronco, copa, etc.).
	 - O módulo `eucalipto.isolation_ff3d`:
		 - chama o script Docker,
		 - lê o arquivo de saída com rótulos,
		 - agrupa pontos por instância
		 - e filtra apenas os pontos classificados como **tronco**.

2. **Cálculo do DAP (DBH)**
	 - O módulo `eucalipto.dbh_methods` recebe apenas os pontos de tronco de cada árvore.
	 - O método padrão é o `ensemble`:
		 - fatias em torno da altura do peito (1,3 m acima da base),
		 - ajuste de círculo por RANSAC em cada fatia,
		 - uso da mediana dos DAPs por fatia como valor final.

3. **Cálculo do volume do tronco**
	 - O módulo `eucalipto.volume_methods` usa o DAP calculado + altura aproximada do tronco
		 (diferença `max(z) - min(z)` dos pontos de tronco) e aplica o **modelo de cilindro**:
		 \( V = \pi r^2 h \).
	 - Opcionalmente, se fornecida a densidade (`kg/m³`), é estimada a massa seca aproximada.

4. **Saída**
	 - O script `run_ff3d_pipeline.py` escreve um CSV resumo com, para cada árvore:
		 - `tree_id`,
		 - `dbh_cm`,
		 - `height_m`,
		 - `volume_m3`, `volume_liters`,
		 - `mass_kg` (se densidade fornecida),
		 - método de DAP utilizado.

---

## Como usar o `run_ff3d_pipeline.py`

1. Ajuste os caminhos na seção **CONFIGURATION (EDIT HERE)** do arquivo
	 `run_ff3d_pipeline.py`:

	 - `FF3D_REPO_DIR`: caminho para `FF3D_inference/ff3d_forestsens`.
	 - `BUCKET_IN_DIR` e `BUCKET_OUT_DIR`: pastas usadas pelo FF3D para entrada/saída.
	 - `INPUT_LAZ`: caminho para o arquivo LAS/LAZ do talhão que você quer processar.
	 - `INSTANCE_DIM` / `SEMANTIC_DIM` / `TRUNK_LABEL`: nomes dos campos e rótulo inteiro
		 que o FF3D grava para instância e classe tronco (ajuste de acordo com o formato
		 real de saída do seu modelo).
	 - `RESULTS_DIR`: pasta onde o CSV final será salvo.

2. Garanta que o Docker está instalado e que o fluxo do `FF3D_inference` já foi testado
	 isoladamente na máquina (incluindo download de pesos, build da imagem etc.).

3. Dentro do diretório deste repositório, execute:

	 ```bash
	 python run_ff3d_pipeline.py
	 ```

4. O script irá:
	 - copiar o `INPUT_LAZ` para o `BUCKET_IN_DIR`,
	 - chamar o Docker do FF3D,
	 - extrair os pontos de tronco por árvore,
	 - calcular DAP (ensemble) e volume (cilindro),
	 - gravar o resumo em `RESULTS_DIR/ff3d_metrics_summary.csv`.

---

## Fluxos alternativos (treeiso + heurística de tronco)

Além do fluxo FF3D, o repositório já implementa um pipeline baseado em
**treeiso** + heurística de tronco:

- `eucalipto.isolation_treeiso`:
	- função `run_treeiso_on_dir` chama diretamente o algoritmo original do
	  treeiso (artemis_treeiso) sobre um diretório de arquivos LAS/LAZ,
	  gerando arquivos `*_treeiso.laz` com o campo `final_segs`;
	- função `split_by_tree_id` permite separar uma nuvem por um campo de ID
	  (por exemplo `treeID` ou `final_segs`).
- `eucalipto.trunk_heuristic`: aplica a heurística baseada em PCA local/global para
	separar tronco de folhas em cada árvore.

O script `run_treeiso_pipeline.py` demonstra esse fluxo completo:

1. Roda o treeiso em todos os arquivos de `INPUT_DIR`, gerando `*_treeiso.laz`.
2. Usa `final_segs` para agrupar pontos por árvore/segmento.
3. Aplica a heurística de tronco em cada árvore.
4. Calcula DAP (ensemble) e volume (cilindro, por padrão) para cada tronco.
5. Gera um CSV resumo em `results_treeiso/treeiso_metrics_summary.csv`.

Mesmo com os dois caminhos implementados, o fluxo FF3D continua sendo o
**padrão recomendado** quando disponível, por já trazer a segmentação
semântica de tronco diretamente da rede.