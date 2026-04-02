# projeto_eucalipto

Pipeline para processamento de nuvens de pontos de árvores de eucalipto.

Este repositório integra, de forma reprodutível, os seguintes blocos:

- Isolamento e segmentação semântica de árvores com **ForestFormer3D / FF3D_inference** (via Docker).
- Isolamento de árvores com **treeiso** (artemis_treeiso) quando já existem rótulos de árvores.
- Extração heurística de tronco (quando não há classe semântica de tronco).
- Cálculo de **DAP (DBH)** por métodos robustos (ensemble de slices com RANSAC).
- Estimativa de **volume de tronco** usando o modelo de cilindro (método padrão) e estrutura para outros métodos.

Toda a lógica reutilizável fica no pacote interno `eucalipto/` e o fluxo padrão hoje é
orquestrado pelo script `run_ff3d_pipeline.py`.

---

## Estrutura principal

- `eucalipto/io.py`: leitura/escrita de LAS/LAZ/PLY, normalização, altura da árvore.
- `eucalipto/isolation_ff3d.py`: wrapper para o **FF3D_inference** via Docker, incluindo
	extração direta dos pontos de tronco usando o rótulo semântico de tronco.
- `eucalipto/isolation_treeiso.py`: helpers para separar a nuvem por `treeID` (pensando em integração
	futura com o **treeiso**/artemis_treeiso).
- `eucalipto/trunk_heuristic.py`: heurística de identificação de tronco (PCA global, distância ao eixo,
	linearidade, scattering, componente conectada). Usado na rota treeiso.
- `eucalipto/dbh_methods.py`: métodos de cálculo de DAP (RANSAC em fatia única, least squares e
	**ensemble multi‑slice**, que é o padrão recomendado).
- `eucalipto/volume_methods.py`: métodos de volume; atualmente o fluxo de produção usa
	**modelo de cilindro** a partir de DAP + altura.
- `eucalipto/pipeline_core.py`: funções de alto nível para:
	- rodar isolamento (FF3D ou treeiso),
	- extrair tronco (semântico ou heurístico),
	- calcular DAP e volume para cada árvore.
- `run_ff3d_pipeline.py`: script principal de exemplo usando FF3D como fonte de isolamento + tronco.

---

## Dependências externas esperadas

O projeto assume que, no mesmo nível deste repositório, existem os diretórios (já clonados):

- `FF3D_inference/` (repositório com o wrapper Docker de inferência).
- `ForestFormer3D/` (repositório principal da rede, usado indiretamente pelo FF3D_inference).
- `artemis_treeiso/` (treeiso; usado hoje principalmente como referência e para fluxos futuros).

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

Já existem módulos para suportar um fluxo baseado em **treeiso** + heurística de tronco:

- `eucalipto.isolation_treeiso`: separa as árvores a partir de um campo `treeID`.
- `eucalipto.trunk_heuristic`: aplica a heurística baseada em PCA local/global para
	separar tronco de folhas em cada árvore.

Um script análogo ao `run_ff3d_pipeline.py` pode ser criado usando as funções de
`pipeline_core` para orquestrar esse caminho. Neste momento, o fluxo FF3D é o
**padrão recomendado** por ser mais estável e já incorporar a segmentação
semântica de tronco.

---

## Futuras extensões

- Integração mais direta com o código do **treeiso** (artemis_treeiso) para cenários
	sem rótulos prévios de árvore.
- Implementação de métodos adicionais de volume (taper, frustum, QSM via PyTLidar)
	a partir do código existente nos notebooks.
- Criação de uma CLI única (por exemplo, `eucalipto run --config config.yaml`) sobre
	os módulos já implementados.
