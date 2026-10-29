# Volume Validation Benchmark (Canoas Dataset)

Validação experimental da estimativa volumétrica LiDAR (ForestFormer3D + reconstrução de fuste + integração geométrica) confrontada com o inventário florestal de campo (`Result_IFPC_Canoas.xlsx`).

---

## 1. Objetivos da Pesquisa

1. **Validação Cruzada e Curva de Saturação Amostral (Experimento 1 — Talhão 37, 10cm):**
   - Mensurar o erro preditivo em nível de árvore e de parcela utilizando validação *Leave-One-Plot-Out* (LOPO, 11 folds) sem vazamento de dados entre calibração e teste.
   - Comparar lado a lado dois critérios de seleção de hiperparâmetros nas parcelas de treino:
     - **Mode 1A (Tree-RMSE Calibrated):** minimiza o RMSE individual por árvore.
     - **Mode 1B (Plot-Bias Calibrated):** minimiza o viés relativo médio dentre os modelos com RMSE estável.
     - **Mode 2 (Human-in-the-Loop / Oracle):** limite superior teórico com seleção ótima por árvore.
   - Avaliar a curva de aprendizado variando o número de parcelas de calibração ($N \in \{2, 4, 6, 8, 10\}$) em combinações sem repetição.

2. **Transferibilidade Intra-Clone entre Talhões Adjacentes (Experimento 2 — Talhão 62 $\to$ Talhão 63, 5cm):**
   - Avaliar a extrapolação de modelos volumétricos calibrados nas 4 parcelas de campo do Talhão 62 (25,96 ha) para uma amostra do bloco adjacente de mesmo clone e idade (Talhão 63, 15,04 ha, que não possui parcelas de campo anotadas na planilha oficial).

3. **Sensibilidade à Densidade de Pontos / Resolução de Voo (Experimento 3 — 5cm vs. 10cm no Talhão 62):**
   - Comparar diretamente a precisão volumétrica individual e por parcela nas mesmas 4 unidades amostrais do Talhão 62 (`500, 503, 508, 509`) segmentadas de forma independente pelo ForestFormer3D nos voos de 5cm (~548 pts/m²) e de 10cm (~378 pts/m²).

4. **Transferibilidade Entre Talhões em Mesma Resolução vs. Resolução Cruzada (Experimento 4 — T37 10cm $\leftrightarrow$ T62 10cm vs. 5cm):**
   - Testar se a divergência observada ao transferir parâmetros entre os talhões T37 e T62 decorre de diferenças alométricas entre os talhões ou da incompatibilidade de resolução das nuvens de pontos (10cm vs. 5cm), utilizando as densidades oficiais de inventário (`1034,09` árvores/ha no T37 e `650,00` árvores/ha no T62).

---

## 2. Estrutura de Diretórios e Dados

Todos os recortes de nuvens de pontos e nuvens classificadas estão centralizados em `/mnt/scratch/dataset/euc/canoa/` :

- `/mnt/scratch/dataset/euc/canoa/IFPC-Canoas/Result_IFPC_Canoas.xlsx` : Inventário florestal de referência (abas `Covas`, `Parcelas`, `Talhões`).
- `/mnt/scratch/dataset/euc/canoa/recortes/talhao37_plots_10cm/` : 11 parcelas do Talhão 37 (voo 10cm, brutas e classificadas pelo FF3D).
- `/mnt/scratch/dataset/euc/canoa/recortes/talhao62_plots_5cm/` : 4 parcelas do Talhão 62 (voo 5cm, brutas e classificadas pelo FF3D).
- `/mnt/scratch/dataset/euc/canoa/recortes/talhao62_plots_10cm/` : 4 parcelas do Talhão 62 (voo 10cm, brutas e classificadas nativamente pelo FF3D).
- `/mnt/scratch/dataset/euc/canoa/recortes/samples_5cm/` : Recortes amostrais dos talhões 62 e 63 em 5cm.

Estrutura do módulo:
```text
volume_validation/
├── configs/
│   ├── config_t37_cv.yaml           # Configuração do Experimento 1 (T37 10cm)
│   ├── config_t62_t63.yaml          # Configuração do Experimento 2 (T62 -> T63 5cm)
│   └── config_flight_res.yaml       # Configuração do Experimento 3 (T62 5cm vs 10cm)
├── src/
│   ├── data_loader.py               # Leitura do inventário Excel e nuvens LAS/LAZ
│   ├── stem_enhancement.py          # Reatribuição de pontos de tronco via KDTree e controle de qualidade
│   ├── spatial_matcher.py           # Alinhamento rígido 2D + algoritmo Húngaro (Munkres)
│   ├── biometrics.py                # Extração de DAP, altura e grade de métodos volumétricos
│   └── cross_validation.py          # Motores LOPO (Tree-RMSE e Plot-Bias) e curva de saturação
├── experiments/
│   ├── exp1_t37_learning_curve.py
│   ├── exp2_t62_to_t63_transfer.py
│   ├── exp3_flight_resolution_5vs10cm.py
│   └── exp4_t37_to_t62_10cm_transfer.py
├── notebooks/
│   └── volume_validation_analysis.ipynb
└── results/                         # Tabelas CSV e sumários JSON gerados pelos experimentos
```

---

## 3. Síntese dos Resultados e Implicações Técnicas

### Experimento 1: Validação LOPO e Curva de Saturação (Talhão 37 — 10cm, $n = 118$ árvores em 11 parcelas)
- **Desalinhamento entre RMSE individual e Viés de Parcela:** Como o volume voxelizado escala com $\text{voxel\_size}^3$, a minimização direta do RMSE por árvore (**Mode 1A**) penaliza quadraticamente *outliers* positivos (vazamento de copa/galhos para o tronco) e seleciona sistematicamente `voxel_vs0.08` (RMSE individual de `0,1140 m³`, mas viés médio de `-25,9%`), resultando em erro absoluto médio nas parcelas cegas de **`34,38%`** (viés de `-33,81%`).
- **Calibração por Viés Estável (Mode 1B):** Ao selecionar o hiperparâmetro que minimiza o viés relativo médio dentro da faixa de RMSE estável (`voxel_vs0.09`, RMSE individual de `0,1232 m³` e viés de `+2,8%`), o erro médio LOPO nas parcelas cai para **`22,61%`** (viés de `-8,24%`), igualando o patamar do *Human-in-the-Loop* (**`22,40%`**).
- **Curva de Saturação Amostral (Mode 1B):** O erro médio em parcelas de teste cai monotonicamente com o aumento do número de parcelas de treino: **`32,8%`** ($N=2$) $\to$ **`27,5%`** ($N=4$) $\to$ **`24,5%`** ($N=6$) $\to$ **`22,7%`** ($N=8$) $\to$ **`22,6%`** ($N=10$), estabilizando a partir de 6 a 8 parcelas.

### Experimento 2: Transferibilidade Intra-Clone (Talhão 62 $\to$ Talhão 63 — 5cm)
- **Calibração no Talhão 62 ($n = 40$ árvores em 4 parcelas, `650` árvores/ha, referência oficial `201,29 m³/ha`):**
  - **Mode 1A (Tree-RMSE, `voxel_vs0.07`):** RMSE individual de `0,1709 m³` (viés `-22,7%`), extrapolando `146,8 m³/ha` (`-27,1%` no T62).
  - **Mode 1B (Plot-Bias, `voxel_vs0.08`):** RMSE individual de `0,2024 m³` (viés `+10,8%`, mediana `-0,6%`), extrapolando **`210,4 m³/ha` (`+4,5%` no T62)**.
  - **Mode 2 (Human-in-the-Loop):** extrapola **`193,9 m³/ha` (`-3,7%` no T62)**.
- **Efeito da Densidade na Amostra do Talhão 63 (`ff3d_full`):** A nuvem de teste disponível para o T63 (`samples_5cm/ff3d_full/classified_cloud.laz`, originada de um recorte de `22,6m × 22,6m` na faixa de sobreposição entre os blocos `cloud0.las` e `cloud1.las` do T63) possui densidade de `2.484 pts/m²` (4,5x superior às parcelas de 5cm do T62, que têm `~548 pts/m²`). Essa sobreposição de passadas infla a ocupação de voxels em ~36%, fazendo com que `voxel_vs0.07` atinja `201,1 m³/ha` (`-0,1%`) e `voxel_vs0.08` suba para `285,3 m³/ha` (`+41,8%`). Isso demonstra que até dentro de um mesmo voo é necessário normalizar/subamostrar faixas de sobreposição de passadas antes de aplicar métodos baseados em ocupação de voxels.

### Experimento 3: Comparação de Resolução 5cm vs. 10cm com Segmentação Nativa FF3D (Talhão 62)
- Passando as 4 parcelas de 10cm (`~378 pts/m²`) diretamente pelo ForestFormer3D e comparando com as mesmas 4 parcelas em 5cm (`~548 pts/m²`):
  - **5cm ($n = 40$ árvores pareadas):** `Mode 1A` seleciona `voxel_vs0.07` (RMSE `0,1709 m³`, erro de parcela `28,8%`); `Mode 1B` seleciona `voxel_vs0.08` (RMSE `0,2024 m³`, erro de parcela `28,7%`).
  - **10cm ($n = 38$ árvores pareadas):** `Mode 1A` seleciona `voxel_vs0.08` (RMSE `0,2202 m³`, erro de parcela **`18,7%`**, LOPO **`21,8%`**); `Mode 1B` seleciona `voxel_vs0.09` (RMSE `0,2657 m³`, erro de parcela `34,5%`).
- **Implicação:** A segmentação nativa do ForestFormer3D opera com sucesso em nuvens de 10cm (`38` árvores pareadas com campo vs. `40` em 5cm), exigindo apenas o deslocamento de $+1\text{ cm}$ na aresta do voxel (`0.07 -> 0.08` no critério RMSE ou `0.08 -> 0.09` no critério de viés) para compensar a menor densidade superficial de pontos no tronco.

### Experimento 4: Transferência Cruzada entre Talhões em Mesma Resolução (10cm $\leftrightarrow$ 10cm vs. 5cm)
- Utilizando as densidades oficiais da aba `Talhões` (`1034,09` árvores/ha no T37 e `650,00` árvores/ha no T62):
  - **Confluência de Hiperparâmetros em 10cm:** Tanto o T37 (10cm) quanto o T62 (10cm) selecionam exatamente os mesmos modelos ótimos em ambos os critérios: **`voxel_vs0.08`** pelo critério de RMSE individual e **`voxel_vs0.09`** pelo critério de viés.
  - **Transferência T62 $\to$ T37 (10cm):**
    - Quando calibrado em resolução incompatível (**T62 5cm $\to$ T37 10cm**), o erro no T37 é de **`-51,0%`** (Tree-RMSE, `voxel_vs0.07`) e **`-28,3%`** (Plot-Bias, `voxel_vs0.08`).
    - Quando calibrado na mesma resolução (**T62 10cm $\to$ T37 10cm**), o erro no T37 cai para **`-28,3%`** (Tree-RMSE, `voxel_vs0.08`) e **`-0,5%`** (`190,3 m³/ha` vs. `191,1 m³/ha` oficial) no modelo calibrado por viés (`voxel_vs0.09`).
  - **Transferência T37 (10cm) $\to$ T62:**
    - Para **T37 (10cm) $\to$ T62 (10cm)**, o erro é de **`-18,4%`** (`164,3 m³/ha`, `voxel_vs0.08`) e **`+13,0%`** (`227,5 m³/ha`, `voxel_vs0.09`), enquanto na resolução cruzada **T37 (10cm) $\to$ T62 (5cm)** o modelo de viés (`voxel_vs0.09`) superestima o talhão em **`+42,1%`** (`286,1 m³/ha`).

---

## 4. Como Reproduzir

```bash
# Executar os 4 experimentos em sequência
/home/matheuspimenta/.conda/envs/euc/bin/python research/volume_validation/experiments/exp1_t37_learning_curve.py
/home/matheuspimenta/.conda/envs/euc/bin/python research/volume_validation/experiments/exp2_t62_to_t63_transfer.py
/home/matheuspimenta/.conda/envs/euc/bin/python research/volume_validation/experiments/exp3_flight_resolution_5vs10cm.py
/home/matheuspimenta/.conda/envs/euc/bin/python research/volume_validation/experiments/exp4_t37_to_t62_10cm_transfer.py
```
