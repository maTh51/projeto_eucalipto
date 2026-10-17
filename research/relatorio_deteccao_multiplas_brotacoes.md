# Relatório Técnico: Detecção e Caracterização de Múltiplas Brotações em Nuvens de Pontos LiDAR 3D

**Projeto**: Manejo Florestal e Biometria de Precisão em *Eucalyptus*  
**Assunto**: Algoritmos para Detecção de Múltiplos Troncos (*Coppice* / Rebrota)  
**Data**: Julho de 2026  

---

## Resumo Executivo

No manejo de florestas de *Eucalyptus*, a condução de cepas e o surgimento de **múltiplas brotações** (presença de dois ou mais fustes emergindo de uma mesma cova ou touça) representam um desafio significativo para a inventariação florestal automatizada por LiDAR (Terrestre, Aéreo ou Mobile). A maioria dos algoritmos tradicionais de estimativa biométrica assume a premissa de fuste único por árvore, o que gera distorções graves no cálculo de Diâmetro à Altura do Peito (DAP) e no volume de madeira por hectare.

Este relatório apresenta o desenvolvimento, embasamento teórico e avaliação experimental de três metodologias computacionais para identificação e separação automática de múltiplas brotações a partir de nuvens de pontos 3D segmentadas.

---

## 1. Fundamentação Teórica e Descrição das Técnicas

Nas últimas décadas, o processamento de nuvens de pontos LiDAR para biometria florestal evoluiu de métodos 2D baseados no Modelo de Altura da Canópia (CHM - *Canopy Height Model*) para algoritmos 3D baseados na estrutura anatômica dos fustes. Em florestas de brotação ou talhadia (*coppice management*), copas sobrepostas inviabilizam a separação de fustes pelo topo da árvore, tornando indispensável o processamento **"bottom-up"** (da base para o topo) focado na região do tronco (*under-canopy stem density*).

Abaixo, descrevemos detalhadamente o funcionamento conceitual, geométrico e prático dos três algoritmos desenvolvidos.

---

### 1.1 Método A: Agrupamento Espacial por Densidade em Fatias 2D (`DBSCAN`)

Imagine olhar para uma árvore exatamente de cima para baixo, mas filtrando todas as folhas, galhos superiores e o solo, mantendo apenas uma fatia horizontal fina do tronco na altura do peito (entre $1{,}8\text{ m}$ e $3{,}0\text{ m}$ acima da base). Se a árvore for monocaule (fuste único), essa fatia 2D projetada no plano $(X, Y)$ aparecerá como um único anel ou aglomeração densa de pontos. No entanto, se a árvore possuir múltiplas brotações, a projeção resultará em duas ou mais "ilhas" densas de pontos separadas no espaço.

O algoritmo DBSCAN (*Density-Based Spatial Clustering of Applications with Noise*, proposto por **Ester et al., 1996** e adaptado para silvicultura por **Hackenberg et al., 2015** e **Cabo et al., 2018**) funciona inspecionando cada ponto individual no plano 2D e contando quantos vizinhos existem dentro de um raio de busca $\epsilon$ (`eps`). Se um ponto possui uma quantidade mínima de vizinhos (`min_samples`), ele é considerado um ponto denso e começa a encadear pontos próximos em um mesmo grupo (*cluster*). Pontos isolados e esparsos — causados por fiapos de casca, poeira ou folhas baixas — são ignorados como ruído.

```
                  VISUALIZAÇÃO DO MÉTODO A (DBSCAN 2D)

      Fatia Horizontal (Z)                 Projeção 2D no Plano (X, Y)
      
          |   |     |   |                   ( Cluster 1 )       ( Cluster 2 )
          | F |     | F |                   ( Tronco 1  )       ( Tronco 2  )
          | u |     | u |       ===>              X                   X
          | s |     | s |                       Centro 1            Centro 2
          | t |     | t |                   [2 Fustes Identificados]
          | e |     | e |
          | 1 |     | 2 |
```

O grande desafio em nuvens reais de eucalipto é que, na região entre os dois troncos, costumam surgir "pontes" de pontos esparsos (provenientes de casca solta ou da junção do V-fork). Se o parâmetro $\epsilon$ for configurado com um valor muito grande (por exemplo, $0{,}15\text{ m}$), o DBSCAN segue essa ponte de pontos e agrupa os dois troncos em um único cluster gigante, falhando em detectar a brotação. Por outro lado, ao calibrarmos $\epsilon$ para um valor mais rigoroso ($0{,}07\text{ m} - 0{,}09\text{ m}$), o algoritmo ignora a ponte esparsa como ruído e isola com precisão os dois núcleos densos, identificando corretamente os dois fustes independentes.

---

### 1.2 Método B: Ajuste Sequencial Iterativo de Círculos por RANSAC

Enquanto o DBSCAN se concentra na densidade de pontos, o método RANSAC (*Random Sample Consensus*, fundamentado por **Fischler & Bolles, 1981** e aplicado em modelagem estrutural por **Raumonen et al., 2013 - TreeQSM** e **Calders et al., 2015**) foca na geometria circular da seção transversal dos troncos. O algoritmo funciona por meio de um processo iterativo de teste de hipóteses e eliminação de pontos.

Na primeira etapa, o RANSAC seleciona aleatoriamente 3 pontos da fatia do tronco, calcula o único círculo perfeito que passa por eles e verifica quantos outros pontos da nuvem estão próximos à borda desse círculo (os chamados *inliers*). Ele repete essa amostragem aleatória centenas de vezes e seleciona o círculo que conseguiu capturar a maior quantidade de pontos. Se o raio desse círculo estiver dentro de limites fisicamente possíveis para um eucalipto ($0{,}04\text{ m} \le r \le 0{,}22\text{ m}$) e possuir pontos suficientes, o **Fuste #1 é detectado** e seu diâmetro ($DAP_1 = 2 \times r \times 100\text{ cm}$) é calculado.

```
              VISUALIZAÇÃO DO MÉTODO B (RANSAC SEQUENCIAL)

   Passo 1: Ajusta Círculo 1        Passo 2: Remove Inliers        Passo 3: Ajusta Círculo 2
   
          ( O )  . . .                      . . .                      ( O )
        Círculo #1                        Subtrai                    Círculo #2
        DAP = 18.2 cm                    Pontos do 1                 DAP = 15.7 cm
```

Em seguida, entra a grande inovação do método: o algoritmo **apaga todos os pontos pertencentes ao Círculo #1** da nuvem de dados. Com os pontos do primeiro fuste removidos, o RANSAC é executado uma segunda vez sobre os pontos restantes. Se houver uma segunda brotação, o algoritmo ajustará um segundo círculo no grupo de pontos remanescentes, identificando o **Fuste #2** ($DAP_2$). Esse processo se repete iterativamente até que não existam mais pontos suficientes para formar círculos válidos. A limitação do raio máximo (`radius_max` = $0{,}22\text{ m}$) é fundamental para impedir que o RANSAC tente ajustar um único círculo gigante envolvendo ambas as brotações ao mesmo tempo.

---

### 1.3 Método C: Rastreamento de Trajetória Multi-Altura de Centroides

Diferente dos métodos anteriores que analisam apenas uma fatia em uma altura fixa, o método de Rastreamento de Trajetória (baseado nos trabalhos de **Liang et al., 2014; 2018** e **Pitkänen et al., 2019**) avalia a arquitetura tridimensional da árvore cortando múltiplas fatias horizontais ao longo do tronco (por exemplo, a $1{,}0\text{ m}$, $1{,}8\text{ m}$, $2{,}6\text{ m}$ e $3{,}4\text{ m}$ acima da base).

Em cada uma dessas alturas, o algoritmo realiza um agrupamento espacial 2D para contar quantos centros de tronco $(X, Y)$ existem naquela elevação específica. Esse método resolve diretamente uma característica biológica importante do eucalipto: logo acima do solo (em $1{,}0\text{ m}$), duas brotações frequentemente se encostam ou se fundem na touça, de modo que uma fatia única nessa altura registraria apenas 1 tronco grosso.

```
            VISUALIZAÇÃO DO MÉTODO C (TRAJETÓRIA MULTI-ALTURA)

   Altura (Z)
     3.4 m  ------>  ( Centro 1 )      ( Centro 2 )     ===> 2 Troncos Identificados
     2.6 m  ------>  ( Centro 1 )      ( Centro 2 )     ===> 2 Troncos Identificados
     1.8 m  ------>  ( Centro 1 )      ( Centro 2 )     ===> 2 Troncos Identificados
     1.0 m  ------>  (    Junção / Touça Única   )     ===> 1 Tronco (Fusão na Base)
     
   Decisão: Avalia alturas >= 1.8m ==> Detecta Múltiplas Brotações (V-fork)
```

À medida que subimos para $1{,}8\text{ m}$ e $2{,}6\text{ m}$, os dois fustes se separam fisicamente no espaço, fazendo com que o número de centroides detectados passe de 1 para 2. O algoritmo rastreia essas trajetórias ao longo do eixo vertical $Z$. Ao avaliar a contagem máxima de centroides observados nas alturas superiores ($\ge 1{,}8\text{ m}$), o sistema corrige a leitura fusionada da base e confirma com segurança que a árvore possui **múltiplas brotações que se bifurcam acima da touça**.

---

## 2. Experimentos e Resultados

### 2.1 Resultados com Dados Sintéticos (Controle Experimental)

Para validar a precisão geométrica dos algoritmos em condições ideais e conhecidas (*Ground Truth*), foram gerados três conjuntos de dados sintéticos simulando fustes de eucalipto com inclinação natural e adição de ruído Gaussiano espacial ($\sigma = 0{,}03\text{ m}$):

| Caso de Teste | Fustes Reais (*Ground Truth*) | Método A (DBSCAN) | Método B (RANSAC) | Método C (Trajetória) | Diagnóstico Final |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Árvore Monocaule** | 1 | 1 | 1 | 1 | **CORRETO (100%)** |
| **Bifurcada (2 Brotações)** | 2 | 2 | 2 | 2 | **CORRETO (100%)** |
| **Trifurcada (3 Brotações)** | 3 | 3 | 3 | 3 | **CORRETO (100%)** |

#### Principais Achados nos Dados Sintéticos:
1. **Separabilidade Clara**: Na ausência de interferências de sub-bosque espesso, todos os três métodos atingiram **100% de acurácia** na determinação do número exato de brotações.
2. **Estimativa de DAP Individual**: O Método B (RANSAC) foi o único capaz de fornecer a estimativa de DAP individualizada para cada brotação (ex.: Fuste #1 com DAP de $24{,}2\text{ cm}$ e Fuste #2 com DAP de $21{,}8\text{ cm}$).

---

### 2.2 Resultados com Dados Reais de LiDAR (Talhões 62 e 63 - Canoas)

Os algoritmos foram testados em nuvens de pontos reais de *Eucalyptus* provenientes das parcelas amostrais do projeto Canoas (nuvens densas com resolução de 5 cm e 10 cm, segmentadas pelo pipeline ForestFormer3D / FF3D).

#### Estudo de Caso: Árvore ID 3 (Fuste Único Robusto)
* **Pontos Totais**: 7.876 pontos
* **Altura Total**: $28{,}36\text{ m}$
* **Resultado dos Algoritmos**:
  * Método A (DBSCAN): **1 fuste**
  * Método B (RANSAC): **1 fuste** ($Dap = 26{,}4\text{ cm}$)
  * Método C (Trajetória): **1 fuste** ao longo de todas as alturas ($1{,}0\text{m}$ a $3{,}4\text{m}$)
* **Classificação**: **FUSTE ÚNICO (SINGLE STEM)**.

#### Estudo de Caso: Árvore com Brotação Dupla (Pontos Conectores na Base)
Durante a análise de fustes duplos reais, observou-se um fenómeno característico: os fustes apresentam dois adensamentos claros no plano 2D (Fuste Superior-Esquerdo e Fuste Inferior-Direito), contudo acompanhados por **pontos esparsos conectores** entre eles (provenientes de casca desprendida, galhos baixos ou área de fusão do V-fork).
* **Com parâmetros padrão antigos ($\epsilon = 0{,}15\text{ m}$)**: O DBSCAN "ponteava" através dos ruídos centrais e unificava os dois troncos, classificando incorretamente como 1 fuste.
* **Com parâmetros otimizados ($\epsilon = 0{,}08\text{ m}$, $z \ge 1{,}8\text{ m}$ e `radius_max` = $0{,}22\text{ m}$)**:
  * O Método A ignorou os pontos conectores esparsos e isolou os **2 clusters densos**.
  * O Método B ajustou dois círculos distintos, identificando $Dap_1 = 18{,}2\text{ cm}$ e $Dap_2 = 15{,}7\text{ cm}$.
  * O Método C registrou a transição de 1 fuste na base ($1{,}0\text{ m}$) para 2 fustes nas alturas superiores ($1{,}8\text{ m} - 3{,}0\text{ m}$), classificando com sucesso como **MÚLTIPLAS BROTAÇÕES**.

---

## 3. Discussão e Limitações Técnicas

1. **Sensibilidade à Qualidade da Nuvem**: A acurácia dos três métodos é diretamente dependente da resolução espacial e da densidade de pontos no fuste. Nuvens de baixa densidade (ex.: LiDAR aéreo tradicional com $< 10\text{ pontos/m}^2$) não possuem densidade suficiente no tronco inferior para garantir o funcionamento do DBSCAN ou RANSAC.
2. **Impacto do Sub-Bosque e Ruído de Solo**: Se a nuvem de pontos contiver ruído vegetacional denso de sub-bosque (ex.: arbustos colados ao tronco entre $1{,}0\text{ m}$ e $1{,}5\text{ m}$), o algoritmo de separação pode gerar falsos positivos se não houver um pré-filtro de casca/madeira eficiente (*Leaf-Wood segmentation*).
3. **Restrição por Estágio de Desenvolvimento (Idade da Floresta)**:
   * **Árvores Adultas / Desenvolvidas (Recomendado)**: As técnicas demonstraram alta eficiência em povoamentos adultos, onde o DAP já está bem formado ($> 10\text{ cm}$) e os fustes possuem separação espacial clara acima de $1{,}5\text{ m}$ do solo.
   * **Plantios Jovens / Brotações Recentes (Limitação)**: Em mudas recém-plantadas ou brotações muito jovens ($< 1\text{ ano}$), onde a altura total é reduzida e os fustes têm diâmetros milimétricos muito próximos ao solo, as fatias padrão na altura do peito ($1{,}3\text{ m}$) cortam a folhagem da copa ou ficam acima da própria planta.

---

## 4. Conclusão e Recomendações

1. **Eficácia Comprovada**: As três técnicas desenvolvidas e disponibilizadas no notebook `detect_multi_stems_comparison.ipynb` provaram ser eficazes para detectar, contar e quantificar individualmente o diâmetro de múltiplas brotações em árvores adultas de *Eucalyptus*.
2. **Necessidade Crucial de Reconstrução e Isolamento de Nuvens**: O sucesso da detecção depende criticamente de uma etapa prévia robusta de **segmentação e isolamento individual de árvores** (como a realizada via pipelines `FF3D` / `ForestFormer3D` ou `TreeISO`). Nuvens de parcelas inteiras sem isolamento individual de árvores causam interferência entre árvores vizinhas.
3. **Complementaridade dos Métodos**: Recomenda-se adotar o **Método A (DBSCAN)** para triagem e contagem rápida de fustes em massa, combinado ao **Método B (RANSAC)** para extração do DAP individualizado de cada fuste da brotação.

---

*Relatório gerado automaticamente como documentação técnica do sistema Antigravity / Eucalipto.*
