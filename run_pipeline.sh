#!/bin/bash
# Activate conda environment and run canonical pipeline

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "Ativando ambiente conda: euc"
conda activate euc

if [ $? -ne 0 ]; then
    echo "Erro ao ativar conda env 'euc'"
    echo "Tente: conda env list"
    exit 1
fi

echo "Ambiente ativado. Rodando pipeline canônico..."
cd /home/matheuspimenta/Jobs/Eucalipto/projeto_eucalipto

CONFIG_PATH="${1:-configs/treeiso_leafwood.yaml}"
python run_pipeline.py "$CONFIG_PATH"

exit $?
