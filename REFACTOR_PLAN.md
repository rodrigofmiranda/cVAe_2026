Quero profissionalizar este repositório de pesquisa (ML para modelagem de canal VLC com cVAE). Hoje quase toda a lógica está concentrada em dois scripts monolíticos:

src/training/cvae_TRAIN_documented.py (treino + grid + logging + saving + plots)

src/evaluation/analise_cvae_reviewed.py (avaliação + métricas + plots + leitura do state_run)

Objetivo: refatorar para uma arquitetura modular, reprodutível e “publicável”, mantendo o comportamento e outputs atuais (mesmas pastas, nomes e JSONs), mas separando responsabilidades. Evitar mudanças de algoritmo — foco em engenharia/estrutura e mínima alteração funcional.

Requisitos gerais

Não quebrar paths do projeto:

DATASET_ROOT aponta para data/dataset_fullsquare_organized

OUTPUT_BASE aponta para outputs/

Manter estrutura outputs/run_YYYYMMDD_HHMMSS/{models,logs,plots,tables,state_run.json}

Manter o dataset e outputs fora do versionamento normal (já usamos Git LFS para data/).

Criar uma API interna clara (funções + módulos), com type hints e docstrings.

Introduzir CLI mínima (argparse) para treino/avaliação:

python -m src.training.train --config configs/train.yaml (ou equivalente)

python -m src.evaluation.evaluate --run_dir outputs/run_xxx

Entregáveis (modularização)

Crie/organize os seguintes módulos, movendo código do monolito para cada um:

src/config/

defaults.py (valores padrão, nomes de chaves)

io.py (load/save config YAML/JSON)

schema.py (dataclasses/pydantic opcional; pelo menos dataclass)

src/data/

manter channel_dataset.py

adicionar loading.py (discover experiments, load arrays, build inventories)

adicionar splits.py (split por experimento, sem shuffle global)

adicionar normalization.py (peak norm, power norm, sync rules; carregar do state_run)

src/models/

cvae.py (definição do encoder/decoder/prior e forward)

losses.py (recon loss, KL, free-bits, beta schedule)

callbacks.py (early stop, reduce LR, logging callback)

sampling.py (reparam trick, prior sampling)

src/training/

train.py (entrypoint CLI, chama engine)

engine.py (loop de treino: build model, compile, fit, save artifacts)

gridsearch.py (gera combinações e executa runs, registra results.csv)

logging.py (state_run.json writer, metrics writer, run_dir creation)

src/evaluation/

evaluate.py (entrypoint CLI: recebe --run_dir e gera outputs)

metrics.py (EVM, SNR, KL diagnostics, active dims)

plots.py (overlay, histos, scatter, etc.)

report.py (salvar tabelas .csv/.json e sumarizar)

scripts/

manter train.sh, gridsearch.sh, reproduce_run.sh como wrappers, mas atualizar para chamar os novos entrypoints (python -m ...), mantendo compatibilidade.

Compatibilidade (crítico)

Garantir que state_run.json continue sendo gerado com as mesmas chaves essenciais (normalization, data_split, hyperparams, run metadata).

Garantir que analise consiga rodar em runs antigos (backward compatibility): se faltar alguma chave no state_run.json, usar defaults.

Qualidade de código

Inserir pyproject.toml com ferramentas básicas (opcional) ou pelo menos requirements.txt já existe.

Adicionar README já está ok; não mexer nele agora.

Evitar dependências pesadas novas (pydantic só se realmente necessário).

Estratégia de refatoração (para minimizar risco)

Criar módulos novos e mover funções “sem alterar lógica”.

Manter cvae_TRAIN_documented.py inicialmente como wrapper chamando o novo train.py.

Manter analise_cvae_reviewed.py inicialmente como wrapper chamando evaluate.py.

Só depois, reduzir os wrappers.

Critérios de aceite

Treino roda e gera outputs/run_* idêntico (ou extremamente próximo) ao atual.

Avaliação roda em runs existentes.

Novo CLI funciona:

python -m src.training.train

python -m src.evaluation.evaluate --run_dir outputs/run_...

Código mais legível, modular e testável.

Comece implementando a estrutura de pastas, movendo funções de IO/log/run_dir/state_run, depois data loading/split/normalization, depois model/loss, e por fim os entrypoints.