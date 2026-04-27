# Gêmeo Digital de Canal VLC - cVAE

Repositório público de pesquisa para um gêmeo digital orientado a dados de um
canal de Comunicação por Luz Visível (VLC), usando Autoencoders Variacionais
Condicionais (cVAE) e baselines generativos relacionados.

A tarefa central é aprender a distribuição condicional do canal

$$p(y \mid x, d, c)$$

a partir de medições I/Q sincronizadas, em que:

- `x` é a amostra de banda base transmitida
- `y` é a amostra recebida após o canal físico
- `d` é a distância LED-fotodetector
- `c` é a corrente de acionamento do LED

O objetivo é fidelidade distribucional, não apenas predição de média.

## Comece Por Aqui

Se você chegou neste repositório pelo GitHub e não sabe por onde iniciar,
use esta ordem:

1. [docs/BRANCH_GUIDE.md](docs/BRANCH_GUIDE.md) - para que serve cada branch pública
2. [PROJECT_STATUS.md](PROJECT_STATUS.md) - estado científico e do código atualmente
3. [docs/README.md](docs/README.md) - mapa da documentação
4. [docs/reference/PROTOCOL.md](docs/reference/PROTOCOL.md) - runner canônico de experimentos
5. [docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt](docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt) - fluxo fim a fim da aquisição ao treino

Membros internos que usam o servidor compartilhado também devem ler:

- [docs/active/INFRA_GUIDE.md](docs/active/INFRA_GUIDE.md) - SSH, usuários Unix, Docker, tmux, Git LFS
- [docs/active/MULTI_PC_WORKFLOW.md](docs/active/MULTI_PC_WORKFLOW.md) - worktrees, dois slots quentes e regras de operação multi-PC

Observação importante sobre o GitHub:

- o README exibido no site do GitHub muda conforme a branch selecionada
- a branch pública padrão atualmente é `main`
- se você quer a linha de pesquisa coordenada mais recente, consulte primeiro o guia de branches

## Qual Branch Devo Usar?

Para a maioria das pessoas, existem apenas três pontos de partida realmente relevantes:

| Branch | Use quando | Observações |
|---|---|---|
| `main` | você quer a página pública inicial e uma visão estável | branch padrão no GitHub |
| `feat/mdn-g5-recovery-explore-remote` | você quer a linha de pesquisa remota coordenada atualmente | melhor escolha padrão para colaboração ativa hoje |
| `release/cvae-online` | você quer um snapshot em estilo release em vez de experimentação ao vivo | mais conservadora que a linha ativa de pesquisa |

O mapa completo das branches públicas está em [docs/BRANCH_GUIDE.md](docs/BRANCH_GUIDE.md).

## Clone O Trabalho Ativo Atual

Para copiar o trabalho remoto ativo atual em um clone local novo:

```bash
git clone https://github.com/rodrigofmiranda/cVAe_2026.git
cd cVAe_2026
git fetch --all --prune
git switch feat/mdn-g5-recovery-explore-remote
git lfs install --local
git lfs pull
```

Sem `git lfs pull`, a árvore de dataset pode aparecer, mas os arquivos grandes
não estarão realmente presentes.

## Execução Rápida

O entrypoint canônico é:

```bash
python -m src.protocol.run \
  --dataset_root data/dataset_fullsquare_organized \
  --output_base outputs \
  --protocol configs/protocol_default.json
```

Para a família sequencial `seq_bigru_residual`, use também:

```bash
--no_data_reduction
```

porque a redução padrão em blocos balanceados quebra o contexto temporal.

## Resumo De Modelagem

Este repositório mantém múltiplas famílias de modelos sob um caminho compartilhado de
protocolo/treino:

- `concat` - cVAE original ponto a ponto
- `channel_residual` - decoder residual ponto a ponto
- `delta_residual` - família ponto a ponto com alvo residual
- `seq_bigru_residual` - família residual com consciência sequencial
- `legacy_2025_zero_y` - linha histórica de referência da era de notebooks

No dia a dia da pesquisa, a seleção de arquitetura geralmente é feita por
`arch_variant`, `grid_tag` ou `grid_preset`, sem abrir um repositório separado.

## Documentação

- [docs/BRANCH_GUIDE.md](docs/BRANCH_GUIDE.md) - mapa público de branches
- [docs/README.md](docs/README.md) - índice da documentação
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - estado atual do código e da ciência
- [docs/reference/PROTOCOL.md](docs/reference/PROTOCOL.md) - runner de protocolo e artefatos
- [docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt](docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt) - fluxo da aquisição ao treino
- [docs/reference/MODELING_ASSUMPTIONS.md](docs/reference/MODELING_ASSUMPTIONS.md) - racional de modelagem
- [docs/archive/reference/DIAGNOSTIC_CHECKLIST.md](docs/archive/reference/DIAGNOSTIC_CHECKLIST.md) - workflow diagnóstico histórico
- [docs/active/INFRA_GUIDE.md](docs/active/INFRA_GUIDE.md) - onboarding interno de servidor e guia de isolamento

## Estrutura Do Repositório

```text
configs/        Configuração de protocolo e grid
data/           Dataset armazenado com Git LFS
docker/         Build e runtime de container
docs/           Docs públicas, guias internos, arquivo histórico e referências
notebooks/      Notebooks exploratórios
outputs/        Artefatos de experimentos
scripts/        Auxiliares operacionais e de análise
src/            Código principal de treino, avaliação, protocolo e modelos
tests/          Testes unitários e de integração
```

## Para Novos Colaboradores

Se seu objetivo é entender o programa e reproduzir o trabalho atual:

1. Leia [docs/BRANCH_GUIDE.md](docs/BRANCH_GUIDE.md)
2. Faça clone do repositório e troque para `feat/mdn-g5-recovery-explore-remote`
3. Rode `git lfs pull`
4. Leia [PROJECT_STATUS.md](PROJECT_STATUS.md)
5. Leia [docs/reference/PROTOCOL.md](docs/reference/PROTOCOL.md)
6. Leia [docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt](docs/reference/VLC_DATA_FLOW_FROM_ACQUISITION_TO_TRAINING.txt)
7. Se estiver usando o servidor compartilhado do laboratório, siga [docs/active/INFRA_GUIDE.md](docs/active/INFRA_GUIDE.md)
