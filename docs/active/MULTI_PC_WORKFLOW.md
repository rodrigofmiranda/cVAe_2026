# Guia De Operacao Multi-PC

Este documento define a organizacao operacional para usar o projeto em mais de
um PC sem misturar branch, outputs, `tmux` ou containers.

Use este guia junto com:

- [INFRA_GUIDE.md](INFRA_GUIDE.md)
- [switch_slot2.sh](../../scripts/ops/switch_slot2.sh)

## 1. Objetivo

Queremos quatro linhas registradas:

- `mdn-return`
- `mdn-explore`
- `shape`
- `legacy2025`

Mas apenas dois slots quentes por maquina:

- slot fixo: `return`
- slot rotativo: `explore` ou `shape` ou `legacy2025`

## 2. Layout canonico

No host do Rodrigo, o layout padrao e este:

- repo neutro:
  - `/home/rodrigo/cVAe_2026`
  - mantido em `detached HEAD`
  - nao usar para trabalho diario
- worktree `mdn-return`:
  - `/home/rodrigo/cVAe_2026_mdn_return`
  - branch `research/mdn-return-20260416`
- worktree `mdn-explore`:
  - `/home/rodrigo/cVAe_2026_mdn_explore`
  - branch `feat/mdn-g5-recovery-explore-remote`
- worktree `shape`:
  - `/home/rodrigo/cVAe_2026_shape`
  - branch `feat/probabilistic-shaping-nonlinearity`
- worktree `legacy2025`:
  - `/home/rodrigo/cVAe_2026_legacy2025`
  - branch `feat/pointwise-2025-revival`

Slots quentes:

- `return` -> container `cvae_return`
- `explore` -> container `cvae_explore`
- `shape` -> container `cvae_shape`
- `legacy2025` -> container `cvae_legacy2025`

Observacao:

- `return` deve permanecer ligado quando a linha MDN return for a referencia ativa
- o segundo slot gira conforme a hipotese da vez

## 3. Regras De Organizacao

- Uma branch por worktree. Nao troque de branch dentro de um worktree nomeado.
- Nao use `/home/rodrigo/cVAe_2026` para editar, treinar ou comitar.
- Nao mantenha quatro containers abertos sem necessidade. Mantenha so dois
  slots quentes por maquina.
- Codigo e docs sincronizam por `git`. `outputs/` nao sincroniza por `git`.
- Nunca assuma que um `exp_...` existe nos dois PCs. Se precisar retomar um
  experimento em outra maquina, copie os artefatos explicitamente.
- Sempre registre em notas ou mensagem curta:
  - qual PC
  - qual worktree
  - qual branch
  - qual `exp_...`
- Use nomes canonicos para `tmux` e container. Evite nomes genericos como
  `rodrigo`, `rodrigo2` ou `tf25_gpu` para linhas cientificas.

## 4. Quando Usar Cada Linha

- `mdn-return`:
  - reproducao
  - comparacao com a melhor linha estavel
  - reruns serios
- `mdn-explore`:
  - novas hipoteses dentro da familia MDN/seq
- `shape`:
  - mudanca conceitual de hipotese para probabilistic shaping / nonlinearity
- `legacy2025`:
  - comparacao historica com a familia point-wise 2025

## 5. Preparar Um PC Novo

Clone inicial:

```bash
cd /home/rodrigo
git clone git@github.com:rodrigofmiranda/cVAe_2026.git
cd /home/rodrigo/cVAe_2026
git fetch --all --prune
git lfs install --local
git lfs pull
git switch --detach
```

Criar os worktrees:

```bash
git -C /home/rodrigo/cVAe_2026 worktree add /home/rodrigo/cVAe_2026_mdn_explore feat/mdn-g5-recovery-explore-remote
git -C /home/rodrigo/cVAe_2026 worktree add /home/rodrigo/cVAe_2026_mdn_return research/mdn-return-20260416
git -C /home/rodrigo/cVAe_2026 worktree add /home/rodrigo/cVAe_2026_shape feat/probabilistic-shaping-nonlinearity
git -C /home/rodrigo/cVAe_2026 worktree add -b feat/pointwise-2025-revival /home/rodrigo/cVAe_2026_legacy2025 origin/feat/pointwise-2025-revival
```

Subir os dois slots padrao:

```bash
CVAE_TF25_TMUX_SESSION=return \
CVAE_TF25_CONTAINER_NAME=cvae_return \
bash /home/rodrigo/cVAe_2026_mdn_return/scripts/ops/run_tf25_gpu.sh

CVAE_TF25_TMUX_SESSION=explore \
CVAE_TF25_CONTAINER_NAME=cvae_explore \
bash /home/rodrigo/cVAe_2026_mdn_explore/scripts/ops/run_tf25_gpu.sh
```

## 6. Atualizar Tudo Em Um PC Ja Preparado

Do repo neutro:

```bash
git -C /home/rodrigo/cVAe_2026 fetch --all --prune
```

Atualizar os quatro worktrees:

```bash
git -C /home/rodrigo/cVAe_2026_mdn_return pull --ff-only
git -C /home/rodrigo/cVAe_2026_mdn_explore pull --ff-only
git -C /home/rodrigo/cVAe_2026_shape pull --ff-only
git -C /home/rodrigo/cVAe_2026_legacy2025 pull --ff-only
```

Garantir os objetos LFS:

```bash
git -C /home/rodrigo/cVAe_2026_mdn_explore lfs pull
```

Se quiser conferir o estado dos slots:

```bash
/home/rodrigo/cVAe_2026_mdn_explore/scripts/ops/switch_slot2.sh status
```

## 7. Girar O Segundo Slot

O `return` fica como referencia fixa. O segundo slot e trocado pelo script:

```bash
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/switch_slot2.sh explore
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/switch_slot2.sh shape
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/switch_slot2.sh legacy2025
```

O script:

- preserva `return`
- derruba qualquer slot rotativo antigo
- sobe o novo `tmux` + container com nomes canonicos

## 8. Politica De Outputs

`outputs/` e local de cada maquina.

Isso significa:

- `git pull` nao traz `outputs/`
- um experimento treinado no PC A nao aparece magicamente no PC B
- para retomar um experimento em outro PC, copie o diretorio explicitamente

Quando for necessario mover um run entre maquinas, copie pelo menos:

- `outputs/exp_.../manifest.json`
- `outputs/exp_.../train/`
- `outputs/exp_.../eval/` se quiser manter as avaliacoes prontas
- `outputs/_launch_logs/` apenas se os logs forem relevantes

Se a copia nao for feita, considere o experimento como local daquela maquina.

## 9. Checklist Antes De Rodar No Outro PC

Antes de iniciar um treino novo:

```bash
hostname
git -C /home/rodrigo/cVAe_2026_mdn_explore fetch --all --prune
git -C /home/rodrigo/cVAe_2026_mdn_return status -sb
git -C /home/rodrigo/cVAe_2026_mdn_explore status -sb
/home/rodrigo/cVAe_2026_mdn_return/scripts/ops/switch_slot2.sh status
tmux ls
docker ps
```

E confirme:

- o PC certo esta sendo usado
- a linha certa esta no slot certo
- o worktree da linha esta limpo ou com alteracoes intencionais
- o experimento que voce quer continuar realmente existe naquele host

## 10. Regra De Ouro

Se a duvida for entre simplicidade e paralelismo, prefira simplicidade:

- `return` sempre previsivel
- apenas um segundo slot rotativo por maquina
- worktree nomeado para cada linha cientifica
- `outputs` tratados como locais, a menos que sejam copiados de forma explicita
