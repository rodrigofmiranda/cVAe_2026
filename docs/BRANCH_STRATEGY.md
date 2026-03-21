# Branch Strategy

> Historical note: this document describes the older branch split.
> Current daily work is unified in `feat/seq-bigru-residual-cvae`, and
> experiments should now be separated primarily by `arch_variant`,
> `grid_tag`, or `grid_preset` rather than by switching branches.

## Objetivo

Separar claramente:

- a linha que deve virar uma versao online funcional do cVAE
- a linha da arquitetura residual como experimento
- a linha `seq_bigru_residual` como pesquisa em andamento para o digital twin

## Estado atual do repositorio

Branches locais:

- `main`
- `exp/refactor_architecture`
- `feat/channel-residual-architecture`
- `feat/seq-bigru-residual-cvae`

Branches remotas observadas:

- `origin/main`
- `origin/exp/refactor_architecture`
- `origin/feat/channel-residual-architecture`

## Leitura correta da arvore

A linha real de desenvolvimento nao esta no `main`.

Hoje, a progressao funcional do projeto e:

```text
main
└─ exp/refactor_architecture
   └─ feat/channel-residual-architecture
      └─ feat/seq-bigru-residual-cvae
         + mudancas locais ainda sem commit
```

Interpretacao:

- `main` nao representa o estado tecnico mais novo do projeto
- `exp/refactor_architecture` contem a base moderna do pipeline
- `feat/channel-residual-architecture` contem essa base moderna mais a variante residual
- `feat/seq-bigru-residual-cvae` foi aberta em cima da residual, mas o trabalho `seq_bigru_residual` ainda nao foi consolidado em commit

## O que cada linha representa

### `main`

Deve ser tratado hoje como historico publico antigo, nao como fonte de verdade da linha ativa.

### `exp/refactor_architecture`

Representa a base de engenharia que tornou o pipeline utilizavel:

- protocolo mais maduro
- suporte a `train_once_eval_all`
- melhorias de grid e diagnostico
- plots e relatorios mais fortes

Esta branch e hoje a melhor base para a sua "versao online funcional com cVAE".

Razoes:

- contem a base moderna do pipeline sem acoplar a estrategia residual
- representa melhor o cVAE "padrao" operacional
- evita misturar release online com uma tentativa arquitetural que ainda nao venceu cientificamente
- e mais limpa que a `feat/channel-residual-architecture` para servir de referencia publica inicial

### `feat/channel-residual-architecture`

Esta branch deve ser tratada como a linha da estrategia residual.

Razoes:

- herda o trabalho do `exp/refactor_architecture`
- adiciona a arquitetura `channel_residual`
- preserva o caminho antigo `concat`
- ja esta no remoto
- registra a tentativa residual sem perder o baseline anterior

Importante:

- nao significa que a arquitetura residual venceu cientificamente
- significa que esta branch guarda a segunda estrategia arquitetural do projeto

### `feat/seq-bigru-residual-cvae`

Esta branch deve ser tratada como linha de pesquisa.

Ela nao deve ser sua candidata primaria para versao online agora, porque:

- o trabalho ainda nao foi commitado
- a arquitetura ainda esta em tentativa cientifica
- o primeiro run completo falhou nos gates de validacao
- ainda ha risco de ajuste estrutural, novo grid e novas rodadas de diagnostico

## Recomendacao pratica

Separar o projeto em tres niveis:

### 1. Linha online funcional

Base recomendada:

- `exp/refactor_architecture`

Uso recomendado:

- publicar uma versao funcional do pipeline e do cVAE
- usar o cVAE point-wise atual como base publica inicial
- manter a linha residual separada, sem misturar com a release base

Racional:

- `concat` e o caminho mais conservador
- isso permite ter uma versao online util mesmo sem a arquitetura final do twin
- o nome e o historico ficam coerentes: release online de um lado, estrategias experimentais do outro

### 2. Linha experimental residual

Base:

- `feat/channel-residual-architecture`

Uso recomendado:

- manter como marco da tentativa residual
- comparar contra `concat`
- nao vender como solucao final do digital twin

### 3. Linha de pesquisa seq-bigru

Base:

- `feat/seq-bigru-residual-cvae`

Uso recomendado:

- consolidar os commits da implementacao `seq_bigru_residual`
- continuar grids, smoke runs e protocolo final
- tratar como branch de P&D, nao de release

## Estrategia de integracao recomendada

### Etapa A - Preservar a pesquisa atual

Na branch `feat/seq-bigru-residual-cvae`:

- commitar a implementacao local do `seq_bigru_residual`
- nao misturar isso com release

Objetivo:

- parar de depender de mudancas nao commitadas
- permitir retomada segura da pesquisa

### Etapa B - Criar uma branch de release online

Criar uma branch nova a partir de `exp/refactor_architecture`, por exemplo:

- `release/cvae-online`

Objetivo:

- ter uma linha clara para deploy/publicacao
- congelar uma base funcional sem depender da seq-bigru

### Etapa C - Definir o default online

Na linha online:

- usar `concat` como default se a prioridade for robustez operacional
- nao acoplar a residual como parte obrigatoria da release

Objetivo:

- evitar publicar como default uma arquitetura que cientificamente ainda nao fechou
- manter a leitura do projeto simples: release funcional separada das hipoteses de pesquisa

### Etapa D - Promocao futura

Somente depois:

- se a residual mostrar ganho robusto, ela pode virar default
- se a `seq_bigru_residual` realmente vencer cientificamente e operacionalmente, ela pode substituir a linha online numa fase posterior

## Decisao recomendada agora

Se o objetivo imediato e "ter uma versao online funcionando com cVAE mesmo sem a melhor arquitetura", a decisao mais segura e:

1. tratar `exp/refactor_architecture` como base online funcional
2. criar uma release branch em cima dela
3. deixar `feat/channel-residual-architecture` como a linha da estrategia residual
4. manter `feat/seq-bigru-residual-cvae` isolada como pesquisa do digital twin final

## Proximo passo operacional

Ordem recomendada:

1. commitar a WIP da `feat/seq-bigru-residual-cvae`
2. criar `release/cvae-online` a partir de `exp/refactor_architecture`
3. validar a release branch com testes e um smoke run curto
4. so depois decidir se ela sobe para `main` ou se `main` fica como historico e a release branch vira a referencia publica
