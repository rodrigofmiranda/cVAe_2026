# GEMINI_PROMPTS.md

Templates de prompt para usar com Gemini ou outra IA complementar.

Principios praticos:

- mande contexto curto e task-focused
- use arquivos especificos em vez de "leia tudo"
- diga qual output voce espera
- diga explicitamente o que nao deve ser usado como criterio final
- para tarefas complexas, quebre em duas ou tres mensagens

## 1. Prompt de bootstrap curto

```text
Contexto:
- Repo: /workspace/2026/feat_delta_residual_adv
- Branch: feat/channel-residual-architecture
- Commit: eaef7f0
- Objetivo atual: avaliar se a arquitetura residual reduz o vies por regime visto no concat

Leia nesta ordem:
1. PROJECT_STATUS.md
2. TRAINING_PLAN.md
3. README.md
4. docs/PROTOCOL.md
5. docs/GEMINI_PLAYBOOK.md

Seu papel:
- atuar como revisor tecnico do experimento
- priorizar evidencia cientifica por regime, nao apenas loss

Regras:
- se houver conflito entre state_run.json e manifest/csvs, confie em manifest/csvs
- nao conclua nada com base so em train loss ou val loss

Quando terminar, me devolva:
- resumo de 10 linhas do projeto
- principais riscos tecnicos
- melhor proximo passo
```

## 2. Prompt para analisar um experimento fechado

```text
Analise o experimento:
- outputs/exp_20260313_153655

Arquivos a consultar:
- manifest.json
- global_model/tables/gridsearch_results.csv
- global_model/logs/training_history.json
- tables/summary_by_regime.csv
- tables/stat_fidelity_by_regime.csv

Perguntas:
1. qual foi o melhor grid?
2. o treino foi estavel?
3. em quais regimes o modelo foi melhor ou pior que o baseline?
4. ha evidencia real de twin global util?

Restricoes:
- nao use apenas val_loss para decidir
- se houver inconsistencia com state_run.json, trate state_run.json como secundario

Formato de saida:
- resumo executivo
- 5 achados principais
- 3 riscos ou duvidas
- recomendacao objetiva
```

## 3. Prompt para comparar concat vs residual

```text
Compare duas arquiteturas:

Arquitetura anterior:
- experimento de referencia: outputs/exp_20260313_153655
- arquitetura: concat

Arquitetura nova:
- linha de trabalho: channel_residual
- run principal atual: outputs/residual_small_global

Tarefa:
- diga quais metricas devem ser usadas para uma comparacao justa
- diga quais artefatos ainda faltam para concluir a comparacao
- antecipe quais sinais indicariam melhora real da residual

Prioridades:
- fidelidade por regime
- comparacao com baseline
- testes estatisticos

Nao faca:
- nao trate loss melhor como vitoria automatica
- nao compare treino exploratorio com validacao cientifica como se fossem a mesma coisa
```

## 4. Prompt para diagnosticar run ruim

```text
Quero diagnosticar se um run ruim e bug de pipeline ou limite do modelo.

Leia:
- docs/DIAGNOSTIC_CHECKLIST.md
- TRAINING_PLAN.md
- o log e os artefatos do run que eu indicar

Responda em ordem:
1. quais checks operacionais devem ser feitos primeiro
2. quais arquivos decidiram sua leitura
3. quais sinais apontam para bug
4. quais sinais apontam para limite do modelo
5. proximo experimento minimo que reduz a incerteza
```

## 5. Prompt para acompanhar um treino em andamento

```text
Analise apenas o estado atual do treino em andamento:
- log: outputs/residual_small_global.launch.log

Responda:
1. em qual grid e epoca o run esta
2. se ha sinais de instabilidade
3. se o scheduler de LR esta atuando
4. se existe algum motivo para abortar cedo

Restricao:
- nao conclua sobre qualidade cientifica final enquanto os CSVs e manifest finais nao existirem
```

## 6. Prompt para obter proxima acao recomendada

```text
Com base no estado atual do projeto e nos documentos:
- PROJECT_STATUS.md
- TRAINING_PLAN.md
- docs/GEMINI_PLAYBOOK.md

Me diga a proxima acao mais informativa entre:
- novo grid search
- comparacao residual vs concat
- protocolo global train_once_eval_all
- investigacao de bug

Quero:
- uma resposta curta
- justificativa tecnica
- risco principal
- criterio de sucesso
```
