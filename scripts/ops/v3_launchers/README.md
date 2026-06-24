# V3 launchers (snapshots operacionais)

Snapshots dos scripts de lançamento/monitoramento dos experimentos V3 (E1/E2/E3,
cross-dist, modulações, seed-sweep), versionados para histórico/reprodutibilidade.

**Paths são machine-specific** (`/home/rodrigo/...`, box do rodrigo) — rodam como
estão nessa máquina; em outra, ajustar os caminhos. Não são chamados pelo código
do repo; documentam COMO os runs foram disparados (docker, env de determinismo,
ntfy, fila sequencial por teto de RAM). Ver `docs/CRITICAL_PROJECT_KNOWLEDGE.md` §2.
