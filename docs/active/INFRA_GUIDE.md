# Guia de Infraestrutura - Servidor de IA (vlc-ia)

Linguagem simples. Se algo aqui parecer errado ou desatualizado, avise o Rodrigo.

---

## 1. Objetivo deste guia

Queremos que Rodrigo, Gabriele e Eduardo possam:

- entrar no mesmo servidor
- cada um com seu proprio usuario Unix
- cada um com seu proprio clone do projeto
- cada um com seu proprio container Docker
- rodar em paralelo sem misturar arquivos, `git`, logs ou testes

O modelo recomendado agora e este:

- `vlc` = usuario administrador/bootstrap
- `rodrigo`, `gabriele`, `eduardo` = usuarios de trabalho diario
- um clone do repo por usuario
- um container por usuario
- uma sessao `tmux` por usuario

---

## 2. Visao geral - arquitetura recomendada

```text
┌──────────────────────────────────────────────────────────────┐
│  MAQUINA FISICA  (PC com RTX 5090, 32 GB VRAM)              │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  PROXMOX VE                                            │  │
│  │                                                        │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  LXC "vlc-ia"  (o Linux acessado por SSH)        │  │  │
│  │  │                                                  │  │  │
│  │  │  Usuario admin: vlc                              │  │  │
│  │  │                                                  │  │  │
│  │  │  Usuario rodrigo                                 │  │  │
│  │  │    tmux "rodrigo"                                │  │  │
│  │  │    container cvae_rodrigo                        │  │  │
│  │  │    /home/rodrigo/cVAe_2026                       │  │  │
│  │  │    /workspace/cVAe_2026                          │  │  │
│  │  │                                                  │  │  │
│  │  │  Usuario gabriele                                │  │  │
│  │  │    tmux "gabriele"                               │  │  │
│  │  │    container cvae_gabriele                       │  │  │
│  │  │    /home/gabriele/cVAe_2026                      │  │  │
│  │  │    /workspace/cVAe_2026                          │  │  │
│  │  │                                                  │  │  │
│  │  │  Usuario eduardo                                 │  │  │
│  │  │    tmux "eduardo"                                │  │  │
│  │  │    container cvae_eduardo                        │  │  │
│  │  │    /home/eduardo/cVAe_2026                       │  │  │
│  │  │    /workspace/cVAe_2026                          │  │  │
│  │  │                                                  │  │  │
│  │  │  GPU RTX 5090 - compartilhada por todos          │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

O ponto central e este:

- o host continua sendo um so
- a GPU continua sendo uma so
- mas agora cada pessoa tem seu proprio usuario, sua propria `home`, seu proprio clone e seu proprio container

## 2.1. Layout atual do Rodrigo - 4 linhas, 2 slots ativos

No setup atual do Rodrigo, mantemos quatro worktrees nomeados, mas apenas dois
containers/sessoes `tmux` quentes ao mesmo tempo.

Worktrees:

- `mdn-anchor` -> `/home/rodrigo/cVAe_2026_mdn_anchor`
- `mdn-explore` -> `/home/rodrigo/cVAe_2026_mdn_explore`
- `shape` -> `/home/rodrigo/cVAe_2026_shape`
- `legacy2025` -> `/home/rodrigo/cVAe_2026_legacy2025`

Slots ativos:

- slot fixo: `anchor` -> branch `feat/mdn-g5-recovery`
- slot rotativo: `explore` ou `shape` ou `legacy2025`

Regra operacional:

- `anchor` permanece ligado para a linha cientifica estavel
- o segundo slot gira conforme a hipotese da vez
- nao manter os quatro containers abertos sem necessidade

Script canonico para girar o segundo slot:

- [switch_slot2.sh](/home/rodrigo/cVAe_2026_mdn_explore/scripts/ops/switch_slot2.sh)

Uso:

```bash
/home/rodrigo/cVAe_2026_mdn_explore/scripts/ops/switch_slot2.sh status
/home/rodrigo/cVAe_2026_mdn_explore/scripts/ops/switch_slot2.sh explore
/home/rodrigo/cVAe_2026_mdn_explore/scripts/ops/switch_slot2.sh shape
/home/rodrigo/cVAe_2026_mdn_explore/scripts/ops/switch_slot2.sh legacy2025
```

O script:

- preserva `anchor`
- desliga qualquer slot rotativo antigo
- sobe o novo `tmux` + container com o nome correto
- reaproveita `scripts/ops/run_tf25_gpu.sh` do proprio worktree alvo

---

## 3. O que cada camada faz (e o que NAO faz)

| Camada | Para que serve | O que NAO faz |
|--------|----------------|---------------|
| Proxmox | Gerencia o LXC | Nao participa do dia a dia |
| LXC `vlc-ia` | Host Linux - SSH, Docker, tmux, arquivos | Nao divide a GPU |
| Usuario Unix proprio | Isola `home`, `tmux`, `.ssh`, permissoes e clones | Nao isola o uso da GPU |
| Docker | Isola processo, libs e execucao | Nao transforma a GPU em recurso exclusivo |
| Clone Git proprio | Separa branch, `git status`, outputs e checkpoints | Nao protege se a pessoa ganhar acesso root/sudo |
| GPU | Roda treino/inferencia | Nao se divide sozinha |

---

## 4. Qual e o papel do usuario `vlc`?

O `vlc` passa a ser o usuario de administracao inicial.
Use `vlc` para:

- criar usuarios novos
- ajustar grupos
- instalar chave SSH inicial do usuario
- migrar pastas antigas
- manutencao do host

**Nao use `vlc` como ambiente diario de desenvolvimento**, a nao ser em manutencao pontual.

Importante:

- `vlc` nao e `root`
- mas `vlc` tem `sudo`
- quem conhece a senha do `vlc` consegue fazer administracao

---

## 5. Como cada pessoa cria seu proprio usuario Unix

Primeiro acesso:

```bash
ssh vlc@IP_DO_SERVIDOR
```

Depois, criar o usuario. Exemplo para Rodrigo:

```bash
sudo adduser rodrigo
```

Depois, criar o usuario. Exemplo para Gabriele:

```bash
sudo adduser gabriele
```

Exemplo para Eduardo:

```bash
sudo adduser eduardo
```

O `adduser` vai:

- criar `/home/rodrigo`, `/home/gabriele` ou `/home/eduardo`
- criar o grupo do proprio usuario
- pedir a senha do novo usuario

Na pratica, a `home` nasce com permissao privada do Linux, tipicamente:

```bash
/home/usuario   -> 750
```

Isso significa:

- o proprio usuario entra normalmente
- outro usuario comum nao entra normalmente
- um administrador com `sudo` consegue acessar

Importante:

- **nao precisa criar uma subpasta repetindo o nome do usuario**
- exemplo errado: `/home/gabriele/GABRIELE`
- exemplo certo: usar a propria `home` e o clone dentro dela, como `/home/gabriele/cVAe_2026`

### 5.1. Dar acesso ao Docker

Se a pessoa vai subir o proprio container:

```bash
sudo usermod -aG docker rodrigo
sudo usermod -aG docker gabriele
sudo usermod -aG docker eduardo
```

Depois disso, a pessoa precisa sair e entrar de novo no SSH para o grupo valer.

> Aviso importante: entrar no grupo `docker` da muita autonomia. Na pratica, isso e quase poder de administrador do host.

### 5.2. NAO adicionar em `sudo` por padrao

Nao faca isto, a nao ser que a pessoa tambem seja administradora do servidor:

```bash
sudo usermod -aG sudo gabriele
sudo usermod -aG sudo eduardo
```

O recomendado e:

- `vlc` com `sudo`
- usuarios de trabalho sem `sudo`

### 5.3. Configurar SSH do proprio usuario

O ideal e cada pessoa usar sua propria chave publica.

Exemplo para Gabriele:

```bash
sudo install -d -m 700 -o gabriele -g gabriele /home/gabriele/.ssh
sudo nano /home/gabriele/.ssh/authorized_keys
sudo chown gabriele:gabriele /home/gabriele/.ssh/authorized_keys
sudo chmod 600 /home/gabriele/.ssh/authorized_keys
```

Exemplo para Eduardo:

```bash
sudo install -d -m 700 -o eduardo -g eduardo /home/eduardo/.ssh
sudo nano /home/eduardo/.ssh/authorized_keys
sudo chown eduardo:eduardo /home/eduardo/.ssh/authorized_keys
sudo chmod 600 /home/eduardo/.ssh/authorized_keys
```

Depois disso, cada um passa a entrar assim:

```bash
ssh gabriele@IP_DO_SERVIDOR
ssh eduardo@IP_DO_SERVIDOR
```

### 5.4. Verificar se ficou certo

Entre com o usuario novo e rode:

```bash
whoami
id
groups
pwd
```

O esperado para Gabriele e algo na linha de:

```bash
gabriele
uid=... gid=... groups=...
/home/gabriele
```

Se o passo `usermod -aG docker` ja tiver sido feito e a pessoa tiver feito login de novo,
ai o `docker` tambem aparece nos grupos.

---

## 6. Como cada usuario prepara o GitHub

Agora cada pessoa tem sua propria `home`.
Entao cada pessoa deve ter seu proprio acesso ao GitHub dentro do proprio usuario.

### 6.1. Opcao recomendada - chave SSH propria

Ja logado como o proprio usuario:

```bash
ssh gabriele@IP_DO_SERVIDOR
ssh-keygen -t ed25519 -C "gabriele@vlc-ia"
cat ~/.ssh/id_ed25519.pub
```

Copie essa chave publica e adicione no GitHub da pessoa.

Depois teste:

```bash
ssh -T git@github.com
```

### 6.2. Repo atual

Repo atual:

```bash
git@github.com:rodrigofmiranda/cVAe_2026.git
```

Observacao pratica importante:

- este repo pode ser clonado por HTTPS sem chave do GitHub, porque o acesso de leitura esta funcionando publicamente
- exemplo:

```bash
git clone https://github.com/rodrigofmiranda/cVAe_2026.git /home/gabriele/cVAe_2026
```

- isso resolve o clone inicial rapido
- para `push`, o ideal continua sendo configurar a chave SSH ou outro metodo de autenticacao da propria pessoa

### 6.3. Clonar o repo

Logado como Gabriele:

```bash
git clone git@github.com:rodrigofmiranda/cVAe_2026.git /home/gabriele/cVAe_2026
```

Logado como Eduardo:

```bash
git clone git@github.com:rodrigofmiranda/cVAe_2026.git /home/eduardo/cVAe_2026
```

Depois do clone, baixe os arquivos grandes do dataset com Git LFS:

```bash
git -C /home/gabriele/cVAe_2026 lfs install --local
git -C /home/gabriele/cVAe_2026 lfs pull

git -C /home/eduardo/cVAe_2026 lfs install --local
git -C /home/eduardo/cVAe_2026 lfs pull
```

Sem esse passo, a estrutura de `data/` aparece no clone, mas os arquivos grandes
nao sao baixados de verdade. O sintoma tipico no treino e:

```bash
ValueError: Nenhum dataset carregado com sucesso.
```

### 6.4. Configurar identidade Git

Logado como Gabriele:

```bash
git -C /home/gabriele/cVAe_2026 config user.name "Gabriele"
git -C /home/gabriele/cVAe_2026 config user.email "gabriele@exemplo.com"
```

Logado como Eduardo:

```bash
git -C /home/eduardo/cVAe_2026 config user.name "Eduardo"
git -C /home/eduardo/cVAe_2026 config user.email "eduardo@exemplo.com"
```

---

## 7. Se ja existe pasta antiga dentro de `/home/vlc`

Hoje o servidor ja tem historico em pastas como:

```bash
/home/vlc/GABRIELA
/home/vlc/EDUARDO
```

Se voce quiser migrar isso para o usuario novo:

### Gabriele

```bash
sudo rsync -a /home/vlc/GABRIELA/ /home/gabriele/
sudo chown -R gabriele:gabriele /home/gabriele
```

### Eduardo

```bash
sudo rsync -a /home/vlc/EDUARDO/ /home/eduardo/
sudo chown -R eduardo:eduardo /home/eduardo
```

Se a pasta antiga estiver vazia ou nao for mais necessaria, pode simplesmente ignorar e fazer clone novo na `home` correta.

---

## 8. Como subir o container do proprio usuario

Agora o `docker run` deve ser feito logado no proprio usuario.

O padrao e:

- container com nome da pessoa
- mount da `home` da propria pessoa
- `-u $(id -u):$(id -g)` para os arquivos nascerem com o dono correto

### 8.1. Exemplo - Gabriele

Logado como `gabriele`:

```bash
tmux new -s gabriele

docker rm -f cvae_gabriele >/dev/null 2>&1 || true

docker run --rm -it \
  --name cvae_gabriele \
  --runtime=nvidia \
  --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -u $(id -u):$(id -g) \
  -v /home/gabriele/cVAe_2026:/workspace/cVAe_2026 \
  -w /workspace/cVAe_2026 \
  --entrypoint bash \
  vlc/tf25-gpu-ready:1
```

### 8.2. Exemplo - Eduardo

Logado como `eduardo`:

```bash
tmux new -s eduardo

docker rm -f cvae_eduardo >/dev/null 2>&1 || true

docker run --rm -it \
  --name cvae_eduardo \
  --runtime=nvidia \
  --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -u $(id -u):$(id -g) \
  -v /home/eduardo/cVAe_2026:/workspace/cVAe_2026 \
  -w /workspace/cVAe_2026 \
  --entrypoint bash \
  vlc/tf25-gpu-ready:1
```

### 8.3. Exemplo - Rodrigo

Logado como `rodrigo`:

```bash
tmux new -s rodrigo

docker rm -f cvae_rodrigo >/dev/null 2>&1 || true

docker run --rm -it \
  --name cvae_rodrigo \
  --runtime=nvidia \
  --security-opt apparmor=unconfined \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -u $(id -u):$(id -g) \
  -v /home/rodrigo/cVAe_2026:/workspace/cVAe_2026 \
  -w /workspace/cVAe_2026 \
  --entrypoint bash \
  vlc/tf25-gpu-ready:1
```

### 8.4. Por que isso isola melhor?

- cada pessoa tem sua propria `home`
- cada pessoa tem seu proprio `.ssh`
- cada pessoa tem seu proprio clone
- cada pessoa tem sua propria sessao `tmux`
- cada pessoa roda seu proprio container

---

## 9. Fluxo diario recomendado

### Primeira vez

1. entrar como `vlc`
2. criar o usuario Unix
3. instalar a chave SSH do usuario
4. fazer login com o usuario novo
5. configurar GitHub
6. clonar o repo
7. subir o container

### Depois disso, no dia a dia

Gabriele:

```bash
ssh gabriele@IP_DO_SERVIDOR
tmux attach -t gabriele
```

ou, se ainda nao houver sessao:

```bash
ssh gabriele@IP_DO_SERVIDOR
tmux new -s gabriele
```

Eduardo:

```bash
ssh eduardo@IP_DO_SERVIDOR
tmux attach -t eduardo
```

Rodrigo:

```bash
ssh rodrigo@IP_DO_SERVIDOR
tmux attach -t rodrigo
```

Para sair sem matar a sessao `tmux`:

```bash
Ctrl+B, depois D
```

---

## 10. O que fica isolado e o que continua compartilhado

| Item | Isolado? | Como funciona |
|------|----------|---------------|
| Usuario SSH | Sim | Cada pessoa entra com seu proprio login |
| `home` do Linux | Sim | `/home/gabriele`, `/home/eduardo`, `/home/rodrigo` |
| `.ssh` | Sim | Cada um tem sua propria chave |
| Sessao `tmux` | Sim | Sessao por usuario |
| Clone do repo | Sim | Cada um com sua propria pasta |
| Testes e processos Python | Sim | Cada um roda no proprio container |
| Alteracoes temporarias do container | Sim | Valem so naquele container em execucao |
| GPU | Nao | Continua compartilhada |
| Uso do daemon Docker | Nao totalmente | Quem entra no grupo `docker` ganha muita autonomia |

Resumo honesto:

- **usuarios Unix separados melhoram muito o isolamento**
- **mas usuario no grupo `docker` ainda tem poder alto no host**

Se um dia voces quiserem isolamento ainda mais forte, o proximo passo seria usar rootless Docker, Podman, ou separar por LXC/VM.

---

## 11. O que NAO fazer

Nao faca isto no dia a dia:

```bash
ssh vlc@IP_DO_SERVIDOR
```

para trabalhar normalmente no projeto.

Use `vlc` so para administracao inicial.

Tambem nao faca isto:

```bash
docker exec -it cvae_tf25_gpu bash
```

para todo mundo trabalhar no mesmo container antigo.

E tambem nao faca isto:

```bash
-v /home:/home
```

porque isso entrega as homes de todos para dentro do container.

Por fim, nao compartilhe o mesmo clone do repo entre duas pessoas.

---

## 12. ATENCAO: a GPU continua compartilhada

```text
         ┌─────────────────────────────┐
         │   RTX 5090 - 32 GB VRAM     │
         │                              │
         │   Rodrigo:   usa uma parte   │
         │   Gabriele:  usa uma parte   │
         │   Eduardo:   usa uma parte   │
         │                              │
         │   Se a soma passar de 32 GB  │
         │   algum treino pode cair     │
         └─────────────────────────────┘
```

Antes de treinar:

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Regra de ouro:

- treino pesado: avisar antes
- treino leve: pode dividir, mas monitorar
- so editar/testar CPU: sem problema

---

## 13. Como se localizar e confirmar onde voce esta

Se bater duvida, rode estes comandos:

```bash
whoami
id
groups
pwd
hostnamectl
systemd-detect-virt
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
tmux list-sessions
```

Leitura mental:

```text
SSH no proprio usuario -> tmux do proprio usuario -> container do proprio usuario -> /workspace/cVAe_2026
```

No host atual, o esperado continua sendo:

- `hostnamectl` -> `vlc-ia`
- `systemd-detect-virt` -> `lxc`

---

## 14. Comandos uteis do dia a dia

```bash
# Ver quem esta conectado
who

# Ver grupos do usuario atual
groups

# Ver sessoes tmux visiveis no usuario atual
tmux list-sessions

# Ver containers ativos
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'

# Ver uso da GPU
nvidia-smi

# Atualizar o repo do usuario atual
git -C ~/cVAe_2026 pull

# Ver o usuario atual
whoami
```

---

*Ultima atualizacao: 2026-03-31*
