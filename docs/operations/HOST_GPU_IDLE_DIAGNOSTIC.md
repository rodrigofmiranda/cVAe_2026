# Host GPU Idle Diagnostic

Objetivo: diagnosticar o problema em que a GPU ou o acesso via Docker para de funcionar somente depois que o PC fica parado por muitas horas, sem forcar o erro manualmente.

Regra principal:

- Quando o problema aparecer pela manha, nao reinicie `docker`, nao rode `systemctl restart docker` e nao reinicie a maquina antes da coleta.

## 1. Coleta antes de deixar o PC parado

Rode uma vez, antes de dormir:

```bash
mkdir -p ~/gpu_idle_debug

date -Is | tee ~/gpu_idle_debug/start_time.txt

nvidia-smi | tee ~/gpu_idle_debug/nvidia_before.txt
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi | tee ~/gpu_idle_debug/docker_before.txt

systemctl list-unit-files --state=masked | grep -E 'sleep|suspend|hibernate' | tee ~/gpu_idle_debug/masked_sleep_units.txt
sudo systemd-analyze cat-config systemd/logind.conf | grep -E 'IdleAction|HandleLidSwitch|HandleLidSwitchExternalPower|HandleLidSwitchDocked|HandleSuspendKey|HandleHibernateKey' | tee ~/gpu_idle_debug/logind_effective.txt

gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type | tee ~/gpu_idle_debug/gsettings_ac_type.txt
gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout | tee ~/gpu_idle_debug/gsettings_ac_timeout.txt
gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type | tee ~/gpu_idle_debug/gsettings_battery_type.txt
gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout | tee ~/gpu_idle_debug/gsettings_battery_timeout.txt
gsettings get org.gnome.desktop.session idle-delay | tee ~/gpu_idle_debug/gsettings_idle_delay.txt
gsettings get org.gnome.desktop.screensaver lock-enabled | tee ~/gpu_idle_debug/gsettings_lock_enabled.txt
```

## 2. Coleta na manha seguinte, antes de mexer em qualquer coisa

Se o problema aparecer, rode nesta ordem:

```bash
date -Is | tee ~/gpu_idle_debug/fail_time.txt

nvidia-smi | tee ~/gpu_idle_debug/nvidia_after.txt
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi | tee ~/gpu_idle_debug/docker_after.txt
```

```bash
last -x | head -n 30 | tee ~/gpu_idle_debug/last_x.txt
```

```bash
sudo journalctl -b --since "yesterday 22:00" --no-pager | grep -Ei 'suspend|resume|sleep|hibernate|systemd-logind|gdm|gnome-shell' | tail -n 200 | tee ~/gpu_idle_debug/power_events.txt
```

```bash
sudo journalctl -k -b --since "yesterday 22:00" --no-pager | grep -Ei 'NVRM|Xid|nvidia|nvidia-uvm|drm|pcie|rm_init_adapter|fallen off the bus' | tail -n 200 | tee ~/gpu_idle_debug/gpu_kernel_events.txt
```

```bash
sudo journalctl -u docker -b --since "yesterday 22:00" --no-pager | tail -n 200 | tee ~/gpu_idle_debug/docker_journal.txt
sudo journalctl -u nvidia-persistenced -b --since "yesterday 22:00" --no-pager | tail -n 200 | tee ~/gpu_idle_debug/nvidia_persistenced_journal.txt
```

## 3. Como interpretar

- Se `nvidia-smi` falhar no host e no Docker: o problema esta no host, driver, GPU ou energia.
- Se `nvidia-smi` funcionar no host, mas falhar no Docker: o problema esta no runtime Docker/NVIDIA.
- Se `power_events.txt` mostrar `suspend` ou `resume`: a maquina ainda entrou em algum caminho de idle/suspensao.
- Se `gpu_kernel_events.txt` mostrar `Xid`, `NVRM` ou `fallen off the bus`: houve erro real de driver/GPU.
- Se nao houver `suspend/resume` nem `Xid`, mas o Docker falhar: forte suspeita no stack Docker/NVIDIA apos longo idle.

## 4. O que enviar depois

Quando a falha acontecer, separar estes arquivos:

- `~/gpu_idle_debug/nvidia_after.txt`
- `~/gpu_idle_debug/docker_after.txt`
- `~/gpu_idle_debug/power_events.txt`
- `~/gpu_idle_debug/gpu_kernel_events.txt`
- `~/gpu_idle_debug/docker_journal.txt`

Com isso da para distinguir se o problema e:

- suspensao/idle real do host
- erro do driver NVIDIA
- problema do runtime do Docker com GPU
- ou algum efeito apenas apos varias horas de ociosidade
