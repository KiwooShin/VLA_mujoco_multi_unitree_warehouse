#!/usr/bin/env bash
# GPU/process watchdog: samples every 5 minutes, appends to ops/watchdog.log.
#
# Detects:
#   - halted/zombie python jobs under this repo (D/Z state, or no CPU movement)
#   - jobs burning CPU while GPU utilization stays 0% (CPU-instead-of-GPU)
#
# Emits "ALERT:" lines the orchestrator greps at every wakeup.
# GB10 note: nvidia-smi reports memory as [N/A] (unified memory), so checks
# key off utilization.gpu and the compute-apps process list only.

set -u
REPO="/home/kiwoos/work/VLA_mujoco_multi_unitree_warehouse"
LOG_DIR="${REPO}/ops"
LOG="${LOG_DIR}/watchdog.log"
STATE="${LOG_DIR}/watchdog_state"   # holds "pid cputime" pairs from last sample
INTERVAL=300
mkdir -p "${LOG_DIR}"

echo "$(date '+%F %T') watchdog started (pid $$, interval ${INTERVAL}s)" >> "${LOG}"

while true; do
  TS="$(date '+%F %T')"
  GPU_UTIL="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
  GPU_UTIL="${GPU_UTIL:--1}"
  APPS="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)"
  echo "${TS} gpu_util=${GPU_UTIL}% compute_apps=[$(echo "${APPS}" | tr '\n' ';')]" >> "${LOG}"

  # Python processes whose cwd or cmdline points at this repo.
  declare -A CUR=()
  while read -r PID CPUT STAT CMD; do
    [[ -z "${PID:-}" ]] && continue
    CUR[$PID]="${CPUT}"
    # D (uninterruptible) or Z (zombie) state -> halted
    if [[ "${STAT}" == *D* || "${STAT}" == *Z* ]]; then
      echo "${TS} ALERT: pid ${PID} state=${STAT} (halted?) cmd=${CMD:0:120}" >> "${LOG}"
    fi
    # High CPU but GPU idle -> possible CPU-instead-of-GPU job
    if [[ "${GPU_UTIL}" =~ ^[0-9]+$ ]] && (( GPU_UTIL < 5 )); then
      PCPU="$(ps -o %cpu= -p "${PID}" 2>/dev/null | tr -d ' ' | cut -d. -f1)"
      if [[ -n "${PCPU}" ]] && (( PCPU > 90 )); then
        echo "${TS} ALERT: pid ${PID} cpu=${PCPU}% while gpu_util=${GPU_UTIL}% (CPU-instead-of-GPU?) cmd=${CMD:0:120}" >> "${LOG}"
      fi
    fi
    # No CPU-time movement since last sample while allegedly running -> stalled
    if [[ -f "${STATE}" ]]; then
      PREV="$(grep "^${PID} " "${STATE}" 2>/dev/null | awk '{print $2}')"
      if [[ -n "${PREV}" && "${PREV}" == "${CPUT}" && "${STAT}" == *R* ]]; then
        : # R state with frozen cputime is contradictory; skip
      elif [[ -n "${PREV}" && "${PREV}" == "${CPUT}" ]]; then
        echo "${TS} ALERT: pid ${PID} cputime frozen at ${CPUT} since last sample (stalled?) cmd=${CMD:0:120}" >> "${LOG}"
      fi
    fi
  done < <(pgrep -af python | while read -r PID REST; do
      CWD="$(readlink "/proc/${PID}/cwd" 2>/dev/null)"
      if [[ "${CWD}" == ${REPO}* || "${REST}" == *VLA_mujoco_multi_unitree_warehouse* ]]; then
        CPUT="$(awk '{print $14+$15}' "/proc/${PID}/stat" 2>/dev/null)"
        STAT="$(ps -o stat= -p "${PID}" 2>/dev/null | tr -d ' ')"
        echo "${PID} ${CPUT:-0} ${STAT:-?} ${REST}"
      fi
    done)

  # Persist current sample for next-round stall comparison.
  : > "${STATE}"
  for P in "${!CUR[@]}"; do echo "${P} ${CUR[$P]}" >> "${STATE}"; done
  unset CUR; declare -A CUR=()

  sleep "${INTERVAL}"
done
