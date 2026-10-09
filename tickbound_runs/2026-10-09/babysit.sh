#!/usr/bin/env bash
# Local babysitter for one rented vast.ai box (9 Oct 2026). Usage: VAST_API_KEY=... babysit.sh ID NAME ONBOX_SCRIPT CAP_MIN
# Waits for SSH, uploads tickbound.tgz + the on-box script + backstop.sh, starts the on-box script detached, polls for
# /root/DONE every 60 s (copying the log tail to out_NAME/tail.txt), then fetches /root/res.tgz into out_NAME/ and destroys
# the instance. CAP_MIN minutes after this script starts it packs whatever /root/res holds, fetches it and destroys the
# instance anyway. The API key is read from the environment only, never written anywhere.
set -u
ID=$1; NAME=$2; ONBOX=$3; CAP=$4
D=$(cd "$(dirname "$0")" && pwd); OUT=$D/out_$NAME; mkdir -p "$OUT"
LOG=$OUT/babysit.log
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
DEAD=$(( $(date +%s) + CAP * 60 ))
SSHO=(-i ~/.ssh/msc_gpu -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o LogLevel=ERROR -o BatchMode=yes)
H=; P=
rsh() { ssh "${SSHO[@]}" -p "$P" "root@$H" "$@"; }
past_cap() { [ "$(date +%s)" -ge "$DEAD" ]; }
listed() {   # yes / no / empty when the listing failed
  vastai show instances --api-key "$VAST_API_KEY" --raw 2>/dev/null | python -c "
import sys, json
d = json.load(sys.stdin)
print('yes' if any(i.get('id') == $ID for i in d) else 'no')" 2>/dev/null
}
destroy() {
  for i in 1 2 3 4 5 6; do
    vastai destroy instance "$ID" -y --api-key "$VAST_API_KEY" >> "$LOG" 2>&1
    sleep 15
    if [ "$(listed)" = no ]; then log "instance $ID destroyed"; return 0; fi
  done
  log "WARNING: instance $ID still listed after 6 destroy attempts"
}

log "babysit $NAME id=$ID cap=${CAP} min"
# 1. wait for SSH
while :; do
  if past_cap; then log "cap reached before SSH came up"; destroy; exit 1; fi
  U=$(vastai ssh-url "$ID" --api-key "$VAST_API_KEY" 2>/dev/null | grep -o 'ssh://[^ ]*' | tail -1)
  if [ -n "$U" ]; then
    HP=${U#ssh://}; HP=${HP#*@}; H=${HP%:*}; P=${HP##*:}
    rsh true 2>/dev/null && break
  fi
  sleep 20
done
log "ssh up at $H:$P"

# 2. upload, arm the on-box backstop (stops the instance at CAP+15 min if this babysitter died), start the run
for i in 1 2 3; do
  scp "${SSHO[@]}" -P "$P" "$D/tickbound.tgz" "$D/$ONBOX" "$D/backstop.sh" "root@$H:/root/" >> "$LOG" 2>&1 && break
  log "upload attempt $i failed"; sleep 10
done
rsh "nohup setsid sh /root/backstop.sh $(( (CAP + 15) * 60 )) > /root/backstop.out 2>&1 < /dev/null &
     nohup setsid bash /root/$ONBOX > /root/onbox.out 2>&1 < /dev/null &
     sleep 3; echo started; ls -la /root/tickbound.tgz" >> "$LOG" 2>&1
log "on-box run started"

# 3. poll
while :; do
  if rsh test -f /root/DONE 2>/dev/null; then log "DONE on box"; break; fi
  if past_cap; then log "cap reached: packing partial results"; rsh "cd /root && tar -czf /root/res.tgz res" >> "$LOG" 2>&1; break; fi
  rsh "tail -c 4000 /root/res/onbox.log 2>/dev/null || tail -c 4000 /root/onbox.out" > "$OUT/tail.txt" 2>/dev/null
  sleep 60
done

# 4. fetch, destroy
for i in 1 2 3; do
  scp "${SSHO[@]}" -P "$P" "root@$H:/root/res.tgz" "$OUT/res.tgz" >> "$LOG" 2>&1 && break
  log "fetch attempt $i failed"; sleep 10
done
log "fetched $(stat -c %s "$OUT/res.tgz" 2>/dev/null || echo 0) bytes"
destroy
log "end"
