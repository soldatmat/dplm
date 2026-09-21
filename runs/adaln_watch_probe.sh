#!/bin/bash
# One-shot status probe for the 3 adaLN-single training jobs.
# Prints one line per job: <jobid>|<state>|<info>
#   state PENDING/CONFIGURING/...  -> info is the squeue reason
#   state RUNNING                  -> info is the latest train.log marker (or setup status)
#   state TERMINAL                 -> info is the sacct State+ExitCode
# Used by the local Monitor loop; keeps the srun --overlap quoting in a real file.

for spec in 4487214:lr1em3 4487245:lr1em4 4487254:lr1em5; do
  j=${spec%%:*}; tok=${spec#*:}
  q=$(squeue -j "$j" -h -o '%T|%R' 2>/dev/null)
  if [ -z "$q" ]; then
    a=$(sacct -j "$j" --format=State,ExitCode -n 2>/dev/null | head -1 | tr -s ' ' | sed 's/^ *//;s/ *$//')
    echo "$j|TERMINAL|$a"
    continue
  fi
  st=${q%%|*}; rsn=${q#*|}
  if [ "$st" != "RUNNING" ]; then
    echo "$j|$st|$rsn"
    continue
  fi
  # RUNNING: probe the node-local scratch train.log for progress / failure markers.
  probe=$(timeout 30 srun --jobid="$j" --overlap bash -c '
    tl=$(ls /tmp/dplm_*adaln*'"$tok"'_*/dplm/logs/*/train.log 2>/dev/null | head -1)
    if [ -z "$tl" ]; then
      echo "setup(no-train.log-yet)"
    else
      hit=$(grep -hoE "global step [0-9]+|elapsed_steps=[0-9]+|Sanity Checking|Traceback|Error:|RuntimeError|CUDA error|OutOfMemoryError|ValueError|AssertionError|omegaconf|huggingface_hub|num_embeddings" "$tl" | tail -1)
      [ -z "$hit" ] && hit="train.log(no-marker-yet)"
      echo "$hit"
    fi' 2>/dev/null)
  [ -z "$probe" ] && probe="srun-probe-failed"
  echo "$j|RUNNING|$probe"
done
