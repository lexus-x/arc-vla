#!/bin/bash
# Wait for the first LIBERO pass, then rerun missing cells (e.g. CUDA OOM) at P=2, at most 3 passes.
cd "$(dirname "$0")"
while pgrep -f "bash ./run_libero_screen.sh" >/dev/null; do sleep 60; done
for i in 1 2 3; do
  n=$(ls -d libero_*/eval_info.json 2>/dev/null | wc -l); [ "$n" -ge 16 ] && break
  for d in libero_*_k*/; do [ -f "$d/eval_info.json" ] || rm -rf "$d"; done  # lerobot refuses an existing output_dir
  P=2 ./run_libero_screen.sh >> run_libero_screen.out 2>&1
done
echo "retry done: $(ls -d libero_*/eval_info.json | wc -l)/16" >> run_libero_screen.out
