#!/usr/bin/env bash
# Launch 2-GPU DPO training with WSL-friendly distributed settings.
set -eu

cd "$(dirname "$0")"
source ../.venv/bin/activate

# Kill stale launchers that block the rendezvous port.
if pgrep -f "accelerate launch.*dpo_train.py" >/dev/null 2>&1; then
  echo "Killing stale accelerate launcher(s)..."
  pkill -9 -f "accelerate launch.*dpo_train.py" || true
  sleep 2
fi

# WSL mirrored/VirtioProxy can break 127.0.0.1 TCP loopback for c10d.
# Use the WSL eth IP when localhost rendezvous hangs.
WSL_IP="$(hostname -I | awk '{print $1}')"
export MASTER_ADDR="${MASTER_ADDR:-$WSL_IP}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME=lo
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# Off by default on WSL — activation offloading causes intermittent CUDA "device not ready" on backward.
export DPO_ACTIVATION_OFFLOAD="${DPO_ACTIVATION_OFFLOAD:-0}"

if ss -tlnp 2>/dev/null | grep -q ":${MASTER_PORT} "; then
  echo "Port ${MASTER_PORT} still in use; switching to 29501"
  export MASTER_PORT=29501
fi

echo "WSL networking mode: $(wslinfo --networking-mode 2>/dev/null || echo unknown)"

if [[ "${DPO_PRECOMPUTE_REF:-1}" != "0" ]]; then
  echo "=== Precomputing reference log probs on GPU ${DPO_PRECOMPUTE_GPU:-0} (if needed) ==="
  CUDA_VISIBLE_DEVICES="${DPO_PRECOMPUTE_GPU:-0}" python -u dpo_train.py --precompute-ref-only
fi

echo "Launching DDP (MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT})..."

exec accelerate launch --config_file accelerate_configs/ddp_2gpu.yaml dpo_train.py "$@"
