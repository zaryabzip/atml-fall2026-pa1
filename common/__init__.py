import os

# Let unsupported MPS ops fall back to CPU instead of crashing. Only takes effect if
# `common` is imported before torch, so also export it in your shell (see README).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
