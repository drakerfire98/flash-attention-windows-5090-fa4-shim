# FA4 Windows FlashAttention Runtime Overlay

This package provides a repo-owned `flash_attn.cute` overlay for the Windows
FA4 probe path.

It keeps the currently patched `interface.py` inside this repo instead of
depending on the live `third_party/flash-attention-for-windows/.../interface.py`
file at import time.

The overlay only owns the small surfaces we need locally:

- `flash_attn.__init__`
- `flash_attn.cute.__init__`
- `flash_attn.cute.interface`

Other `flash_attn.cute.*` modules still fall through to the upstream tree via
`__path__`.
