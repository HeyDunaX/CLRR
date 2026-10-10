"""Compatibility imports; implementation lives in src/methods/."""
if __package__.startswith("src."):
    from src.methods.precache.v1.clrr import PrecacheResidualRewire
else:
    from methods.precache.v1.clrr import PrecacheResidualRewire
