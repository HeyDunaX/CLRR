"""Compatibility imports; implementation lives in src/methods/."""
if __package__.startswith("src."):
    from src.methods.attached_source.v1.clrr import AttachedSourceResidualRewire
else:
    from methods.attached_source.v1.clrr import AttachedSourceResidualRewire
