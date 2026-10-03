"""Matched Full control: only the CLRR source branch retains gradients."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from amis_rewire import train
from amis_rewire.ablation import AttachedSourceResidualRewire
from amis_rewire.modeling import JEPAGuidedSeq2SeqLM


def main():
    original_loader=train.load_model
    original_parser=train.parse_args

    def parse():
        args=original_parser()
        if args.method not in ('jepa-clrr','jepa-clrr-enc') or args.rewire_stack!='encoder':
            raise ValueError('No-detach control requires Full encoder routing.')
        args.clrr_source_detach=False
        args.lsr_target_detach=True
        return args

    def load(model_name,method,distance=2,strength=.1,stack='encoder',jepa_weight=.1):
        raw=original_loader(model_name,'baseline')
        route=AttachedSourceResidualRewire(raw,distance=distance,strength=strength,stack=stack)
        return JEPAGuidedSeq2SeqLM(route,jepa_weight=jepa_weight)

    train.parse_args=parse
    train.load_model=load
    try:
        train.main()
    finally:
        train.parse_args=original_parser
        train.load_model=original_loader


if __name__=='__main__':
    main()
