from .resnet import build
from .patchD import build_patchD


def build_model(args):
    if args.arch == 'resnet50':
        return build(args)
    elif args.arch == 'patchD':
        return build_patchD(args)