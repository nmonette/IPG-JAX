import sys

from ipg_jax.train import main as main_direct
from ipg_jax.train_nn_cont import main as main_cont
from ipg_jax.utils import parse_args

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    main_direct(args)

    # if args.param == "nn":
    #     main_nn(args)
    # elif args.param == "c":
    #     main_cont(args)
    # else:
    #     main_direct(args)