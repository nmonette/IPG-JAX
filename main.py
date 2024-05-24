import sys, os

import matplotlib.pyplot as plt

from ipg_jax.train import main as main_direct
from ipg_jax.train_nn_cont import main as main_cont
from ipg_jax.utils import parse_args

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    args.param = "direct"
    nash_gap1, cum_dist1, dist1 = main_direct(args)
    args.param = "nn"
    nash_gap2, cum_dist2, dist2 = main_direct(args)

    os.makedirs("output", exist_ok=True)
    experiment_num = len(list(os.walk('./output')))
    os.makedirs(f"output/experiment-{experiment_num}", exist_ok=True)

    plt.plot(nash_gap1, label="direct", color='r')
    plt.plot(nash_gap2, label="nn",  color = 'g')
    plt.xlabel("Iterations")
    plt.title("Nash Gap")

    plt.savefig(f"output/experiment-{experiment_num}/nash-gap")
    plt.legend()
    plt.close()

    plt.plot(cum_dist1, label="direct", color='r')
    plt.plot(cum_dist2, label="nn", color='g')
    plt.xlabel("Iterations")
    plt.title("Cumulative Avg Euclidean Distance Between Policies")

    plt.savefig(f"output/experiment-{experiment_num}/cumulative-distance")
    plt.legend()
    plt.close()
    
    plt.plot(dist1, label="direct", color='r')
    plt.plot(dist2, label="nn", color='g')
    plt.xlabel("Iterations")
    plt.title("Euclidean Distance Between Policies")

    plt.savefig(f"output/experiment-{experiment_num}/distance")
    plt.legend()
    plt.close()

    # if args.param == "nn":
    #     main_nn(args)
    # elif args.param == "c":
    #     main_cont(args)
    # else:
    #     main_direct(args)