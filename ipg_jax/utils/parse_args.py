import sys
import argparse

from numpy import log
# Experiment Constants for GDmax
dim = 4

S = dim * dim * dim * dim * 2 * dim * dim * 2
H = 12
K = 100
gamma = 0.99
eps = nu = 0.1
p = 1/2
lp = ((320 * gamma)*(16)**(5/2) * S**2) / (nu * (1 / S) * (1 - gamma)**(11 / 2))
delta = ((eps/8)**((1 + p) / p)) / lp**(1/p)
team_T = 32 * lp**(2/(1 + p)) * 2 / (delta**((1 - p)/(1 + p)) * eps**2)
eta = (delta**((1 - p)/(1 + p))) / (32 * lp**(2/(1 + p)) * 2)
lfnu = (2 * gamma**2 * S * 4 / ((1 - gamma)**3)) + ((3 * nu * gamma**2 * S * 4) / ((1 - gamma)**4))
linv = 2 / ((1/S) * (1 - gamma))
zeta = 0.1
C1 = (9 / ((1 - gamma)**4 * zeta**2)) + ((18 * nu)/((1 - gamma)**5 * zeta**2)) + ((30 * nu**2)/((1 - gamma)**6 * zeta**2))
C2 = 6 * ( ((H+1)**2 / ((1 - gamma)**2 * zeta **2)) + (2 * nu * (H+1)**2 / ((1 - gamma)**3 * zeta **2)) + ((nu**2 * (H+1)**2 + 1)/ ((1 - gamma)**4 * zeta **2)) ) + (2 * nu / ((1 - gamma)**5 * zeta**2)) + (2 * nu**2 / ((1 - gamma)**6 * zeta**2))
sigma2 = C1 / K + C2 + gamma**(2 * H)
M = 9 * sigma2  / (2 * eps**2)
adv_T = (lfnu * linv**2 / nu) * log( 1 / eps ) + (lfnu * sigma2 * linv**4 / (nu**2 * eps)) * log(1 / eps)

def csv(arg, typecast):
    return [typecast(i) for i in arg.split(',')]

def parse_args(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l", "--rollout-length", help="Number of rollout episodes", default=K, type=int, dest="rollout_length"
    )
    parser.add_argument(
        "-lr", help="Learning rate", default = eta, type=float
    )
    parser.add_argument(
        "-g", "--gamma", help="Discount Factor", default = gamma, type=float
    )
    parser.add_argument(
        "-na", "--net-arch", help="Network architecture (comma separated)", default=[64,128], dest="net_arch", type=lambda v: csv(v, int)
    )
    parser.add_argument(
        "-e", "--eval", help="Display environment with given policies", action="store_true"
    ) 
    parser.add_argument(
        "-de", "--disable-eval", help="Disable post-training evaluation", action="store_true", dest="disable_eval"
    )
    parser.add_argument(
        "-team", "--team-path", help="Path to team policy (eval or warmstart)", default=None, dest="team"
    )
    parser.add_argument(
        "-adv", "--adv-path", help="Path to adversarial policy (eval or warmstart)", default=None, dest="adv"
    )
    parser.add_argument(
        "-i", "--iters", help="Number of training iterations", default=int(team_T), type=int
    )
    parser.add_argument(
       "-ng","--nash-gap", help="Measure Nash-Gap", action="store_true", dest="nash_gap"
    )
    parser.add_argument(
        "-ds", "--disable-save", help="Disable checkpoint and eval video saving", action="store_true", dest="disable_save"
    )
    parser.add_argument(
        "-br", "--br-length", help="Number of updates to find best respond", type=int, default=int(adv_T), dest="br_length"
    )
    parser.add_argument(
        "-mi", "--metric-interval", help="Number of iterations between metric collection", type=int, default=50, dest="metric_interval"
    )
    parser.add_argument(
        "-si", "--save-interval", help="Number of iterations between model saves", type=int, default=500, dest="save_interval"
    )
    parser.add_argument(
        "-dim", "--grid-dimension", help="Grid dimension", type=int, default=dim, dest="dim"
    )
    parser.add_argument(
        "-f", "--fix-grid", help="Fix grid configuration", action="store_const", const="MultiGrid-Empty-3x3-TeamCoop", default= "MultiGrid-Empty-3x3-Team", dest="env"
    ) # const = "MultiGrid-Empty-3x3-TeamWins"

    ## IPGmax Experiments
    parser.add_argument(
        "-eps", "--epsilon", help="Nash Precision", type=float, default=eps, dest="eps"
    )
    parser.add_argument(
        "-nu", help="Coefficient for lambda in IPGDmax", type=float, default=nu, dest="nu"
    )
    parser.add_argument(
        "-tr", "--team-rollout-length", type=int, default=int(M), dest="tr"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=0, dest="seed"
    )
    parser.add_argument(
        "-p", "--parameterization",help="Parameterization type", default="direct", dest="param", choices=["direct", "nn"]
    )
    args, _ = parser.parse_known_args(cmd_args)

    return args
    



