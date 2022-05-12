"""
Common arguments for BabyAI training scripts
"""

import os
import argparse
import numpy as np
import yaml

class ArgumentParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        # Base arguments
        self.add_argument("--env", default=None,
                            help="name of the environment to train on (REQUIRED)")
        self.add_argument("--model", default=None,
                            help="name of the model (default: ENV_ALGO_TIME)")
        self.add_argument("--pretrained-model", default=None,
                            help='If you\'re using a pre-trained model and want the fine-tuned one to have a new name')
        self.add_argument("--seed", type=int, default=1,
                            help="random seed; if 0, a random random seed will be used  (default: 1)")
        self.add_argument("--task-id-seed", action='store_true',
                            help="use the task id within a Slurm job array as the seed")
        self.add_argument("--procs", type=int, default=64,
                            help="number of processes (default: 64)")
        self.add_argument("--tb", action="store_true", default=False,
                            help="log into Tensorboard")


        # Training arguments
        self.add_argument("--log-interval", type=int, default=10,
                            help="number of updates between two logs (default: 10)")
        self.add_argument("--frames", type=int, default=int(9e10),
                            help="number of frames of training (default: 9e10)")
        self.add_argument("--patience", type=int, default=100,
                            help="patience for early stopping (default: 100)")
        self.add_argument("--epochs", type=int, default=1000000,
                            help="maximum number of epochs")
        self.add_argument("--epoch-length", type=int, default=0,
                            help="number of examples per epoch; the whole dataset is used by if 0")
        self.add_argument("--frames-per-proc", type=int, default=40,
                            help="number of frames per process before update (default: 40)")
        self.add_argument("--lr", type=float, default=1e-4,
                            help="learning rate (default: 1e-4)")
        self.add_argument("--beta1", type=float, default=0.9,
                            help="beta1 for Adam (default: 0.9)")
        self.add_argument("--beta2", type=float, default=0.999,
                            help="beta2 for Adam (default: 0.999)")
        self.add_argument("--recurrence", type=int, default=20,
                            help="number of timesteps gradient is backpropagated (default: 20)")
        self.add_argument("--optim-eps", type=float, default=1e-5,
                            help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
        self.add_argument("--optim-alpha", type=float, default=0.99,
                            help="RMSprop optimizer apha (default: 0.99)")
        self.add_argument("--batch-size", type=int, default=1280,
                                help="batch size for PPO (default: 1280)")
        self.add_argument("--entropy-coef", type=float, default=0.01,
                            help="entropy term coefficient (default: 0.01)")
        self.add_argument("--tvt", action="store_true", default=False)

        # Model parameters
        self.add_argument("--image-dim", type=int, default=128,
                            help="dimensionality of the image embedding.  Defaults to 128 in residual architectures")
        self.add_argument("--memory-dim", type=int, default=128,
                            help="dimensionality of the memory LSTM")
        self.add_argument("--instr-dim", type=int, default=128,
                            help="dimensionality of the memory LSTM")
        self.add_argument("--no-instr", action="store_true", default=False,
                            help="don't use instructions in the model")
        self.add_argument("--instr-arch", default="gru",
                            help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
        self.add_argument("--no-mem", action="store_true", default=False,
                            help="don't use memory in the model")
        self.add_argument("--arch", default='bow_endpool_res',
                            help="image embedding architecture")

        # Validation parameters
        self.add_argument("--val-seed", type=int, default=int(1e9),
                            help="seed for environment used for validation (default: 1e9)")
        self.add_argument("--val-interval", type=int, default=1,
                            help="number of epochs between two validation checks (default: 1)")
        self.add_argument("--val-episodes", type=int, default=500,
                            help="number of episodes used to evaluate the agent, and to evaluate validation accuracy")
        self.add_argument("--note", default='')

        # Query parameters
        self.add_argument("--query", action="store_true", default=False)
        self.add_argument("--restricted_query", action="store_true", default=False)
        self.add_argument("--flat_query", action="store_true", default=False)
        self.add_argument("--two_stage_query", action="store_true", default=False)
        self.add_argument("--n_query", type=int, default=1)
        self.add_argument("--onehot_ans", action="store_true", default=False)
        self.add_argument("--ans_image", action="store_true", default=False)
        self.add_argument("--instr_image", action="store_true", default=False)
        self.add_argument("--step_penalty", action="store_true", default=False)
        self.add_argument("--xy_wrapper", action="store_true", default=False)
        self.add_argument("--kg_wrapper", action="store_true", default=False)
        self.add_argument("--count_explore", action="store_true", default=False)
        self.add_argument("--explore_coef", type=float, default=1)
        self.add_argument("--bin_entropy_coef", type=float, default=0.1)
        self.add_argument("--detail_log", action="store_true", default=False)
        self.add_argument("--penalize_query", action="store_true", default=False)
        self.add_argument("--cc_bonus", type=float, default=0.02)
        self.add_argument("--weighted_bonus", action="store_true", default=False)
        self.add_argument("--kg_repr", type=str, default='one_hot')
        self.add_argument("--invariant_module", type=str, default='max')
        self.add_argument("--gcn_adj_type", type=str, default='no_connect')
        self.add_argument("--controller_arch", type=str, default='film', help='[film, att]')
        self.add_argument("--query_arch", type=str, default='flat', help='[flat, pointer]')
        self.add_argument("--kg_mode", type=str, default='graph_overlap', help='[graph_overlap, graph_cosine, set, no_kg]')
        self.add_argument("--query_limit", type=int, default=100)
        self.add_argument("--n_gram", type=int, default=2)
        self.add_argument("--distractors_path", type=str, default=None)
        self.add_argument("--n_distractors", type=int, default=0)
        self.add_argument("--node_sample_mode", type=str, default='fixed')
        self.add_argument("--reliability", type=float, default=1)
        self.add_argument("--count_obs_coef", type=float, default=0)
        self.add_argument("--decrease_bonus", action="store_true", default=False)
        self.add_argument("--config", type=str, default=None)
        self.add_argument("--failure_neg", action="store_true", default=False)
        self.add_argument("--random_ans", action="store_true", default=False)


    def parse_args(self):
        """
        Parse the arguments and perform some basic validation
        """
        args = super().parse_args()
        if args.config:
            stream = open("../run/config/{}.yaml".format(args.config), 'r')
            config = yaml.load(stream, Loader=yaml.FullLoader)
            for key, value in config.items():
                print (key + " : " + str(value))
                args.__dict__[key] = value



        # Set seed for all randomness sources
        if args.seed == 0:
            args.seed = np.random.randint(10000)
        if args.task_id_seed:
            args.seed = int(os.environ['SLURM_ARRAY_TASK_ID'])
            print('set seed to {}'.format(args.seed))

        # TODO: more validation
        if args.distractors_path is not None:
            print('[INFO]: For initial knowledge w/ w/o distractors experiments, the kg mode is set to "set" ')
            args.kg_mode = 'set'
        return args
