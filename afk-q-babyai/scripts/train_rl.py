#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import os
import logging
import csv
import json
import gym
import time
import datetime
import torch
import numpy as np
import subprocess

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model import ACModel, TwoStageModel
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent



# Parse arguments
parser = ArgumentParser()
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use (default: ppo)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--reward-scale", type=float, default=20.,
                    help="Reward scale multiplier")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
args = parser.parse_args()

utils.seed(args.seed)

# Generate environments
from babyai.utils.make_envs import make_envs
envs = make_envs(args)

# Define model name
suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
instr = args.instr_arch if args.instr_arch else "noinstr"
mem = "mem" if not args.no_mem else "nomem"
model_name_parts = {
    'env': args.env,
    'algo': args.algo,
    'arch': args.arch,
    'instr': instr,
    'mem': mem,
    'seed': args.seed,
    'info': '',
    'coef': '',
    'suffix': suffix,
    'note': args.note}
default_model_name = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}_{note}_{seed}".format(**model_name_parts)
if args.pretrained_model:
    default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
args.model = args.model.format(**model_name_parts) if args.model else default_model_name

utils.configure_logging(args.model)
logger = logging.getLogger(__name__)

# Define obss preprocessor
if 'emb' in args.arch:
    obss_preprocessor = utils.IntObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
else:
    obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model, args.query or args.flat_query, onehot_ans=args.onehot_ans)

if args.flat_query or args.query:
    #if args.query_arch == 'flat':
    #    action_postprocessor = utils.FlatActionPostprocessor(restricted=args.restricted_query, env=args.env)
    #elif 'pointer' in args.query_arch:
    action_postprocessor = utils.PointerActionPostprocessor(obss_preprocessor.vocab)
else:
    action_postprocessor = None

# Define actor-critic model
acmodel = utils.load_model(args.model, raise_not_found=False)
if acmodel is None:
    if args.pretrained_model:
        acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
    else:
        acmodel = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                          args.image_dim, args.memory_dim, args.instr_dim,
                          not args.no_instr, args.instr_arch, not args.no_mem, args.arch, query=args.query,
                          flat_query=args.flat_query, onehot_ans=args.onehot_ans, no_qa_embedding=args.ans_image or args.kg_repr == 'one_hot',
                          n_image_channel=envs[0].observation_space['image'].shape[-1], kg_repr=args.kg_repr,
                          invariant_module=args.invariant_module, gcn_adj_type=args.gcn_adj_type,
                          controller_arch=args.controller_arch, query_arch=args.query_arch,
                          no_kg=args.kg_mode == 'no_kg', vocab=obss_preprocessor.vocab)

#pre_trained_acmodel = ACModel(obss_preprocessor.obs_space, envs[0].unwrapped.action_space,
#                  args.image_dim, args.memory_dim, args.instr_dim,
#                  not args.no_instr, args.instr_arch, not args.no_mem, args.arch, query=False, flat_query=False)
print('# of params {}'.format(sum(p.numel() for p in acmodel.parameters() if p.requires_grad)))
if args.two_stage_query:
    pre_trained_acmodel = utils.load_model(args.pretrained_model, raise_not_found=False)
    pre_trained_acmodel.two_stage_query = True
    acmodel = TwoStageModel(pre_trained_acmodel)

obss_preprocessor.vocab.save()
utils.save_model(acmodel, args.model)

if torch.cuda.is_available():
    acmodel.cuda()

# Define actor-critic algo

reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
if args.algo == "ppo":
    algo = babyai.rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                             args.gae_lambda,
                             args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                             args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size, obss_preprocessor,
                             reshape_reward, postprocess_action=action_postprocessor,
                             update_query_only=args.two_stage_query, tvt=args.tvt, bin_entropy_coef=args.bin_entropy_coef)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
# Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
# the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

utils.seed(args.seed)

# Restore training status

status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
if os.path.exists(status_path):
    with open(status_path, 'r') as src:
        status = json.load(src)
else:
    status = {'i': 0,
              'num_episodes': 0,
              'num_frames': 0}


# Define logger and Tensorboard writer and CSV writer

header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"]
          #+ ["action_{}".format(i) for i in range(sum(env.action_space.nvec))]
          )
if args.tb:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(utils.get_log_dir(args.model))
csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
first_created = not os.path.exists(csv_path)
# we don't buffer data going in the csv log, cause we assume
# that one update will take much longer that one write to the log
csv_writer = csv.writer(open(csv_path, 'a', 1))
if first_created:
    csv_writer.writerow(header)

# Log code state, command, availability of CUDA and model

babyai_code = list(babyai.__path__)[0]
try:
    last_commit = subprocess.check_output(
        'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
    logger.info('LAST COMMIT INFO:')
    logger.info(last_commit)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
try:
    diff = subprocess.check_output(
        'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
    if diff:
        logger.info('GIT DIFF:')
        logger.info(diff)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
logger.info('COMMAND LINE ARGS:')
logger.info(args)
logger.info("CUDA available: {}".format(torch.cuda.is_available()))
logger.info(acmodel)

# Train model

total_start_time = time.time()
best_success_rate = 0
best_mean_return = 0
best_mean_frames = 1000
test_env_name = args.env
save_target = 1_000_000

"""
agent = ModelAgent(args.model, obss_preprocessor, argmax=False, query=args.query or args.flat_query, postprocess_action=action_postprocessor)
agent.model = acmodel
agent.model.eval()
logs = batch_evaluate(agent, test_env_name, args.val_seed, args.val_episodes, query=args.query or args.flat_query,
                      restricted_query=args.restricted_query, args=args)
"""
#utils.save_model(acmodel, args.model + '_not_trained')
#obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_not_trained'))
if args.detail_log:
    log_rq = []


while status['num_frames'] < args.frames:
    # Update parameters

    update_start_time = time.time()
    logs = algo.update_parameters()
    update_end_time = time.time()
    if args.detail_log:
        log_rq.extend(logs['return_query'])
    status['num_frames'] += logs["num_frames"]
    status['num_episodes'] += logs['episodes_done']
    status['i'] += 1
    #print(status)
    # Print logs


    if status['i'] % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = datetime.timedelta(seconds=total_ellapsed_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        #success_per_episode = utils.synthesize(
        #    [1 if r > 0 else 0 for r in logs["return_per_episode"]])
        success_per_episode = utils.synthesize(
            [1 if s else 0 for s in logs["success"]])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        data = [status['i'], status['num_episodes'], status['num_frames'],
                fps, total_ellapsed_time,
                *return_per_episode.values(),
                success_per_episode['mean'],
                *num_frames_per_episode.values(),
                logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                logs["loss"], logs["grad_norm"]]
        #data.extend(logs['action_prob'])
        data_action = list(logs['action_prob'])
        #utils.save_model(acmodel, args.model + '_first_log')
        #obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_first_log'))
        format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")



        logger.info(format_str.format(*data))
        logger.info('action prob {}'.format(logs['action_prob']))
        if args.tb:
            assert len(header) == len(data)
            for key, value in zip(header, data):
                writer.add_scalar(key, float(value), status['num_frames'])
        if args.kg_wrapper:
            logger.info(envs[0].KG)
        data.extend(data_action)
        csv_writer.writerow(data)

    # Save obss preprocessor vocabulary and model

    if args.save_interval > 0 and status['i'] % args.save_interval == 0:
        if args.detail_log:
            rq_path = os.path.join(utils.get_log_dir(args.model), 'log_rq.json')
            #print(rq_path)
            #print(log_rq)
            with open(rq_path, 'w') as f:
                json.dump(log_rq, f, indent=2)
            #print('saved to ', rq_path)
        obss_preprocessor.vocab.save()
        with open(status_path, 'w') as dst:
            json.dump(status, dst)
            utils.save_model(acmodel, args.model)

        # Testing the model before saving
        #agent = ModelAgent(args.model, obss_preprocessor, argmax=True, query=args.query or args.flat_query, postprocess_action=action_postprocessor)
        agent = ModelAgent(args.model, obss_preprocessor, argmax=False, query=args.query or args.flat_query, postprocess_action=action_postprocessor)
        agent.model = acmodel
        agent.model.eval()
        logs = batch_evaluate(agent, test_env_name, args.val_seed, args.val_episodes, query=args.query or args.flat_query,
                              restricted_query=args.restricted_query, args=args) 
        agent.model.train()
        mean_return = np.mean(logs["return_per_episode"])
        mean_frames = np.mean(logs["num_frames_per_episode"])
        #success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
        success_rate = np.mean([1 if s else 0 for s in logs['success']])
        save_model = False
        #if success_rate > best_success_rate:
        if mean_frames < best_mean_frames:
            best_mean_frames = mean_frames
            best_success_rate = success_rate
            save_model = True
        #elif (success_rate == best_success_rate) and (mean_return > best_mean_return):
        #    best_mean_return = mean_return
        #    save_model = True
        if save_model:
            utils.save_model(acmodel, args.model, best=True)
            obss_preprocessor.vocab.save(utils.get_vocab_path(args.model))
            logger.info("Return {: .2f}, Success rate {:.2f}, # of frames {:.2f}; best model is saved".format(mean_return, success_rate, mean_frames))
        else:
            logger.info("Return {: .2f}, Success rate {:.2f}, # of frames {:.2f}; not the best model; not saved".format(mean_return, success_rate, mean_frames))
        #if status['num_frames'] > save_target:
        #    utils.save_model(acmodel, args.model + '_{}'.format(status['num_frames']))
        #    obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_{}'.format(status['num_frames'])))
        #    save_target += 1_000_000