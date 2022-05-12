from gym_minigrid.wrappers import RGBImgPartialObsWrapper
from babyai.levels.query import Query
from babyai.utils.wrapper import DirectionWrapper, TransformReward, AnsWrapper, XYLocationWrapper, InstrWrapper, KGWrapper, RenderWrapper, CountRewardWrapper
import gym
from babyai.levels.query_v2 import ObjInBoxMulti_MultiOracle_Query
import functools
import math

def make_envs(args, num_envs=None, verbose=False, render=False):
    mapping = {
        'BabyAI-ObjInBoxMulti-v0': Query,
        'BabyAI-ObjInBoxMultiMultiOracle-v0': ObjInBoxMulti_MultiOracle_Query,
        'BabyAI-ObjInBoxMultiMultiOracleSufficient-v0': ObjInBoxMulti_MultiOracle_Query,
        'BabyAI-Danger-v0': ObjInBoxMulti_MultiOracle_Query,
        'BabyAI-DangerRS8-v0': ObjInBoxMulti_MultiOracle_Query,
        'BabyAI-DangerRS9-v0': ObjInBoxMulti_MultiOracle_Query,
        'BabyAI-OpenDoorMultiKeys-v0': ObjInBoxMulti_MultiOracle_Query,
        'BabyAI-DangerOld-v0': ObjInBoxMulti_MultiOracle_Query,
        'BabyAI-DangerSanityCheck-v0': ObjInBoxMulti_MultiOracle_Query,
        'BabyAI-DangerBalanced-v0': ObjInBoxMulti_MultiOracle_Query,
        'BabyAI-DangerBalancedDiverse-v0': ObjInBoxMulti_MultiOracle_Query,
        'BabyAI-DangerBalancedDiverseRS9-v0': ObjInBoxMulti_MultiOracle_Query,
        'BabyAI-DangerBalancedDiverseMove-v0': functools.partial(ObjInBoxMulti_MultiOracle_Query, call=False),
        'BabyAI-ObjInBoxV2-v0': functools.partial(ObjInBoxMulti_MultiOracle_Query, call=True),
        'BabyAI-ObjInBoxV2SingleMove-v0': functools.partial(ObjInBoxMulti_MultiOracle_Query, call=False),
        'BabyAI-ObjInBoxV2MultiMove-v0': functools.partial(ObjInBoxMulti_MultiOracle_Query, call=False),
        'BabyAI-GoToFavorite-v0': functools.partial(ObjInBoxMulti_MultiOracle_Query, call=True),
        'BabyAI-GoToFavoriteSingleMove-v0': functools.partial(ObjInBoxMulti_MultiOracle_Query, call=False),



    }
    num_envs = args.procs if not num_envs else num_envs
    #q_wrqpper = mapping[args.env] if args.env in mapping else ObjInBoxMulti_MultiOracle_Query
    envs = []
    env_names = args.env.split(',')
    n_diff_envs = len(env_names)
    n_proc_per_env = math.ceil(num_envs / n_diff_envs)
    for env_name in env_names:
        q_wrqpper = mapping[env_name] if env_name in mapping else functools.partial(ObjInBoxMulti_MultiOracle_Query, random_ans=args.random_ans)
        for i in range(n_proc_per_env):
            env = gym.make(env_name)
            env = DirectionWrapper(env)
            if args.query:
                env = q_wrqpper(env, restricted=False, flat='flat' in args.query_arch, n_q=args.n_query, verbose=verbose, query_limit=100, reliability=1)
            if args.kg_wrapper:
                assert not args.ans_image
                if args.kg_mode == 'no_kg':
                    if args.cc_bonus != 0:
                        print('Set CC Bonus to 0 in no_kg mode')
                        args.cc_bonus = 0
                env = KGWrapper(env, penalize_query=args.penalize_query, cc_bonus=args.cc_bonus,
                                weighted_bonus=args.weighted_bonus, kg_repr=args.kg_repr, mode=args.kg_mode, n_gram=args.n_gram,
                                distractor_file_path=args.distractors_path, n_distractors=args.n_distractors, args=args)
            else:
                if args.ans_image:
                    assert args.query or args.flat_query
                    env = AnsWrapper(env)
            if args.instr_image:
                assert not args.no_instr
                env = InstrWrapper(env)
            if render:
                env = RenderWrapper(env)
            if args.count_obs_coef > 0:
                env = CountRewardWrapper(env, args.count_obs_coef)
            env.seed(100 * args.seed + i)
            envs.append(env)
    return envs