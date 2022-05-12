#!/usr/bin/env python

import argparse
import itertools

import gym

import textworld
import textworld.agents

from tw_cooking_afk_env import TWCookingAFK, TWOracleWrapper, TWCookingAFKBatchGymEnv, register_tw_cooking_afk


def build_parser():
    description = "Play a TextWorld Cooking AFK game."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int,
                        help="Generate game using this seed.")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of games to run in parallel.")
    parser.add_argument("--max-steps", type=int, default=0, metavar="STEPS",
                        help="Limit maximum number of steps.")
    parser.add_argument("--hint", action="store_true",
                        help="Display the oracle trajectory leading to winning the game.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose mode.")
    parser.add_argument("-vv", "--very-verbose", action="store_true",
                        help="Print debug information.")

    group = parser.add_argument_group('The Cooking AFK Game settings')
    group.add_argument("--oracle-name", default="Charlie",
                       help="Name of the oracle to query. Default: %(default)s")

    group.add_argument("--recipe", type=int, default=1, metavar="INT",
                       help="Number of ingredients in the recipe. Default: %(default)s")
    group.add_argument("--take", type=int, default=0, metavar="INT",
                       help="Number of ingredients to find. It must be less or equal to"
                            " the value of `--recipe`. Default: %(default)s")
    # group.add_argument("--go", type=int, default=1, choices=[1, 6, 9, 12],
    #                    help="Number of locations in the game (1, 6, 9, or 12). Default: %(default)s")
    group.add_argument('--open', action="store_true",
                       help="Whether containers/doors need to be opened.")
    group.add_argument('--cook', action="store_true",
                       help="Whether some ingredients need to be cooked.")
    group.add_argument('--cut', action="store_true",
                       help="Whether some ingredients need to be cut.")
    # group.add_argument('--drop', action="store_true",
    #                    help="Whether the player's inventory has limited capacity.")
    group.add_argument('--knife-in-inventory', action="store_true",
                       help="Whether the player's start with the knife in their inventory.")
    group.add_argument("--recipe-seed", type=int, default=0, metavar="INT",
                       help="Random seed used for generating the recipe. Default: %(default)s")

    group.add_argument("--nb-furniture-distractors", type=int, default=0, metavar="INT",
                       help="Number of additional distractor furnitures to add. Default: %(default)s")
    group.add_argument("--nb-food-distractors", type=int, default=0, metavar="INT",
                       help="Number of additional distractor ingredients to add. Default: %(default)s")

    group.add_argument("--split", choices=["train", "valid", "test"],
                       help="Specify the game distribution to use. Food items (adj-noun pairs) are split in three subsets."
                            " Also, the way the training food items can be prepared is further divided in three subsets.\n\n"
                            "* train: training food and their corresponding training preparations\n"
                            "* valid: valid food + training food but with unseen valid preparations\n"
                            "* test: test food + training food but with unseen test preparations\n\n"
                            " Default: game is drawn from the joint distribution over train, valid, and test.")

    return parser


def main():
    args = build_parser().parse_args()

    if args.very_verbose:
        args.verbose = args.very_verbose

    infos = textworld.EnvInfos(description=True, inventory=True, admissible_commands=True, facts=True, score=True, objective=True,
                               extras=["ans", "seed", "walkthrough"])
    env_id = register_tw_cooking_afk(settings=args.__dict__, request_infos=infos, batch_size=args.batch_size, asynchronous=args.batch_size > 1)

    env = gym.make(env_id)
    seeds = env.seed(args.seed)

    for _ in range(100):
        obs, infos = env.reset()
        print("-= Games #{} =-".format(infos["extra.seed"]))
        #if args.verbose:
        # print(obs[0])

        if args.knife_in_inventory:
            assert all(("knife" in inventory) for inventory in infos["inventory"])

        reward = 0
        done = False

        # Example of query
        #obs, rewards, dones, infos = env.step(["ask Charlie where is banana"] * 5)
        #print(infos["extra.ans"])

        # Get game's objective
        # infos["objective"]

        # Get game's walkthrough
        # print(infos["extra.walkthrough"])
        for commands in itertools.zip_longest(*infos["extra.walkthrough"]):
            # Check is noun in commands are found in the observation.

            for cmd_, obs_, inv_, desc_ in zip(commands, obs, infos["inventory"], infos["description"]):
                if cmd_ is None:
                    continue  # Game is already done.

                for word in cmd_.split():
                    if word in ("inventory", "examine", "open", "close", "take", "drop",
                                "slice", "chop", "dice",
                                "from", "the", "with"): # Ignore some words
                        continue

                    assert word in "{} {} {}".format(obs_, inv_, desc_), word

            obs, rewards, dones, infos = env.step(commands)

            # print(">", commands[0])
            # print(obs[0])

            if done:
                break

        assert all(dones) and all(score == 1 for score in infos["score"])

        # if args.verbose:
        #     msg = "Done after {} steps. Score {}/{}."
        #     msg = msg.format(game_state.moves, game_state.score, game_state.max_score)
        #     print(msg)


if __name__ == "__main__":
    main()
