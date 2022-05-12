#!/usr/bin/env python

import argparse
import itertools

import textworld
import textworld.agents

from tw_cooking_afk_env import TWCookingAFK, TWOracleWrapper


def build_parser():
    description = "Play a TextWorld Cooking AFK game."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int,
                        help="Generate game using this seed.")
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

    infos = textworld.EnvInfos(description=True, inventory=True, admissible_commands=True, facts=True)
    env = TWCookingAFK(infos, args.__dict__)
    env = TWOracleWrapper(verbose=args.very_verbose)(env)

    env.seed(args.seed)
    for _ in range(100):
        print("-= Game #{} =-".format(env._seed))
        game_state = env.reset()
        if args.verbose:
            env.render()

        reward = 0
        done = False

        for command in game_state.game.metadata["walkthrough"]:
            game_state, reward, done = env.step(command)

            if args.verbose:
                print(">", command)
                env.render(mode="human")

            if done:
                break

        assert done and game_state.score == game_state.max_score

        if args.verbose:
            msg = "Done after {} steps. Score {}/{}."
            msg = msg.format(game_state.moves, game_state.score, game_state.max_score)
            print(msg)


if __name__ == "__main__":
    main()
