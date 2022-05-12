import os
import re
import sys
import textwrap

from io import StringIO
from functools import partial
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np

import gym
from gym.utils import colorize
from gym.envs.registration import register, registry
from gym.utils import seeding

from textworld import EnvInfos
from textworld.envs.wrappers import Filter, Limit
from textworld.envs.batch import AsyncBatchEnv, SyncBatchEnv

import textworld
from textworld import EnvInfos
from textworld.envs.tw import TextWorldEnv
from textworld.generator.inform7 import Inform7Game

from tw_cooking_afk_gen import make


STOP_WORDS = ["the", "a", "some", "an", "one", "to", "is"]
COOKED_STATES = ["roasted", "grilled", "fried"]
HEAT_SOURCES = ["oven", "griller", "stove"]
CUT_STATES = ["sliced", "diced", "chopped"]


rg1 = re.compile("\[a list of things (in|on) the ([^\]]+)\]")
rg2 = re.compile("\[if ([^\]]+) is open and there is something in the ([^\]]+)\]([^\]]+)\[end if\]")
rg3 = re.compile("\[if ([^\]]+) is open and the ([^\]]+) contains nothing\]([^\]]+)\[end if\]")
rg4 = re.compile("\[if there is something on the ([^\]]+)\]([^\]]+)\[end if\]")
rg5 = re.compile("\[if there is nothing on the ([^\]]+)\]([^\]]+)\[end if\]")
rg6 = re.compile("\[if ([^\]]+) is locked\]([^\]]+)\[else if ([^\]]+) is open\]([^\]]+)\[otherwise\]([^\]]+)\[end if\]")


def format_list(elements):
    if len(elements) == 0:
        return "nothing"

    out = ", ".join(elements)
    idx = out.rfind(", ")
    if idx == -1:
        return out

    return out[:idx] + ", and " + out[idx+2:]


class TWCookingAFK(TextWorldEnv):
    def __init__(self, infos, settings, *args, **kwargs):
        super().__init__(infos=infos, *args, **kwargs)
        self._seed = None
        self.settings = settings
        self.options = textworld.GameOptions()

    def _describe_room(self):
        facts = [(fact.name,) + tuple(fact.names) for fact in self.state["_facts"]]
        room = [fact.names[1] for fact in self.state["_facts"] if fact.name == "at" and fact.names[0] == "P"][0]

        # Replace all listing templates, e.g. "[a list of things in the c_0]"
        template = self._game.infos[room].desc
        out = ""
        idx = 0
        for match in rg1.finditer(template):
            out += template[idx:match.start()]
            items = [fact.names[0] for fact in self.state["_facts"] if fact.name == match.groups()[0] and fact.names[1] == match.groups()[1]]
            items = list(map(self._name_entity, items))
            out += format_list(items)
            idx = match.end()

        out += template[idx:]

        # Replace all conditional templates, e.g. "[if c_0 is open and there is something in the c_0]...[end if]"
        template = out
        out = ""
        idx = 0
        for match in rg2.finditer(template):
            out += template[idx:match.start()]

            if ("open", match.groups()[0]) in facts and any(fact for fact in facts if fact[0] == "in" and fact[2] == match.groups()[0]):
                out += match.groups()[2]

            idx = match.end()

        out += template[idx:]

        # Replace all conditional templates, e.g. "[if c_0 is open and the c_0 contains nothing]...[end if]"
        template = out
        out = ""
        idx = 0
        for match in rg3.finditer(template):
            out += template[idx:match.start()]

            if ("open", match.groups()[0]) in facts and not any(fact for fact in facts if fact[0] == "in" and fact[2] == match.groups()[0]):
                out += match.groups()[2]

            idx = match.end()

        out += template[idx:]

        # Replace all conditional templates, e.g. "[if there is something on the s_0]...[end if]"
        template = out
        out = ""
        idx = 0
        for match in rg4.finditer(template):
            out += template[idx:match.start()]

            if any(fact for fact in facts if fact[0] == "on" and fact[2] == match.groups()[0]):
                out += match.groups()[1]

            idx = match.end()

        out += template[idx:]

        # Replace all conditional templates, e.g. "[if there is nothing on the s_0]...[end if]"
        template = out
        out = ""
        idx = 0
        for match in rg5.finditer(template):
            out += template[idx:match.start()]

            if not any(fact for fact in facts if fact[0] == "on" and fact[2] == match.groups()[0]):
                out += match.groups()[1]

            idx = match.end()

        out += template[idx:]


        # Replace all conditional templates, e.g. "[if there is nothing on the s_0]...[end if]"
        template = out
        out = ""
        idx = 0
        for match in rg6.finditer(template):
            out += template[idx:match.start()]

            if ("locked", match.groups()[0]) in facts:
                out += match.groups()[1]
            elif ("open", match.groups()[2]) in facts:
                out += match.groups()[3]
            else:
                out += match.groups()[4]

            idx = match.end()

        out += template[idx:]

        # Add items on the floor.
        items = [fact.names[0] for fact in self.state["_facts"] if fact.name == "at" and fact.names[1] == room and fact.types[0] in ("f", "o", "meal")]
        items = list(map(self._name_entity, items))
        out += "There is " + format_list(items) + " on the floor."

        assert "[" not in out and "]" not in out, "Tell Marc!"
        return out

    def _describe_inventory(self):
        inventory = "You are carrying: "

        items = [fact.names[0] for fact in self.state["_facts"] if fact.name == "in" and fact.names[1] == "I"]
        items = list(map(self._describe_entity, items))
        inventory += format_list(items) + "."
        return inventory

    def _describe_entity(self, entity):
        cook_states = [fact.name for fact in self.state["_facts"] if fact.name in COOKED_STATES and fact.names[0] == entity]
        cut_states = [fact.name for fact in self.state["_facts"] if fact.name in CUT_STATES and fact.names[0] == entity]

        out = self._game.infos[entity].name
        if cook_states:
            out = cook_states[0] + " " + out
        if cut_states:
            out = cut_states[0] + " " + out

        if cook_states or cut_states:
            out = "a " + out
        else:
            out = self._game.infos[entity].indefinite + " " + out

        return out

    def _name_entity(self, entity):
        out = self._game.infos[entity].indefinite + " " + self._game.infos[entity].name
        return out

    def seed(self, seed: Optional[int] = None) -> None:
        """ Sets the seed controlling the game generation on the next .reset() call. """
        # Seed the random number generator
        if seed is None:
            self.np_random, _ = seeding.np_random()
            seed = self.np_random.randint(2147483647)

        self.np_random, self._seed = seeding.np_random(seed)
        return self._seed

    def load(self, gamefile: str) -> None:
        raise NotImplementedError("Instead call .seed() -> .reset() to generate a new game.")

    def reset(self):
        self.options.seeds = self._seed
        self._seed = self.np_random.randint(2147483647)  # Prepare seed for the next reset.
        self._game = make(self.settings, self.options)
        self._inform7 = Inform7Game(self._game)
        self.oracle_name = self._game.metadata["oracle_name"]

        self.state = super().reset()
        self.state['description'] = self._describe_room()
        self.state['inventory'] = self._describe_inventory()
        self.state['feedback'] = self.state["objective"] + "\n\n" + self.state['description']
        self.state['extra.seed'] = self._seed
        self.state['extra.walkthrough'] = self._game.metadata["walkthrough"]
        return self.state

    def step(self, command: str):
        candidates = [cmd for cmd in self.state["admissible_commands"] if cmd.startswith(command)]
        if len(candidates) == 1:
            command = candidates[0]

        self.state, score, done = super().step(command)

        self.state['description'] = self._describe_room()
        self.state['inventory'] = self._describe_inventory()
        self.state['extra.seed'] = self._seed
        self.state['extra.walkthrough'] = self._game.metadata["walkthrough"]

        if self.state._last_action:
            if command == "look":
                self.state['feedback'] = self.state['description']
            elif command == "inventory":
                self.state['feedback'] = self.state['inventory']
            elif command == "prepare meal":
                self.state['feedback'] = "You mixed the ingredients to make the meal."
            elif command.startswith("drop"):
                self.state['feedback'] = "Dropped."
            elif command.startswith("put") or command.startswith("insert"):
                self.state['feedback'] = "Placed."
            elif command.startswith("eat"):
                self.state['feedback'] = "Eaten."
            elif command.startswith("take"):
                self.state['feedback'] = "Taken."
            elif command.startswith("open"):
                self.state['feedback'] = "Opened."
            elif command.startswith("close"):
                self.state['feedback'] = "Closed."
            elif command.startswith("cook"):
                self.state['feedback'] = "Cooked."
            elif command.startswith("slice"):
                self.state['feedback'] = "Sliced."
            elif command.startswith("chop"):
                self.state['feedback'] = "Chopped."
            elif command.startswith("dice"):
                self.state['feedback'] = "Diced."
            else:
                assert False, "Tell Admin!"
        elif command == "score":
            self.state['feedback'] = "Score: {}/{}.".format(self.state["score"], self.state["max_score"])

        return self.state, score, done


class TWOracleWrapper(textworld.core.Wrapper):
    QUERY_COMMANDS = ["where", "what", "is", "which", "how"]

    def __init__(self, env=None, verbose=False, query_limit=100, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self._gamefile = None
        self._game = None
        self.verbose = verbose
        self.prev_query, self.prev_ans = 'None', 'None'
        self.init_query_limit = query_limit
        self.n_remaining_query = self.init_query_limit

    def reset(self):
        self.state = self._wrapped_env.reset()
        self.state['extra.ans'] = 'None'
        self.prev_query, self.prev_ans = 'None', 'None'

        self.n_remaining_query = self.init_query_limit
        return self.state

    def step(self, command: str):
        command = command.lower()
        if not command.startswith("ask " + self.oracle_name.lower()):
            self.state, score, done = self._wrapped_env.step(command)
            if self.verbose:
                print('Env action:', command)

            self.state['extra.ans'] = self.prev_ans
            return self.state, score, done

        ans = 'I dont know'
        if self.verbose:
            print('Q:', command)

        if self.n_remaining_query <= 0:
            ans = 'limit'

        facts = self.state["facts"]

        # Mapping from ingredient ID to its actual name.
        ingredients = {fact.names[1]: fact.names[0] for fact in facts if fact.name == "base"}

        words = command.lower().split(" ")[2:]  # Skip "ask" and oracle's name.
        words = [word for word in words if word not in STOP_WORDS]
        if words[0] == "where":
            obj_name = " ".join(words[1:])

            matches = [fact for fact in facts if fact.name in ("at", "in", "on") and fact.names[0] == obj_name]
            if len(matches) == 1:
                fact = matches[0]
                if fact.names[1] == "I":
                    ans = "You have the {object_name} on you.".format(object_name=fact.names[0])
                else:
                    where = "on" if fact.name == "on" else "in"
                    ans = "The {object_name} is {where} the {holder}."
                    ans = ans.format(object_name=fact.names[0], where=where, holder=fact.names[1])

            elif len(matches) == 0:
                ans = "You won't find any {} in here.".format(obj_name)
                if obj_name == "kitchen":
                    ans = "You are standing in the middle of it."
            else:
                assert False, "Tell Marc."

        elif words[0] == "how":
            if words[1] == "cook":
                obj_name = " ".join(words[2:])
                if obj_name not in [info.name for info in self.state.game.infos.values()]:
                    ans = "The {object} cannot be cooked.".format(object=obj_name)
                else:
                    matches = [fact for fact in facts if fact.name in COOKED_STATES and ingredients.get(fact.names[0], fact.names[0]) == obj_name]
                    if matches:
                        ans = "The {object} needs to be {state}.".format(object=obj_name, state=matches[0].name)
                        heat_source = HEAT_SOURCES[COOKED_STATES.index(matches[0].name)]
                        ans += " Use {heat_source} to cook it.".format(heat_source=heat_source)
                    else:
                        ans = "The {object} doesn't need to be cooked.".format(object=obj_name)

            elif words[1] == "cut":
                obj_name = " ".join(words[2:])
                if obj_name not in [info.name for info in self.state.game.infos.values()]:
                    ans = "The {object} cannot be cut.".format(object=obj_name)
                else:
                    matches = [fact for fact in facts if fact.name in CUT_STATES and ingredients.get(fact.names[0], fact.names[0]) == obj_name]
                    if matches:
                        ans = "The {object} needs to be {state}.".format(object=obj_name, state=matches[0].name)
                        ans += " Use a knife to cut it."
                    else:
                        ans = "The {object} doesn't need to be cut.".format(object=obj_name)

            else:  # Assume 'how [object]'
                obj_name = " ".join(words[1:])
                if obj_name not in [info.name for info in self.state.game.infos.values()]:
                    ans = "The {object} cannot be cut nor cooked.".format(object=obj_name)
                else:
                    cooked_states = [fact for fact in facts if fact.name in COOKED_STATES and ingredients.get(fact.names[0], fact.names[0]) == obj_name]
                    cut_states = [fact for fact in facts if fact.name in CUT_STATES and ingredients.get(fact.names[0], fact.names[0]) == obj_name]

                    cooked_state = cooked_states[0].name if cooked_states else "uncooked"
                    cut_state = cut_states[0].name if cut_states else "uncut"
                    ans = "The {object} needs to be {cut_state} and {cooked_state}."
                    ans = ans.format(object=obj_name, cut_state=cut_state, cooked_state=cooked_state)

                    if cut_state != "uncut":
                        ans += " Use a knife to cut it."

                    if cooked_state != "uncooked":
                        heat_source = HEAT_SOURCES[COOKED_STATES.index(cooked_state)]
                        ans += " Use {heat_source} to cook it.".format(heat_source=heat_source)

        else:
            # Not a valid query.
            ans = "I don't understand the question."

        self.state, score, done = self._wrapped_env.step("look")  # No-op

        self.n_remaining_query -= 1
        self.state['extra.ans'] = ans
        self.state['feedback'] = ans
        if self.verbose:
            print('Ans:', ans)

        self.prev_query = command  # ' '.join(words) # TODO: needed?
        self.prev_ans = ans

        return self.state, score, done

    def copy(self) -> "TWOracleWrapper":
        """ Returns a copy this wrapper. """
        env = TWOracleWrapper()
        env._wrapped_env = self._wrapped_env.copy()
        env.oracle_name = self.oracle_name
        env.verbose = self.verbose
        env.prev_query, env.prev_ans = self.prev_query, self.prev_ans
        env.init_query_limit = self.init_query_limit
        env.n_remaining_query = env.n_remaining_query

        env._gamefile = self._gamefile
        env._game = self._game  # Reference

        return env


def _make_env(request_infos, settings, max_episode_steps=None):
    env = TWCookingAFK(request_infos, settings)
    if max_episode_steps:
        env = Limit(env, max_episode_steps=max_episode_steps)

    env = TWOracleWrapper(env, verbose=settings["verbose"])
    env = Filter(env)
    return env


class TWCookingAFKBatchGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'text']}

    def __init__(self,
                 settings: Dict,
                 request_infos: Optional[EnvInfos] = None,
                 batch_size: int = 1,
                 asynchronous: bool = True,
                 auto_reset: bool = False,
                 max_episode_steps: Optional[int] = None,
                 action_space: Optional[gym.Space] = None,
                 observation_space: Optional[gym.Space] = None) -> None:
        """ Environment for playing text-based games in batch.

        Arguments:
            settings:
                Settings controlling game generation.
            request_infos:
                For customizing the information returned by this environment
                (see
                :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
                for the list of available information).

                .. warning:: Only supported for TextWorld games (i.e., that have a corresponding `*.json` file).
            batch_size:
                If provided, it indicates the number of games to play at the same time.
                By default, a single game is played at once.

                .. warning:: When `batch_size` is provided (even for batch_size=1), `env.step` expects
                            a list of commands as input and outputs a list of states. `env.reset` also
                            outputs a list of states.
            asynchronous:
                If `True`, wraps the environments in an `AsyncBatchEnv` (which uses
                `multiprocessing` to run the environments in parallel). If `False`,
                wraps the environments in a `SyncBatchEnv`. Default: `True`.
            auto_reset:
                If `True`, each game *independently* resets once it is done (i.e., reset happens
                on the next `env.step` call).
                Otherwise, once a game is done, subsequent calls to `env.step` won't have any effects.
            max_episode_steps:
                Number of steps allocated to play each game. Once exhausted, the game is done.
            action_space:
                The action space be used with OpenAI baselines.
                (see :py:class:`textworld.gym.spaces.Word <textworld.gym.spaces.text_spaces.Word>`).
            observation_space:
                The observation space be used with OpenAI baselines
                (see :py:class:`textworld.gym.spaces.Word <textworld.gym.spaces.text_spaces.Word>`).
        """
        self.settings = settings
        self.batch_size = batch_size
        self.request_infos = request_infos or EnvInfos()

        env_fns = [partial(_make_env, self.request_infos, self.settings, max_episode_steps) for _ in range(self.batch_size)]
        BatchEnvType = AsyncBatchEnv if self.batch_size > 1 and asynchronous else SyncBatchEnv
        self.batch_env = BatchEnvType(env_fns, auto_reset)

        self.action_space = action_space
        self.observation_space = observation_space

        self.seed(1234)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """ Set the seed for this environment's random generator(s).

        This environment use a random generator to shuffle the order in which
        the games are played.

        Arguments:
            seed: Number that will be used to seed the random generators.

        Returns:
            All the seeds used to set this environment's random generator(s).
        """
        # We use the provided seed to generate different seed for each environment.
        rng = np.random.RandomState(seed)
        seeds = rng.randint(2147483647, size=self.batch_size).tolist()
        self.batch_env.seed(seeds)
        return seeds

    def reset(self) -> Tuple[List[str], Dict[str, List[Any]]]:
        """ Resets the text-based environment.

        Resetting this environment means starting the next game in the pool.

        Returns:
            A tuple (observations, infos) where

            * observation: text observed in the initial state for each game in the batch;
            * infos: additional information as requested for each game in the batch.
        """
        if self.batch_env is not None:
            self.batch_env.close()

        self.last_commands = [None] * self.batch_size
        self.ob, infos = self.batch_env.reset()
        return self.ob, infos

    def step(self, commands) -> Tuple[List[str], List[float], List[bool], Dict[str, List[Any]]]:
        """ Runs a command in each text-based environment of the batch.

        Arguments:
            commands: Text command to send to the game interpreter.

        Returns:
            A tuple (observations, scores, dones, infos) where

            * observations: text observed in the new state for each game in the batch;
            * scores: total number of points accumulated so far for each game in the batch;
            * dones: whether each game in the batch is finished or not;
            * infos: additional information as requested for each game in the batch.
        """
        self.last_commands = commands
        self.obs, scores, dones, infos = self.batch_env.step(self.last_commands)
        return self.obs, scores, dones, infos

    def close(self) -> None:
        """ Close this environment. """

        if self.batch_env is not None:
            self.batch_env.close()

        self.batch_env = None

    def render(self, mode: str = 'human') -> Optional[Union[StringIO, str]]:
        """ Renders the current state of each environment in the batch.

        Each rendering is composed of the previous text command (if there's one) and
        the text describing the current observation.

        Arguments:
            mode:
                Controls where and how the text is rendered. Supported modes are:

                    * human: Display text to the current display or terminal and
                      return nothing.
                    * ansi: Return a `StringIO` containing a terminal-style
                      text representation. The text can include newlines and ANSI
                      escape sequences (e.g. for colors).
                    * text: Return a string (`str`) containing the text without
                      any ANSI escape sequences.

        Returns:
            Depending on the `mode`, this method returns either nothing, a
            string, or a `StringIO` object.
        """
        outfile = StringIO() if mode in ['ansi', "text"] else sys.stdout

        renderings = []
        for last_command, ob in zip(self.last_commands, self.obs):
            msg = ob.rstrip() + "\n"
            if last_command is not None:
                command = "> " + last_command
                if mode in ["ansi", "human"]:
                    command = colorize(command, "yellow", highlight=False)

                msg = command + "\n" + msg

            if mode == "human":
                # Wrap each paragraph at 80 characters.
                paragraphs = msg.split("\n")
                paragraphs = ["\n".join(textwrap.wrap(paragraph, width=80)) for paragraph in paragraphs]
                msg = "\n".join(paragraphs)

            renderings.append(msg)

        outfile.write("\n-----\n".join(renderings) + "\n")

        if mode == "text":
            outfile.seek(0)
            return outfile.read()

        if mode == 'ansi':
            return outfile



def register_tw_cooking_afk(settings: Dict,
                            request_infos: Optional[EnvInfos] = None,
                            batch_size: Optional[int] = None,
                            auto_reset: bool = False,
                            max_episode_steps: int = 50,
                            asynchronous: bool = True,
                            action_space: Optional[gym.Space] = None,
                            observation_space: Optional[gym.Space] = None,
                            name: str = "cooking_afk",
                            **kwargs) -> str:
    """ Make an environment that will cycle through a list of games.

    Arguments:
        settings:
            Settings controlling game generation.
        request_infos:
            For customizing the information returned by this environment
            (see
            :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
            for the list of available information).

            .. warning:: Only supported for TextWorld games (i.e., with a corresponding `*.json` file).
        batch_size:
            If provided, it indicates the number of games to play at the same time.
            By default, a single game is played at once.

            .. warning:: When `batch_size` is provided (even for batch_size=1), `env.step` expects
                         a list of commands as input and outputs a list of states. `env.reset` also
                         outputs a list of states.
        auto_reset:
            If `True`, each game *independently* resets once it is done (i.e., reset happens
            on the next `env.step` call).
            Otherwise, once a game is done, subsequent calls to `env.step` won't have any effects.
        max_episode_steps:
            Number of steps allocated to play each game. Once exhausted, the game is done.
        asynchronous:
            If `True`, games in the batch are played in parallel. Only when batch size is greater than one.
        action_space:
            The action space be used with OpenAI baselines.
            (see :py:class:`textworld.gym.spaces.Word <textworld.gym.spaces.text_spaces.Word>`).
        observation_space:
            The observation space be used with OpenAI baselines
            (see :py:class:`textworld.gym.spaces.Word <textworld.gym.spaces.text_spaces.Word>`).
        name:
            Name for the new environment, i.e. "tw-{name}-v0". By default,
            the returned env_id is "tw-v0".

    Returns:
        The corresponding gym-compatible env_id to use.

    Example:

        >>> from textworld.generator import make_game, compile_game
        >>> options = textworld.GameOptions()
        >>> options.seeds = 1234
        >>> game = make_game(options)
        >>> game.extras["more"] = "This is extra information."
        >>> gamefile = compile_game(game)
        <BLANKLINE>
        >>> import gym
        >>> import textworld.gym
        >>> from textworld import EnvInfos
        >>> request_infos = EnvInfos(description=True, inventory=True, extras=["more"])
        >>> env_id = textworld.gym.register_games([gamefile], request_infos)
        >>> env = gym.make(env_id)
        >>> ob, infos = env.reset()
        >>> print(infos["extra.more"])
        This is extra information.

    """
    env_id = "tw-{}-v0".format(name) if name else "tw-v0"

    # If env already registered, bump the version number.
    if env_id in registry.env_specs:
        base, _ = env_id.rsplit("-v", 1)
        versions = [int(env_id.rsplit("-v", 1)[-1]) for env_id in registry.env_specs if env_id.startswith(base)]
        env_id = "{}-v{}".format(base, max(versions) + 1)

    entry_point = "tw_cooking_afk_env:TWCookingAFKBatchGymEnv"
    if batch_size is None:
        batch_size = 1

    register(
        id=env_id,
        entry_point=entry_point,
        kwargs={
            'settings': settings,
            'request_infos': request_infos,
            'batch_size': batch_size,
            'asynchronous': asynchronous,
            'auto_reset': auto_reset,
            'max_episode_steps': max_episode_steps,
            'action_space': action_space,
            'observation_space': observation_space,
            **kwargs}
    )
    return env_id
