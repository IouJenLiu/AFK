
from gym import RewardWrapper
import gym
import numpy as np
from ..info_seeking.knowledge_graph import KG
import random

class TransformReward(gym.Wrapper):
    def __init__(self, env):
        super(TransformReward, self).__init__(env)
        self.count = 0

    def step(self, action):
        self.count += 1
        obs, rewrd, done, info = super().step(action)
        if action == 5:
            if self.env.room_grid[0][2].objs:
                if (self.env.front_pos[0], self.env.front_pos[1]) == self.env.room_grid[0][0].door_pos[0]:
                    rewrd = -0.05
            else:
                if (self.env.front_pos[0], self.env.front_pos[1]) == self.env.room_grid[0][2].door_pos[2]:
                    rewrd = -0.05

        return obs, rewrd, done, info


class TargetLocationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TargetLocationWrapper, self).__init__(env)
        self.count = 0
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 4),
            dtype='uint8'
        )
        self.observation_space = gym.spaces.Dict({
            'image': self.observation_space
        })

    def observation(self, observation):
        if 'Room2' in self.env.mission:
            obs = np.concatenate((observation['image'], np.ones((7, 7, 1))), axis=2)
        else:
            obs = np.concatenate((observation['image'], np.zeros((7, 7, 1))), axis=2)
        observation['image'] = obs
        return observation


class DirectionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DirectionWrapper, self).__init__(env)
        self.count = 0
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 4),
            dtype='uint8'
        )
        self.observation_space = gym.spaces.Dict({
            'image': self.observation_space
        })

    def observation(self, observation):
        size = observation['image'].shape[0]
        obs = np.concatenate((observation['image'], self.env.agent_dir * np.ones((size, size, 1))), axis=2)
        observation['image'] = obs
        return observation


class XYLocationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(XYLocationWrapper, self).__init__(env)
        self.count = 0
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 5),
            dtype='uint8'
        )
        self.observation_space = gym.spaces.Dict({
            'image': self.observation_space
        })

    def observation(self, observation):
        obs = np.concatenate((observation['image'], self.env.agent_pos[0] * np.ones((7, 7, 1)), self.env.agent_pos[1] * np.ones((7, 7, 1))), axis=2)
        observation['image'] = obs
        return observation


import re

class AnsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(AnsWrapper, self).__init__(env)
        self.tokens = ['none',
                       'blue box', 'green box', 'grey box',
                       'blue key', 'green key', 'grey key',
                       ]
        n_channel = env.observation_space['image'].shape[-1] + len(self.tokens)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, n_channel),
            dtype='uint8'
        )
        self.observation_space = gym.spaces.Dict({
            'image': self.observation_space
        })

    def observation(self, observation):
        ans = re.findall("([a-z0-9]+)", observation['ans'].lower())
        ans_channel = None
        ans_channel = np.zeros((7, 7, len(self.tokens)))
        for i, token in enumerate(self.tokens):
            if token in ans:
                ans_channel[:, :, i] = 1
                break
        if ans_channel is None:
                raise ValueError
        obs = np.concatenate((observation['image'], ans_channel), axis=2)
        observation['image'] = obs
        return observation


class InstrWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(InstrWrapper, self).__init__(env)
        self.tokens = ['blue ball', 'green ball', 'grey ball', 'mary', 'jack']
        n_channel = env.observation_space['image'].shape[-1] + len(self.tokens)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, n_channel),
            dtype='uint8'
        )
        self.observation_space = gym.spaces.Dict({
            'image': self.observation_space
        })

    def observation(self, observation):
        ans = observation['mission'].lower()
        ans_channel = None

        for i, token in enumerate(self.tokens):
            if token in ans:
                ans_channel = np.zeros((7, 7, len(self.tokens)))
                ans_channel[:, :, i] = 1
                break
        if ans_channel is None:
            raise ValueError
        obs = np.concatenate((observation['image'], ans_channel), axis=2)
        observation['image'] = obs
        return observation

from collections import defaultdict
import math
class CountRewardWrapper(gym.Wrapper):
    def __init__(self, env, alpha=1, count_action=False):
        super(CountRewardWrapper, self).__init__(env)
        self.memory = defaultdict(int)
        self.alpha = alpha
        self.count_action = count_action
        self.mini_grid_actions_map = {'left': 0, 'right': 1, 'forward': 2, 'pickup': 3, 'drop': 4, 'toggle': 5,
                                      'done': 6}

    def step(self, action):
        obs, reward, done, info = super().step(action)
        tuple_obs = tuple(obs['image'].reshape(1, -1)[0])
        self.memory[tuple_obs] += 1
        reward += self.alpha / math.sqrt(self.memory[tuple_obs])
        return obs, reward, done, info


class KGWrapper(gym.Wrapper):
    """
    A wrapper that returns the connected component of the KG in observation.
    kg_repr = [one_hot, raw]
    one_hot: each sentence is encoded as an onehot channel of the image
    raw: return all raw sentences as a list in observation['kg_cc']
    """
    def __init__(self, env, penalize_query=False, cc_bonus=0.05, weighted_bonus=False, kg_repr='one_hot', mode='graph', n_gram=2,
                 distractor_file_path=None, n_distractors=0, node_sample_mode='fixed', args=None):
        super(KGWrapper, self).__init__(env)
        self.kg_repr = kg_repr
        n_channel = env.observation_space['image'].shape[-1]
        self.moving_actions = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
        self.colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, n_channel),
            dtype='uint8'
        )
        self.observation_space = gym.spaces.Dict({
            'image': self.observation_space
        })
        mode = 'set' if mode == 'no_kg' else mode
        self.KG = KG(mode=mode, n_gram=n_gram)
        self.cc_bonus = cc_bonus
        self.penalize_query = penalize_query
        if self.penalize_query:
            self.query_penalty = -0.01
        self.weighted_bonus = weighted_bonus
        if distractor_file_path:
            # Generate on the fly
            self.distractors = True
        else:
            self.distractors = False
        self.total_frames_per_proc = args.frames // args.procs
        self.cur_total_frames = 0
        self.decrease_bonus = args.decrease_bonus

    def bonus_coef(self):
        if not self.decrease_bonus:
            return 1
        anneal_th = 0.6 * self.total_frames_per_proc
        if self.cur_total_frames <= anneal_th:
            return 1
        else:
            return 1.05 - (self.cur_total_frames - anneal_th) / (self.total_frames_per_proc - anneal_th)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if isinstance(action, list) and len(action) > 1 and action[0] not in self.moving_actions:
            for ans in obs['ans'].split(','):
                is_CC_increase, overlap = self.KG.update(self.pre_proc_asn(ans))
                if is_CC_increase:
                    if self.weighted_bonus:
                        reward += self.bonus_coef() * self.cc_bonus * overlap
                    else:
                        reward += self.bonus_coef() * self.cc_bonus
            if self.penalize_query:
                reward += self.query_penalty
        obs = self.observation(obs, self.KG.getCC())
        self.cur_total_frames += 1
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.KG.reset(self.pre_proc_asn(obs['mission']))
        if self.distractors:
            new_nodes = self.unwrapped.useful_answers + self.gen_distractors()
            random.shuffle(new_nodes)
            for new_node in new_nodes:
                split_node = new_node.split()
                if len(self.unwrapped.useful_answers) > 2:
                    split_ans = self.unwrapped.useful_answers[2].split()
                    if len(split_node) == 4 and split_node[0] == split_ans[0] and split_node[1] == split_ans[1]:
                        continue
                self.KG.update(self.pre_proc_asn(new_node))

        obs = self.observation(obs, self.KG.getCC())
        return obs

    def gen_distractors(self):
        names = ['tim', 'allen', 'tom', 'jack', 'mary']
        objs = ['suitcase', 'toy']
        colors = ['purple', 'orange', 'blue', 'green', 'gray', 'grey', 'yellow', 'red', 'white', 'pink']
        shapes = ['box', 'ball', 'key']
        distractors = []
        for name in names:
            for obj in objs:
                color = random.choice(colors)
                shape = random.choice(shapes)
                distractors.append('{} {} {} {}'.format(name, obj, color, shape))
        places = ['livingroom', 'kitchen', 'restroom']
        rooms = ['room0', 'room1', 'room2', 'room3', 'room4', 'room5', 'room6', 'room7', 'room8']
        for name in names:
            place = random.choice(places)
            room = random.choice(rooms)
            distractors.append('{} {} {}'.format(name, place, room))

        for name in names:
            for color in colors:
                for shape in objs:
                    place = random.choice(places)
                    distractors.append('{} {} {} in {}'.format(name, color, shape, place))
        directions = ['east', 'west']
        for color in colors:
            for room in rooms:
                dir = random.choice(directions)
                distractors.append('{} {} in {}'.format(color, room, dir))



        random.shuffle(distractors)
        return distractors





    def observation(self, observation, CC):
        """
        :param observation: dictionary
        :param CC: list of tuples
        :return: modified observation
        """
        if self.kg_repr == 'one_hot':
            ans_channel = np.zeros((7, 7, len(self.tokens)))
            for ans in CC:
                for i, token in enumerate(self.tokens):
                    if token == ans:
                        ans_channel[:, :, i] = 1
                        break
            obs = np.concatenate((observation['image'], ans_channel), axis=2)
            observation['image'] = obs
        elif self.kg_repr == 'raw':
            raw_repr = []
            for node in CC:
                raw_repr.append(' '.join(node))
            observation['kg_cc'] = raw_repr
        else:
            raise NotImplementedError

        return observation

    def pre_proc_asn(self, ans):
        ans = re.findall("([a-z0-9]+)", ans.lower())
        if 'is' in ans:
            ans.remove('is')
        if 'in' in ans:
            ans.remove('in')
        return ans



class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.eps_steps = 0
        self.action = None

    def step(self, action):
        self.action = action
        self.eps_steps += 1
        return super().step(action)

    def reset(self):
        self.eps_steps = 0
        self.action = None
        return super().reset()


    # Size in pixels of a tile in the full-scale human view
    TILE_PIXELS = 32

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS, KG=None):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.ax.xaxis.label.set_fontsize(10)
            self.window.fig.subplots_adjust(top=1.0, bottom=0.3)
            self.window.show(block=False)


        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * (self.agent_view_size - 1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            prev_query = self.prev_query if hasattr(self, 'prev_query') else ""
            prev_ans = self.prev_ans if hasattr(self, 'prev_ans') else ""
            caption = 'Instr: {} step: {} action: {}\n'.format(self.mission, self.eps_steps, self.action)
            in_bos = 'Q: ' + prev_query + ' A: ' + prev_ans + "\n"
            self.window.set_caption(caption + in_bos)
            self.window.show_img(img)


        return img



