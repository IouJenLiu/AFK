import gym
import re
import numpy as np
import time
other_places = ['kitchen', 'restroom', 'livingroom']
other_colors = ['red', 'yellow', 'white']
other_directions = ['east', 'west', 'second floor']
import random
import math
from ..utils.format import Vocabulary
import copy

class ObjInBoxMulti_MultiOracle_Query(gym.Wrapper):
    def __init__(self, env, verbose=False, restricted=False, flat=False, n_q=1, query_limit=100, reliability=1, talk_th=1, call=True, random_ans=False):
        super(ObjInBoxMulti_MultiOracle_Query, self).__init__(env)
        self.flat = flat
        self.random_ans = random_ans
        if self.flat:
            self.action_space = gym.spaces.Discrete((self.env.action_space.n) + n_q)
        else:
            self.action_space = gym.spaces.MultiDiscrete(
                [self.env.action_space.n, n_q])
        self.mini_grid_actions_map = {'left': 0, 'right': 1, 'forward': 2, 'pickup': 3, 'drop': 4, 'toggle': 5,
                                      'done': 6}
        self.verbose = verbose
        self.prev_query, self.prev_ans = 'None', 'None'
        self.eps_steps = 0
        self.eps_n_q = 0
        self.action = 'None'
        self.n_distractors = 2
        self.talk_th = talk_th
        self.call = call

        if self.random_ans:
            self.vocab = list(Vocabulary('test').vocab.keys())
            shuffled_vocab = copy.deepcopy(self.vocab)
            random.shuffle(shuffled_vocab)
            self.word_mapping = {}
            for i in range(len(self.vocab)):
                self.word_mapping[self.vocab[i]] = shuffled_vocab[i]



    def reset(self, **kwargs):
        obs = super().reset()
        self.other_objs = [{'color': np.random.choice(other_colors),
                            'place': np.random.choice(other_places),
                            'direction': np.random.choice(other_directions)} for _ in range(self.n_distractors)]

        obs['ans'] = 'None'
        self.prev_query, self.prev_ans = 'None', 'None'
        self.eps_steps = 0
        self.action = 'None'

        if self.random_ans:
            shuffled_vocab = copy.deepcopy(self.vocab)
            random.shuffle(shuffled_vocab)
            self.word_mapping = {}
            for i in range(len(self.vocab)):
                self.word_mapping[self.vocab[i]] = shuffled_vocab[i]

        return obs

    def step(self, action):
        """
        :param action:
        (1) minigrid action: 0 - 6
        (2) query action:
        ['where', 'obj / container'] or ['where', 'adj', 'obj / container']

        :return: obs, reward, done, info
        obs = {img, instr, ans}
        ans = [('adj'), 'obj / container', 'in', 'direction / container']
        e.g.
        Query: where red box
        Ans: red box in room1
        Query: where room1
        Ans: room1 in south
        """
        self.action = action
        self.eps_steps += 1
        assert isinstance(action, list)
        if isinstance(action[0], list):
            action = action[0]

        if action[0] in self.mini_grid_actions_map:
            if self.verbose:
                print('MiniGrid action:', action[0])
            obs, rewrd, done, info = super().step(self.mini_grid_actions_map[action[0]])
            obs['ans'] = self.prev_ans
            return obs, rewrd, done, info

        obs, rewrd, done, info = super().step(action=6)

        ans = 'I　dont　know'
        self.eps_n_q += 1

        if self.verbose:
            print('Q:', action)
        linear_idx = self.env.front_pos[1] * self.env.grid.width + self.env.front_pos[0]
        front_cell = self.env.grid.grid[linear_idx]
        try:
            if action[0] == 'where' and  (self.call or not hasattr(self.env, 'oracle') or self.distance(self.env.agent_pos, self.env.oracle.cur_pos) <= self.talk_th):
                if action[1] == 'is' and action[2] in [obj['place'] for obj in self.other_objs]:
                    for i, other_obj in enumerate(self.other_objs):
                        if action[2] == other_obj['place']:
                            idx = i
                    ans = action[2] + ' in {}'.format(self.other_objs[idx]['direction'])
                elif len(action) == 4:
                    color, type = action[2], action[3]
                    ans, _ = self.get_ans(color=color, type=type)
                elif len(action) == 3:
                    color, type = action[1], action[2]
                    ans, _ = self.get_ans(color=color, type=type)
                elif len(action) == 2:
                    type = action[1]
                    if 'room' in type:
                        target_room_id = int(re.split('(\d+)', type)[1])
                        if target_room_id < self.env.num_cols * self.env.num_rows:
                            ans = self.get_ans_room(target_room_id)
                    else:
                        ans, _ = self.get_ans(type=type)
            elif action[0] == 'what':
                if (action[2] == 'key' or action[3] == 'key') and hasattr(self.env, 'real_key_color') and self.is_locked_door(tuple(self.env.front_pos)):
                    ans = self.env.real_key_color + ' key to the door'
                elif action[2] == 'danger':
                    if self.call:
                        ans = 'danger zone is ' + self.env.lava_colors[1 - self.env.target_idx]
                    else:
                        assert hasattr(self.env, 'oracle_pos')
                        if self.distance(self.env.agent_pos, self.env.oracle_pos) <= self.talk_th:
                            ans = 'danger zone is ' + self.env.lava_colors[1 - self.env.target_idx]
                elif len(action) == 4 and action[3] == 'suitcase':
                    if action[2] in ['jack', 'mary'] and 'room' not in action[3]:
                        if self.call or (action[2] == 'jack' and self.distance(self.env.agent_pos, self.env.jack_pos) <= self.talk_th) or \
                                (action[2] == 'mary' and self.distance(self.env.agent_pos, self.env.mary_pos) <= self.talk_th):
                            if self.env.unwrapped.box_owner == action[2]:
                                ans = 'different from ' + 'tim,' + ' ' + action[2] + ' suitcase ' + self.env.unwrapped.target_box.color + ' box'
                elif len(action) > 2:
                    if action[2] in ['jack', 'mary'] and 'room' not in action[3]:
                        if self.call or (action[2] == 'jack' and hasattr(self.env, 'jack_pos') and self.distance(self.env.agent_pos, self.env.jack_pos) <= self.talk_th) or \
                                (action[2] == 'mary' and hasattr(self.env, 'mary_pos') and self.distance(self.env.agent_pos, self.env.mary_pos) <= self.talk_th) or \
                                (self.distance(self.env.agent_pos, self.env.oracle.cur_pos) <= self.talk_th):
                            #if front_cell and front_cell.type == 'ball' and ((action[2] == 'jack' and front_cell.color == 'yellow') or (action[2] == 'mary' and front_cell.color == 'red')):
                            if action[2] in self.unwrapped.mission:
                                ans = action[2] + ' toy is ' + self.unwrapped.instrs.desc.color + ' ' + self.unwrapped.instrs.desc.type
                            elif hasattr(self.unwrapped, 'others_fav'):
                                ans = action[2] + ' toy is ' + self.unwrapped.others_fav.color + ' ' + self.unwrapped.others_fav.type
        except:
            ans = 'i dont know'
        obs['ans'] = ans
        if self.verbose:
            print('Ans:', ans)
        self.prev_query = ' '.join(action)
        self.prev_ans = ans
        if self.random_ans:
            ans = ans.split()
            for i in range(len(ans)):
                w = ans[i]
                if w not in self.word_mapping: continue
                ans[i] = self.word_mapping[w]
            obs['ans'] = ' '.join(ans)
        return obs, rewrd, done, info

    def distance(self, loc1, loc2):
        return np.sum((loc1 - loc2) ** 2).item()


    def is_locked_door(self, pos):
        for room_row in self.env.room_grid:
            for room in room_row:
                for door in room.doors:
                    if door is not None and door.cur_pos == pos and door.is_locked:
                        return True
        return False

    def get_ans_room(self, target_room_id):
        agent_room_row, agent_room_col = self.env.agent_pos[1] // (self.env.room_size - 1), self.env.agent_pos[0] // (
                self.env.room_size - 1)
        # print(agent_room_row, agent_room_col)
        target_room_row, target_room_col = target_room_id // self.env.num_cols, target_room_id % self.env.num_rows
        dir = (np.clip(target_room_col - agent_room_col, -1, 1), np.clip(target_room_row - agent_room_row, -1, 1))
        dir_map = {(-1, -1): 'northwest', (0, -1): 'north', (1, -1): 'northeast',
                   (-1, 0): 'west', (0, 0): 'here', (1, 0): 'east',
                   (-1, 1): 'southwest', (0, 1): 'south', (1, 1): 'southeast'}
        return dir_map[dir]

    def get_ans(self, type, color=None):
        room_id = 0
        ans = 'I dont know'
        for room_row in self.env.room_grid:
            for room in room_row:
                for obj in room.objs:
                    if obj.type == type:
                        if color is None:
                            ans = type + ' in room{}'.format(room_id)
                            return ans, room_id
                        elif obj.color == color:
                            ans = color + ' ' + type + ' in room{}'.format(room_id)
                            return ans, room_id
                    if obj.type == 'box' and obj.contains:
                        if hasattr(self.env.unwrapped, 'box_owner'):
                            if obj.contains.type == type and obj.contains.color == color and color == self.env.unwrapped.instrs.desc.color:
                                other_obj = np.random.choice(self.other_objs)
                                ans = 'while {} ball in the {}, '.format(other_obj['color'], other_obj['place']) + color + ' ' + type + ' in ' + self.env.unwrapped.box_owner + ' suitcase'
                        elif obj.contains.type == type and obj.contains.color == color:
                            ans = color + ' ' + type + ' in ' + obj.color + ' ' + obj.type

                for door in room.doors:
                    if door is not None and color is not None:
                        if door.type == type and door.color == color:
                            ans = color + ' ' + type + ' in room{}'.format(room_id)
                            return ans, room_id
                room_id += 1
        #ans = 'red ball in room2'
        return ans, room_id

    def get_obj(self, type):
        for room_row in self.env.room_grid:
            for room in room_row:
                for obj in room.objs:
                    if obj.type == type:
                        return obj
        return None





if __name__ == "__main__":
    import babyai
    from babyai.utils.wrapper import RenderWrapper
    #env_name = 'BabyAI-ObjInBoxMultiSufficient-v0'
    #env_name = 'BabyAI-ObjInBoxMultiMultiOracle-v0'
    #env_name = 'BabyAI-Danger-v0'
    #env_name = 'BabyAI-OpenDoorMultiKeys-v0'
    #env_name = 'BabyAI-ObjInBoxV2-v0'
    #env_name = 'BabyAI-ObjInBoxV2-v0'
    #actions = [
    #    ['what', 'is', 'jack', 'favorite', 'toy'],
    #    ['what', 'is', 'mary', 'favorite', 'toy'],
    #    ['what', 'is', 'mary', 'suitcase'],
    #    ['what', 'is', 'jack', 'suitcase']
    #]

    env_name = 'BabyAI-DangerBalancedDiverse-v0'
    seed = 2
    succ = True
    if succ:
        actions = [
            ['what', 'is', 'danger', 'zone'],
            ['what', 'is', 'jack', 'toy'],
            ['where', 'is', 'red', 'ball'],
            ['right'], ['forward'],  ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'],
            ['left'], ['forward'], ['forward'], ['forward'], ['forward'], ['left'], ['forward'],
            ['forward'], ['forward'], ['forward'], ['forward'],['forward'],
            ['left'], ['forward'], ['forward'], ['forward'], ['forward']
        ]
    else:
        actions = [
            ['right'], ['forward'],  ['forward'], ['forward'], ['forward'],
        ]


    """
    env_name = 'BabyAI-ObjInBoxV2SingleMove-v0'
    actions = [
        ['right'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'],
        ['right'], ['forward'], ['forward'],
        ['what', 'is', 'jack', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'favorite', 'toy'],
        ['where', 'is', 'green', 'ball'],
        ['what', 'is', 'mary', 'suitcase'],
        ['what', 'is', 'jack', 'suitcase'],
        ['right'], ['forward'], ['forward'], ['forward'],
        ['left'], ['forward'], ['toggle']

    ]
    """
    """
    env_name = 'BabyAI-ObjInBoxV2MultiMove-v0'
    actions = [
        ['right'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'],
        #['right'], ['forward'], ['forward'],
        ['what', 'is', 'jack', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'suitcase'],
        ['what', 'is', 'jack', 'suitcase'],
        ['where', 'is', 'blue', 'ball'],
        ['right'], ['forward'], ['forward'],
        ['what', 'is', 'jack', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'suitcase'],
        ['what', 'is', 'jack', 'suitcase'],

    ]
    """

    """
    env_name = 'BabyAI-GoToFavoriteSingleMove-v0'
    actions = [
        ['forward'], ['forward'], ['forward'],
        ['what', 'is', 'jack', 'toy'],
        ['what', 'is', 'mary', 'toy'],
        ['where', 'is', 'blue', 'box'],
        ['where', 'is', 'blue', 'ball'],
        ['where', 'is', 'green', 'box'],
        ['where', 'is', 'green', 'ball'],

        ['right'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'],
        #['right'], ['forward'], ['forward'],
        ['what', 'is', 'jack', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'suitcase'],
        ['what', 'is', 'jack', 'suitcase'],
        ['right'], ['forward'], ['forward'],
        ['what', 'is', 'jack', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'suitcase'],
        ['what', 'is', 'jack', 'suitcase'],

    ]
    """



    """
    actions = [
        ['forward'], ['forward'], ['forward'], ['forward'], ['left'],
        ['what', 'is', 'key'],
        ['left'], ['forward'], ['forward'], ['forward'], ['forward'],
        ['left'], ['forward'], ['forward'], ['forward'], ['pickup'],
        ['left'], ['left'], ['forward'], ['forward'], ['forward'], ['forward'],
        ['right'], ['forward'], ['forward'], ['forward'], ['forward'],  ['left'], ['toggle']

    ]
    """

    """
    env_name = 'BabyAI-GoToFavorite-v0'
    seed = 0
    actions = [
        ['where', 'is', 'room2'],
        ['right'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'],
        #['right'], ['forward'], ['forward'],
        ['what', 'is', 'jack', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'suitcase'],
        ['what', 'is', 'jack', 'suitcase'],
        ['where', 'is', 'blue', 'ball'],
        ['right'], ['forward'], ['forward'],
        ['what', 'is', 'jack', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'suitcase'],
        ['what', 'is', 'jack', 'suitcase'],

    ]
    """

    """
    env_name = 'BabyAI-OpenDoorMultiKeys-v0'
    seed = 1
    actions = [
        ['right'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'],
        #['right'], ['forward'], ['forward'],
        ['what', 'is', 'jack', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'suitcase'],
        ['what', 'is', 'jack', 'suitcase'],
        ['where', 'is', 'blue', 'ball'],
        ['right'], ['forward'], ['forward'],
        ['what', 'is', 'jack', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'favorite', 'toy'],
        ['what', 'is', 'mary', 'suitcase'],
        ['what', 'is', 'jack', 'suitcase'],

    ]
    """

    #env_name = 'BabyAI-DangerBalancedDiverseMove-v0'
    #env_name = 'BabyAI-ObjInBoxV2SingleMove-v0'
    #env_name = 'BabyAI-ObjInBoxV2MultiMove-v0'
    #env_name = 'BabyAI-ObjInBoxV2-v0'
    #actions = [['what', 'is', 'mary', 'toy'], ['what', 'is', 'jack', 'toy'], ['right'], ['right'], ['forward'], ['forward'],['forward'],
    #           ['right'], ['forward'], ['forward'], ['forward'],
    #           #['right'], ['forward'],['forward'], ['forward'],
    #           ['what', 'is', 'mary', 'toy'], ['what', 'is', 'jack', 'toy'], ['where', 'is', 'green', 'ball'], ['what', 'is', 'mary', 'suitcase'],
    #           ['right'], ['right'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['left'],
    #           ['what', 'is', 'mary', 'suitcase']]


    """
    env_name = 'BabyAI-DangerOpenDoor-v0'
    seed = 12
    succ = True
    if succ:
        actions = [['what', 'is', 'danger', 'zone'], ['left'],['left'],
                   ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['right'],
                   ['forward'], ['forward'], ['forward'], ['forward'], ['forward'],
                   ['what', 'is', 'key'],
                   ['right'], ['right'],
                   ['pickup'],
                   ['left'], ['left'], ['toggle'],
                   ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'],
                   ['right'],
                   ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward']
                   ]
    else:
        actions = [
            ['forward'], ['left'], ['forward'],  ['forward'],  ['forward'],  ['forward'],  ['forward'],
            ['right'], ['right'], ['forward'], ['forward'], ['forward']
        ]
    """




    env_name = 'BabyAI-OpenDoorObjInBox-v0'
    seed = 1
    actions = [
                ['right'], ['right'], ['forward'], ['forward'],
                ['right'], ['forward'], ['forward'], ['forward'], ['forward'],
                ['what', 'is', 'key'],
                ['left'], ['left'],
                ['forward'], ['forward'],
                ['left'], ['pickup'],
                ['left'],
                ['forward'], ['forward'], ['forward'], ['forward'], ['toggle'],
                ['what', 'is', 'mary', 'toy'],
                ['where', 'is', 'blue', 'ball'],
                ['what', 'is', 'jack', 'suitcase'],
                ['forward'], ['forward'], ['forward'], ['forward'], ['forward'],
                ['right'],
                ['forward'], ['forward'], ['forward'], ['toggle']
               ]
    
    """
    env_name = 'BabyAI-OpenDoorGoToFavorite-v0'
    seed = 3
    actions = [
        ['what', 'is', 'mary', 'toy'],
        ['where', 'is', 'blue', 'box'],
        ['forward'], ['forward'], ['forward'],
        ['left'],
        ['forward'], ['forward'], ['forward'], ['right'],
        ['toggle'], ['forward'], ['forward'], ['left'],
        ['what', 'is', 'key'],
        ['left'], ['forward'],
        ['pickup'],
        ['right'], ['right'],
        ['forward'], ['left'], ['toggle'],
        ['forward'], ['forward'], ['forward'], ['forward'], ['forward'],
        ['right'],
        ]
    """
    """
    env_name = 'BabyAI-GoToFavoriteObjInBox-v0'
    seed = 1
    actions = [
        ['what', 'is', 'mary', 'toy'],
        ['where', 'is', 'blue', 'ball'],
        ['what', 'is', 'jack', 'suitcase'],
        ['where', 'is', 'blue', 'box'],
        ['left'], ['forward'], ['toggle'],
        
    ]
    """
    """
    env_name = 'BabyAI-DangerObjInBox-v0'
    seed = 1
    succ = False
    if succ:
        actions = [
            ['what', 'is', 'danger', 'zone'],
            ['what', 'is', 'mary', 'toy'],
            ['where', 'is', 'blue', 'ball'],
            ['what', 'is', 'jack', 'suitcase'],
            ['right'], ['forward'],
            ['forward'], ['left'], ['forward'], ['forward'], ['forward'], ['left'], ['forward'], ['right'], ['forward'], ['left'],
            ['forward'], ['forward'], ['forward'], ['toggle']
        ]
    else:
        actions = [
            ['left'], ['forward'], ['forward'], ['forward']
        ]
    """

    """
    #env_name = 'BabyAI-DangerGoToFavNeg-v0'
    #env_name = 'BabyAI-DangerGoToFavSymmetry-v0'
    env_name = 'BabyAI-DangerGoToFav2Room-v0'
    seed = 10
    succ = True
    if succ:
        actions = [
            ['what', 'is', 'danger', 'zone'],
            ['what', 'is', 'jack', 'toy'],
            ['where', 'is', 'red', 'ball'],
            ['left'], ['forward'],  ['forward'], ['forward'], ['forward'],
            ['right'], ['forward'], ['forward'], ['forward'], ['forward'], ['left'], ['forward'],
            ['forward'], ['forward'], ['forward'], ['forward'],['forward'],
            ['left'], ['forward'], ['forward'], ['forward'], ['forward']
        ]
    else:
        actions = [
            ['right'], ['forward'],  ['forward'], ['forward'], ['forward'],
        ]
    """

    """
    env_name = 'BabyAI-DangerGoToFav2Room-v0'
    seed = 10
    succ = True
    if succ:
        actions = [
            ['what', 'is', 'danger', 'zone'],
            ['what', 'is', 'jack', 'toy'],
            ['where', 'is', 'red', 'ball'],
            ['left'], ['forward'], ['right'], ['forward'], ['forward'], ['left'],
            ['forward'], ['forward'], ['forward'], ['right'], ['forward'],
            ['forward'], ['forward'], ['left'], ['forward'],['forward'],
            ['left'], ['forward'], ['forward'], ['forward'], ['forward']
        ]
    else:
        actions = [
            ['right'], ['forward'],  ['forward'], ['forward'], ['forward'],
        ]


    """
    """
    env_name = 'BabyAI-DangerGoToFavOpenDoor-v0'
    seed = 1
    succ = True
    if succ:
        actions = [
            ['what', 'is', 'danger', 'zone'],
            ['what', 'is', 'jack', 'toy'],
            ['where', 'is', 'grey', 'box'],
            ['right'],
            ['what', 'is', 'key'],
            ['left'],
            ['forward'],
            ['forward'], ['left'], ['forward'], ['right'],
            ['forward'], ['forward'], ['forward'], ['right'], ['pickup'],
            ['right'], ['forward'], ['forward'], ['left'],
            ['forward'], ['right'], ['forward'], ['forward'], ['forward'], ['left'], ['toggle'],
            ['forward'], ['forward'], ['left'], ['forward'], ['forward'], ['forward'], ['forward'],['right'],
            ['forward'], ['forward'],['forward'], ['forward']
        ]
    else:
        actions = [
            ['right'], ['forward'],  ['forward'], ['forward'], ['forward'],
        ]
    """
    """
    env_name = 'BabyAI-OpenDoorGoToFavoriteObjInBox-v0'
    seed = 15
    succ = True
    if succ:
        actions = [
            ['what', 'is', 'mary', 'toy'],
            ['where', 'is', 'green', 'ball'],
            ['what', 'is', 'jack', 'suitcase'],
            ['where', 'is', 'green', 'box'],
            ['toggle'],
            ['forward'],['forward'],['forward'], ['forward'],
            ['left'], ['forward'], ['forward'], ['forward'],
            ['right'],
            ['forward'], ['forward'], ['forward'],['toggle'],
            ['what', 'is', 'key'],
            ['right'],
            ['forward'], ['forward'], ['forward'],
            ['right'], ['forward'], ['forward'], ['forward'], ['forward'], ['right'],
            ['forward'], ['pickup'],
            ['right'], ['right'],
            ['forward'], ['forward'], ['left'],
            ['forward'],['forward'],['forward'], ['forward'],
            ['left'], ['forward'], ['forward'], ['forward'],
            ['right'],
            ['forward'], ['forward'], ['forward'],['toggle'],
            ['forward'],['forward'],['forward'], ['right'],
            ['forward'], ['toggle'],

        ]
    else:
        actions = [
            ['right'], ['forward'],  ['forward'], ['forward'], ['forward'],
        ]
    """


    # DangerGoToFavObjInBox
    """
    env_name = 'BabyAI-DangerGoToFavObjInBox-v0'
    seed = 15
    succ = False
    if succ:
        actions = [
            ['what', 'is', 'mary', 'toy'],
            ['where', 'is', 'blue', 'ball'],
            ['what', 'is', 'jack', 'suitcase'],
            ['where', 'is', 'green', 'box'],
            ['what', 'is', 'danger', 'zone'],
            ['left'], ['forward'], ['forward'], ['forward'],
            ['right'],
            ['forward'], ['forward'], ['forward'],
            ['left'], ['forward'], ['forward'], ['forward'],
            ['right'], ['toggle']
        ]
    else:
        actions = [
            ['left'], ['forward'],  ['forward'], ['forward'], ['forward'],
        ]
    """

    # DangerOpenDoorObjInBox
    """
    env_name = 'BabyAI-DangerOpenDoorObjInBox-v0'
    seed = 15
    succ = True
    if succ:
        actions = [
            ['what', 'is', 'mary', 'toy'],
            ['where', 'is', 'green', 'ball'],
            ['what', 'is', 'jack', 'suitcase'],
            ['where', 'is', 'green', 'box'],
            ['what', 'is', 'danger', 'zone'],
            ['left'], ['forward'], ['forward'], ['forward'], ['forward'],
            ['left'],
            ['forward'], ['forward'], ['forward'], ['forward'],
            ['right'],
            ['what', 'is', 'key'],
        ]
    else:
        actions = [
            ['left'], ['forward'],  ['forward'], ['forward'], ['forward'],
        ]
    """
    """
    # Boss
    env_name = 'BabyAI-Boss-v0'
    seed = 1
    succ = True
    if succ:
        actions = [
            ['what', 'is', 'danger', 'zone'],
            ['what', 'is', 'jack', 'toy'],
            ['where', 'is', 'green', 'ball'],
            ['what', 'is', 'mary', 'suitcase'],
            ['where', 'is', 'green', 'box'],
            ['forward'], ['right'], ['forward'], ['forward'],
            ['forward'], ['forward'],
            ['left'], ['forward'], ['right'],
            ['what', 'is', 'key'],
            ['where', 'is', 'green', 'key'],
            ['left'], ['forward'], ['forward'], ['left'], ['forward'], ['forward'], ['right'], ['toggle'],
            ['forward'], ['forward'], ['forward'], ['forward'], ['right'], ['forward'], ['forward'], ['forward'], ['toggle'],
            ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['forward'], ['toggle'],['forward'], ['forward'],
            ['left'], ['forward'], ['right'], ['forward'], ['forward'], ['forward'], ['forward'], ['left'], ['pickup'], ['left'],
            ['forward'], ['forward'], ['forward'], ['forward'], ['left'], ['forward'], ['right'], ['forward'], ['forward'], ['forward'], ['forward'],
            ['left'], ['forward'], ['forward'], ['toggle'], ['forward'], ['forward'], ['right'], ['forward'], ['forward'], ['forward'], ['left'],
            ['forward'], ['forward'], ['left'], ['toggle']
        ]
    else:
        actions = [
            ['left'], ['forward'],  ['forward'], ['forward'], ['forward'],
        ]

    """

    is_render = True
    env = gym.make(env_name)
    from query import Query
    #env = Query(env, verbose=True)
    env = ObjInBoxMulti_MultiOracle_Query(env, verbose=True, query_limit=20, call=True)
    env = RenderWrapper(env)
    #for seed in range(1, 1000):

    obs = env.seed(seed)
    env.reset()
    #sys.exit()
    print(env.unwrapped.mission)
    if is_render:
        env.render("human"); time.sleep(10)
    #env.step(2); time.sleep(1)
    #env.step(2); time.sleep(1)
    #env.step(2); time.sleep(1)
    #env.step(2); time.sleep(1)
    #env.step(2); time.sleep(1)
    #env.step(2); time.sleep(1)
    #time.sleep(10)

    #env.step(['what', 'is', 'jack', 'toy']);env.render("human"); time.sleep(5)
    #env.step(['left']);env.render("human"); time.sleep(1)
    done = False
    #actions = [['what', 'is', 'danger', 'zone'], ['left'], ['forward'], ['forward'], ['forward'], ['what', 'is', 'danger', 'zone'], ['left'], ['forward'], ['forward'],  ['forward'], ['right'], ['what', 'is', 'key'], ['left'], ['left'], ['forward'], ['forward'], ['left'], ['pickup'], ['left'], ['forward'], ['forward'], ['forward'], ['forward'], ['toggle']]
    i = 0
    #env.step(['what', 'is', 'danger', 'zone']);
    #env.step(['what', 'is', 'key']);
    while True:
        obs, rew, done, _ = env.step(actions[i])
        if is_render:
            env.render("human"); time.sleep(0.4)
        print(done, rew)
        i += 1
        if done:
            break
    import sys
    sys.exit()

    obs, _, done, _ = env.step(['forward']);env.render("human"); time.sleep(1)
    print(done)
    _, _, done, _ = env.step(['right']);env.render("human"); time.sleep(1)
    print(done)
    _, _, done, _ = env.step(['forward']);env.render("human"); time.sleep(1)
    print(done)
    _, _, done, _ = env.step(['forward']);env.render("human"); time.sleep(1)
    print(done)
    _, _, done, _ = env.step(['forward']);env.render("human"); time.sleep(1)
    print(done)
    env.step(['forward']);env.render("human"); time.sleep(1)

    env.step(['forward']);env.render("human"); time.sleep(1)
    env.step(['forward']);env.render("human"); time.sleep(1)
    env.step(['what', 'is', 'mary', 'toy']);env.render("human"); time.sleep(1)
    env.step(['what', 'is', 'jack', 'toy']);env.render("human"); time.sleep(1)

    env.step(['where', 'is', 'green', 'ball'])
    env.step(['where', 'is', 'blue', 'ball'])
    sys.exit()
    env.step(['what', 'is', 'jack', 'suitcase'])
    env.step(['what', 'is', 'mary', 'suitcase'])
    env.step(['where', 'is', 'livingroom'])
    env.step(['where', 'is', 'kitchen'])
    env.step(['where', 'is', 'restroom'])
    env.reset()
    env.step(['which', 'key', 'to', 'blue', 'box'])
    env.reset()
    env.step(['which', 'key', 'to', 'blue', 'box'])
    env.step(['what', 'is', 'adam', 'favorite', 'toy'])
    env.step(['what', 'is', 'jack', 'favorite', 'toy'])
    env.step(['what', 'is', 'mary', 'favorite', 'toy'])
    #env.step(['is', 'green', 'box', 'open'])
    env.step(['where', 'blue', 'key'])
    env.step(['where', 'blue', 'ball'])
    env.step(['where', 'green', 'ball'])
    env.step(['where', 'blue', 'ball'])

    env.step(['where', 'blue', 'box'])
    env.step(['where', 'green', 'ball'])
    env.step(['where', 'blue', 'ball'])
    env.step(['where', 'green', 'key'])
    env.step(['where', 'green', 'box'])
    env.step(['where', 'room0'])
    env.step(['where', 'room2'])
    env.step(['what', 'key'])
    env.step(['where', 'ball'])
    env.reset()
    env.step(['where', 'ball'])
    env.reset()
    env.step(['where', 'ball'])
    print(env.env)
    env.step(['left'])
    print(env.env)
    env.step(['forward'])
    print(env.env)
    env.step(['left'])
    print(env.env)
    env.step(['forward'])
    print(env.env)
    env.step(['toggle'])
    print(env.env)
    env.step(['forward'])
    print(env.env)
    env.step(['forward'])
    print(env.env)
    _, r, done, _ = env.step(['forward'])
    print(env.env)
    _, r, done, _ = env.step(['left'])
    print(env.env)
    _, r, done, _ = env.step(['forward'])
    print(env.env)
    _, r, done, _ = env.step(['forward'])
    print(env.env)
    env.step(['what', 'key'])
    env.step(['forward'])
    env.step(['forward'])
    env.step(['forward'])
    env.step(['forward'])
    env.step(['forward'])
    obs = env.step(['what', 'key'])
    obs = env.step(['where', 'blue', 'key'])
    env.step(['where', 'room6'])
    env.step(['where', 'grey', 'ball'])
    env.step(['where', 'ball'])
    env.step(['where', 'grey', 'door'])
    env.step(['where', 'room1'])
    env.step(['where', 'room5'])
    env.step(['where', 'room11'])
    env.step(['where', 'yellow', 'room11'])
    env.step(['wewe', 'yeqqq', 'asdf'])
