import gym
import re
import numpy as np
import time
other_places = ['kitchen', 'restroom', 'livingroom']
other_colors = ['red', 'yellow', 'white']
other_directions = ['east', 'west', 'second floor']
import random
class Query(gym.Wrapper):
    def __init__(self, env, n_q_type=2, n_color=6, n_object_type=12, verbose=False, restricted=False, flat=False, n_q=1, query_limit=100, reliability=1):
        super(Query, self).__init__(env)
        if restricted:
            if type(env.unwrapped).level_name == "GoToBall":
                n_q_type = 1
                n_color = 0
                n_object_type = 1
                self.res_actions = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done', 'where']
                self.res_color_actions = ['None']
                self.res_object_actions = ['ball', 'room0', 'room1']
            elif type(env.unwrapped).level_name == "MultiKeys":
                n_q_type = 1
                n_color = 0
                n_object_type = 1
                self.res_actions = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done', 'what']
                self.res_color_actions = ['None']
                self.res_object_actions = ['key', 'room0', 'room1']
            elif type(env.unwrapped).level_name == "GoToTwoBall":
                n_q_type = 1
                n_color = 2
                n_object_type = 1
                self.res_actions = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done', 'where']
                self.res_color_actions = ['None', 'blue', 'green']
                self.res_object_actions = ['ball']
        self.restricted = restricted
        self.flat = flat
        if self.flat:
            self.action_space = gym.spaces.Discrete((self.env.action_space.n) + n_q)
        else:
            self.action_space = gym.spaces.MultiDiscrete(
                [self.env.action_space.n, n_q])
        self.mini_grid_actions_map = {'left': 0, 'right': 1, 'forward': 2, 'pickup': 3, 'drop': 4, 'toggle': 5,
                                      'done': 6}
        self.verbose = verbose
        self.prev_query, self.prev_ans = 'None', 'None'
        self.bonus = [0.05, 0.05]
        self.counts = [1, 1]
        self.eps_steps = 0
        self.eps_n_q = 0
        self.action = 'None'
        self.init_query_limit = query_limit
        self.n_remaining_query = self.init_query_limit
        self.n_distractors = 2
        self.truth_th = reliability

    def reset(self, **kwargs):
        obs = super().reset()
        self.other_objs = [{'color': np.random.choice(other_colors),
                            'place': np.random.choice(other_places),
                            'direction': np.random.choice(other_directions)} for _ in range(self.n_distractors)]

        obs['ans'] = 'None'
        self.prev_query, self.prev_ans = 'None', 'None'
        self.bonus = [0.05, 0.05]
        self.eps_steps = 0
        self.eps_n_q = 0
        self.action = 'None'
        self.n_remaining_query = self.init_query_limit
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

            # obs['ans'] = 'None'
            obs['ans'] = self.prev_ans
            # debug
            #if rewrd > 0:
            #    #rewrd = 0 if 'ball' not in obs['ans'] else rewrd
            #    rewrd = 0 if 'None' in obs['ans'] else rewrd
            # self.prev_query, self.prev_ans = 'None', 'None'
            return obs, rewrd, done, info

        obs, rewrd, done, info = super().step(action=6)



        ans = 'I　dont　know'
        self.eps_n_q += 1
        prob = random.uniform(0, 1)

        if self.verbose:
            print('Q:', action)
        if self.n_remaining_query <= 0:
            ans = 'limit'
        elif action[0] == 'where' and prob >= self.truth_th:
            if action[1] == 'is' and action[2] in [obj['place'] for obj in self.other_objs]:
                for i, other_obj in enumerate(self.other_objs):
                    if action[2] == other_obj['place']:
                        idx = i
                ans = action[2] + ' in {}'.format(self.other_objs[idx]['direction'])
            elif len(action) == 4 and action[1] == 'is' and hasattr(self.env.unwrapped, 'names'):
                if action[2] in self.env.unwrapped.names and 'toy' in action[3] and hasattr(self.env.unwrapped.instrs, 'names'):
                    if action[2] == self.env.unwrapped.instrs.name:
                        color, type = self.env.unwrapped.instrs.desc.color, self.env.unwrapped.instrs.desc.type
                    else:
                        color, type = self.env.unwrapped.others_fav.color, self.env.unwrapped.others_fav.type
                    _, romm_id = self.get_ans(color=color, type=type)
                    ans = action[2] + ' ' + action[3] + ' in room{}'.format(romm_id)
                elif 'room' not in action[3]:
                    color, type = action[2], action[3]
                    ans, _ = self.get_ans(color=color, type=type)
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
                        ans, _ = self.get_ans_room(target_room_id)
                    #rewrd = rewrd + self.bonus[0] / np.sqrt(self.counts[0])
                    #self.bonus[0] = 0
                    #self.counts[0] += 1
                else:
                    ans, _ = self.get_ans(type=type)
                    #rewrd = rewrd + self.bonus[1] / np.sqrt(self.counts[1])
                    #self.bonus[1] = 0
                    #self.counts[1] += 1
                    #if type in obs['mission']:
                    #    rewrd += self.bonus[0]
                    #    self.bonus[0] = 0
        elif action[0] == 'what':
            if len(action) == 4 and action[3] == 'suitcase':
                if self.env.unwrapped.box_owner == action[2]:
                    if prob > self.truth_th:
                        ans = 'different from ' + 'tim' + ' ' + action[2] + ' suitcase ' + self.env.unwrapped.target_box.color + ' box'
                    else:
                        other_color = 'grey' if self.env.unwrapped.target_box.color == 'blue' else 'blue'
                        ans = 'different from ' + 'tim' + ' ' + action[2] + ' suitcase ' + other_color + ' box'
            elif action[1] == 'key' and hasattr(self.env, 'real_key') and self.is_locked_door(tuple(self.env.front_pos)):
                i, j, obj_id = self.env.real_key
                ans = self.env.room_grid[i][j].objs[obj_id].color + ' key'
            elif len(action) > 2:
                if action[2] in ['jack', 'mary'] and 'room' not in action[3]:
                    if action[2] in self.unwrapped.mission:
                        ans = action[2] + ' toy is ' + self.unwrapped.instrs.desc.color + ' ' + self.unwrapped.instrs.desc.type
                    elif hasattr(self.unwrapped, 'others_fav'):
                        ans = action[2] + ' toy is ' + self.unwrapped.others_fav.color + ' ' + self.unwrapped.others_fav.type
            #if self.restricted and action[1] == 'key':
            #    # for MultiKey only
            #    ans = self.real_key_color + ' key'
        elif action[0] == 'is':
            if action[2] == 'box':
                obj = self.get_obj(type='box')
                if obj:
                    ans = obj.color + ' box is not open'
            elif action[2] == 'ball':
                obj = self.get_obj(type='ball')
                if obj:
                    ans = obj.color + ' ball is not a basketball'
            elif action[2] == 'key':
                obj = self.get_obj(type='ball')
                if obj:
                    ans = obj.color + ' key is not around'
        elif action[0] == 'which' and action[1] == 'key' and hasattr(self.unwrapped, 'target_key_color'):
                    ans = self.unwrapped.target_key_color + ' key to ' + action[3] + ' ' + action[4]
        self.n_remaining_query -= 1
        obs['ans'] = ans
        if self.verbose:
            print('Ans:', ans)
        self.prev_query = ' '.join(action)
        self.prev_ans = ans

        return obs, rewrd, done, info

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
        return ans, room_id

    def get_obj(self, type):
        for room_row in self.env.room_grid:
            for room in room_row:
                for obj in room.objs:
                    if obj.type == type:
                        return obj
        return None


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
            caption = 'Instr: {} step: {} action: {} # of queries: {}\n'.format(self.mission, self.eps_steps, self.action, self.eps_n_q)
            in_bos = 'Q: ' + self.prev_query + ' A: ' + self.prev_ans + "\n"
            if KG:
                in_bos += repr(KG)
            self.window.set_caption(caption + in_bos)
            #if self.prev_query != '':
            #    in_bos += self.prev_query + '\n' + self.prev_ans
            #    self.window.set_caption(caption + )
            #else:
            #    in_bos += self.mission
            #    self.window.set_caption(caption + )
            self.window.show_img(img)
            # if self.prev_query != '':
            #    time.sleep(0.5)
        return img


if __name__ == "__main__":
    import babyai

    # env_name = 'BabyAI-Unlock-v0'
    # env_name = 'BabyAI-GoTo-v0'
    #env_name = 'BabyAI-GoToBall-v0'
    # env_name = 'BabyAI-GoToBallGTAns-v0'
    # env_name = 'BabyAI-MultiKeys-v0'
    # env_name = 'BabyAI-MultiKeysGTAns-v0'
    # env_name = 'BabyAI-GoToTwoBall-v0'
    #env_name = 'BabyAI-General-v0'
    #env_name = 'BabyAI-GoToObjMaze2-v0'
    #env_name = 'BabyAI-GoToFavorite-v0'
    env_name = 'BabyAI-ObjInBoxMultiSufficient-v0'
    #env_name = 'BabyAI-ObjInLockedBox-v0'
    env = gym.make(env_name)
    env = Query(env, verbose=True, query_limit=20)
    obs = env.seed(2)
    env.reset()
    env.step(['what', 'is', 'mary', 'toy'])
    env.step(['what', 'is', 'jack', 'toy'])
    env.step(['where', 'is', 'green', 'ball'])
    env.step(['where', 'is', 'blue', 'ball'])
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
