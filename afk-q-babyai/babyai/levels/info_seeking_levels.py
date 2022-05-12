import gym
from gym_minigrid.envs import Key, Ball, Box
from .levelgen import *
from .objects import DoorWID, KeyWID, BoxWID
import random
import copy
from gym_minigrid.minigrid import fill_coords, point_in_rect, point_in_line, WorldObj, COLORS
from .utils import *

import itertools as itt



class Goal(WorldObj):
    def __init__(self, color):
        super().__init__('goal', color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Oracle(Ball):
    def __init__(self, color):
        super().__init__(color)

    def can_overlap(self):
        return True


class Level_GoToFavorite3Room(RoomGridLevel):
    def __init__(self, seed=None, room_size=7, call=True, failure_neg=False):
        self.lava_colors = ['yellow', 'blue']
        self.n_target = 2
        self.goal_pos = (room_size * 2 - 3, room_size - 2)
        self.agent_start_pos = (2, 1)
        self.call = call
        self.names = ['jack', 'mary']
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )

    def gen_mission(self):

        self.target_idx = self.np_random.randint(0, self.n_target)

        self.room_grid[0][0].door_pos[0] = (6, 1)
        self.room_grid[0][1].door_pos[2] = (6, 1)
        self.room_grid[0][1].door_pos[0] = (12, 5)
        self.room_grid[0][2].door_pos[2] = (12, 5)
        self.add_door(0, 0, door_idx=0, locked=False)
        self.add_door(1, 0, door_idx=0, locked=False)
        # Place obstacles (lava or walls)
        #self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
        #add_danger_tiles_anti_diag(self)
        #self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
        #add_danger_tiles_diag(self)
        #self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
        #add_danger_tiles_anti_diag(self, offset=12)

        if self.call:
            self.grid.set(2, 1, None)
            self.agent_pos = (self.room_size, 1)
        else:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.agent_pos = (1, 1)
            self.oracle_pos = (3, 1)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))
        self.agent_dir = 1
        self.open_all_doors()
        # GoToFav
        self.num_dists = 2
        objs = []
        obj = create_rnd_obj(self)
        place_obj(self, 1, self.room_size - 2, obj)
        objs.append(obj)
        obj = create_rnd_obj(self)
        place_obj(self, (self.room_size - 1) * 3 - 1, 1, obj)
        objs.append(obj)
        target_obj = self._rand_elem(objs)

        self.instrs = FavoriteInstr(ObjDesc(target_obj.type, target_obj.color), name=self.names[0], surface="go to {} toy".format(self.names[0]), danger=False)
        room_id = (target_obj.cur_pos[0] // self.room_size) + (target_obj.cur_pos[1] // self.room_size) * self.num_cols
        self.useful_answers = ['{} toy is {} {}'.format(self.names[0], target_obj.color, target_obj.type),
                               '{} {} in room{}'.format(target_obj.color, target_obj.type, room_id)]




class Level_DangerBalancedDiverse(RoomGridLevel):
    def __init__(self, seed=None, room_size=7, call=True, failure_neg=False):
        self.lava_colors = ['yellow', 'blue']
        self.n_target = 2
        self.goal_pos = (room_size - 2, room_size - 2)
        self.agent_start_pos = (2, 1)
        self.call = call
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )

    def gen_mission(self):
        self.target_idx = self.np_random.randint(0, self.n_target)
        height, width = self.room_size, self.room_size
        # Place obstacles (lava or walls)
        self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        colored_tiles = [set(), set()]
        for i, j in obstacle_pos:
            color_idx = self.np_random.randint(0, 2)
            self.grid.set(i, j, Lava(self.lava_colors[color_idx]))
            colored_tiles[color_idx].add((i, j))

        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        openings = set()
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False

            self.grid.set(i, j, Lava(self.lava_colors[self.target_idx]))
            openings.add((i, j))
            if (i, j) in colored_tiles[1 - self.target_idx]:
                a, b = None, None
                while colored_tiles[self.target_idx]:
                    a, b = colored_tiles[self.target_idx].pop()
                    if (a, b) not in openings:
                        break
                    if not colored_tiles[self.target_idx]:
                        a, b = None, None
                        break
                if a is not None:
                    self.grid.set(a, b, Lava(self.lava_colors[1 - self.target_idx]))



            #self.grid.set(i, j, None)
        self.instrs = GoToGoalInstr()
        self.put_obj(Goal('green'), *self.goal_pos)
        if self.call:
            self.grid.set(2, 1, None)
            self.agent_pos = (2, 1)
        else:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.agent_pos = (1, 1)
            self.oracle_pos = (3, 1)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))
        self.agent_dir = 1


        self.useful_answers = ['danger zone is {}'.format(self.lava_colors[1 - self.target_idx])]


class Level_DangerBalancedDiverseMove(Level_DangerBalancedDiverse):
    def __init__(self, seed=None, room_size=7):
        super().__init__(seed=seed, room_size=room_size, call=False)


class Level_DangerBalancedDiverseRS9(Level_DangerBalancedDiverse):
    def __init__(self, seed=None, room_size=9):
        super().__init__(seed=seed, room_size=room_size)




class Level_OpenDoorMultiKeys(RoomGridLevel):
    def __init__(self, seed=None):
        room_size = 7
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            seed=seed
        )
        self.real_key_color = None

    def gen_mission(self):
        #colors = self._rand_subset(COLOR_NAMES, 2)
        colors = COLOR_NAMES[0:3]
        n_keys = 3
        # Add a door of color A connecting left and middle room
        locked_door_id = self._rand_int(0, n_keys)
        self.locked_door, _ = add_id_door(self, 0, 0, id=locked_door_id, door_idx=0, color=colors[0], locked=True)

        for i in range(n_keys):
            obj, _ = add_id_object(self, 0, 0, id=i, kind="key", color=colors[i])
            if i == locked_door_id:
                self.real_key_color = colors[i]
        self.place_agent(0, 0)
        self.instrs = OpenInstr(ObjDesc(self.locked_door.type, color=self.locked_door.color))
        self.useful_answers = ['{} key to {} door'.format(self.real_key_color, self.locked_door.color)]




class Level_DangerBalancedDiverseNeg(Level_DangerBalancedDiverse):
    def __init__(self, seed=None):
        super().__init__(seed=seed, failure_neg=True)


#### Two Compositional


class Level_DangerOpenDoor(RoomGridLevel):
    def __init__(self, seed=None, room_size=7, call=True, failure_neg=False, anti_danger=True, n_rivers=3, open_door_bonus=True):
        self.lava_colors = ['yellow', 'blue']
        self.n_target = 2
        self.goal_pos = (room_size * 2 - 3, room_size - 2)
        self.agent_start_pos = (2, 1)
        self.call = call
        self.failure_neg = failure_neg
        self.anti_danger = anti_danger
        self.max_n_rivers = n_rivers
        self.open_door_bonus = open_door_bonus
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )


    def gen_mission(self):
        self.target_idx = self.np_random.randint(0, self.n_target)
        self.room_grid[0][0].door_pos[0] = (6, 1)
        self.room_grid[0][1].door_pos[2] = (6, 1)

        # Place obstacles (lava or walls)
        if self.anti_danger:
            self.num_crossings = self.np_random.randint(1, self.max_n_rivers + 1)
            add_danger_tiles_anti_diag(self)
        self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
        add_danger_tiles_diag(self)

        self.instrs = GoToGoalInstr(door=self.open_door_bonus)
        #self.put_obj(Goal('green'), *self.goal_pos)
        if self.call:
            self.grid.set(2, 1, None)
            self.agent_pos = (1, self.room_size - 2)
        else:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.agent_pos = (1, 1)
            self.oracle_pos = (3, 1)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))
        self.agent_dir = 1

        # Open Door Part
        colors = COLOR_NAMES[0:3]
        n_keys = 2
        # Add a door of color A connecting left and middle room
        locked_door_id = self._rand_int(0, n_keys)
        self.locked_door, _ = add_id_door(self, 0, 0, id=locked_door_id, door_idx=0, color=colors[0], locked=True)
        self.target_doors = [self.locked_door]
        for i in range(n_keys):
            obj = KeyWID(i, colors[i])
            if i == 0:
                self.grid.set(self.room_size - 3, 1, obj)
            elif i == 1:
                self.grid.set(self.room_size - 2, 2, obj)
            if i == locked_door_id:
                self.real_key_color = colors[i]

        self.put_obj(Goal('green'), *self.goal_pos)
        self.useful_answers = ['danger zone is {}, {} key to {} door'.format(self.lava_colors[1 - self.target_idx], self.real_key_color, self.locked_door.color)]



class Level_DangerOpenDoorV2(Level_DangerOpenDoor):
    def __init__(self, seed=None, room_size=7):
        super().__init__(seed=seed, room_size=room_size, open_door_bonus=False)


class Level_OpenDoorObjInBox(RoomGridLevel):
    def __init__(self, seed=None, n_boxes=2, failure_neg=False):
        room_size = 7
        self.n_boxes = n_boxes
        self.names = ['jack', 'mary']
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )
        self.real_key_color = None

    def gen_mission(self):

        colors = COLOR_NAMES[0:3]
        n_keys = 2
        # Add a door of color A connecting left and middle room
        locked_door_id = self._rand_int(0, n_keys)
        self.locked_door, _ = add_id_door(self, 0, 0, id=locked_door_id, door_idx=0, color=colors[0], locked=True)
        self.target_doors = [self.locked_door]
        for i in range(n_keys):
            obj, _ = add_id_object(self, 0, 0, id=i, kind="key", color=colors[i])
            if i == locked_door_id:
                self.real_key_color = colors[i]



        # ObjInBox Part
        colors = COLOR_NAMES[:self.n_boxes]
        shuffled_colors = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors)
        shuffled_colors2 = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors2)

        target_id = self._rand_int(0, self.n_boxes)
        toy_owner = self._rand_int(0, 2)
        box_owner = 1 - toy_owner

        for i in range(self.n_boxes):
            ball = Ball(color=shuffled_colors[i])
            box = Box(color=shuffled_colors2[i], contains=ball)
            self.place_in_room(1, 0, box)
            if i == target_id:
                self.box_owner = self.names[box_owner]
                self.target_box = box

                self.instrs = FindOrFailInstr(ObjDesc(ball.type, ball.color), surface='find the key to the door, and find ' + self.names[toy_owner] + ' toy', door=True)

                self.useful_answers = ['{} toy is {} ball'.format(self.names[toy_owner], ball.color),
                                       '{} ball in {} suitcase'.format(ball.color, self.names[box_owner]),
                                       '{} suitcase {} box'.format(self.names[box_owner], self.target_box.color),
                                       '{} box in room1'.format(self.target_box.color)]

        self.place_agent(0, 0)
        self.useful_answers.append('{} key to the door'.format(self.real_key_color))







class Level_OpenDoorGoToFavorite(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
            self,
            room_size=5,
            num_rows=3,
            num_cols=3,
            num_dists=3,
            doors_open=False,
            seed=None,
            all_doors=True,
            num_colors=2,
            oracle_mode='call'
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.all_doors = all_doors
        self.num_colors = num_colors
        self.names = ['jack', 'mary']
        self.others_fav = None
        self.oracle_mode = oracle_mode
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed
        )


    def gen_mission(self):
        agent_room_i = self._rand_int(0, self.num_cols)
        agent_room_j = self._rand_int(0, self.num_rows)
        self.place_agent(agent_room_i, agent_room_j)
        locked = False
        if self.all_doors:
            add_id_door(self, 0, 0, id=100, door_idx=0, locked=locked)
            add_id_door(self, 0, 1, id=100, door_idx=0, locked=locked)
            add_id_door(self, 1, 1, id=100, door_idx=0, locked=locked)
            add_id_door(self, 0, 2, id=100, door_idx=0, locked=locked)
            add_id_door(self, 1, 2, id=100, door_idx=0, locked=locked)

            add_id_door(self, 0, 0, id=100, door_idx=1, locked=locked)
            add_id_door(self, 1, 0, id=100, door_idx=1, locked=locked)
            add_id_door(self, 2, 0, id=100, door_idx=1, locked=locked)

            add_id_door(self, 0, 1, id=100, door_idx=1, locked=locked)
            add_id_door(self, 1, 1, id=100, door_idx=1, locked=locked)
            add_id_door(self, 2, 1, id=100, door_idx=1, locked=locked)
        else:
            self.connect_all()

        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=True, num_colors=self.num_colors)
        obj = self._rand_elem(objs)
        self.check_objs_reachable()
        random.shuffle(self.names)
        self.instrs = FavoriteInstr(ObjDesc(obj.type, obj.color), name=self.names[0], surface='find the key to the door, go to {} toy'.format(self.names[0]), open_door=True)
        room_id = (obj.cur_pos[0] // self.room_size) + (obj.cur_pos[1] // self.room_size) * self.num_cols
        target_obj = obj
        self.others_fav = self._rand_elem(objs)
        if self.doors_open:
            self.open_all_doors()
        if self.oracle_mode == 'single_move':
            oracle = Oracle(color='red')
            self.place_in_room(agent_room_i, agent_room_j, oracle)
            self.oracle = oracle


        # OpenDoor
        colors = COLOR_NAMES[0:3]
        n_keys = 2
        # Add a door of color A connecting left and middle room
        locked_door_id = self._rand_int(0, n_keys)
        #self.locked_door, _ = self.add_id_door(0, 0, id=locked_door_id, door_idx=0, color=colors[0], locked=True)

        for i in range(n_keys):
            obj, _ = add_id_object(self, agent_room_i, agent_room_j, id=i, kind="key", color=colors[i])
            if i == locked_door_id:
                self.real_key_color = colors[i]
        self.target_doors = []
        room_i, room_j = room_id // 3, room_id % 3
        for door in self.room_grid[room_i][room_j].doors:
            if door is not None:
                door.is_locked = True
                door.id = locked_door_id
                self.target_doors.append(door)

        self.useful_answers = ['{} toy is {} {}'.format(self.names[0], target_obj.color, target_obj.type),
                               '{} {} in room{}'.format(target_obj.color,  target_obj.type, room_id),
                               '{} key to the door'.format(self.real_key_color)
                               ]


    def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True, num_colors=2):
        """
        Add random objects that can potentially distract/confuse the agent.
        """

        # Collect a list of existing objects
        objs = []
        for row in self.room_grid:
            for room in row:
                for obj in room.objs:
                    objs.append((obj.type, obj.color))

        # List of distractors added
        dists = []

        while len(dists) < num_distractors:
            color = self._rand_elem(COLOR_NAMES[:num_colors])
            type = self._rand_elem(['ball', 'box'])
            obj = (type, color)

            if all_unique and obj in objs:
                continue

            # Add the object to a random room if no room specified
            room_i = i
            room_j = j
            if room_i == None:
                room_i = self._rand_int(0, self.num_cols)
            if room_j == None:
                room_j = self._rand_int(0, self.num_rows)

            dist, pos = self.add_object(room_i, room_j, *obj)

            objs.append(obj)
            dists.append(dist)

        return dists





class Level_GoToFavoriteObjInBox(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
            self,
            room_size=5,
            num_rows=3,
            num_cols=3,
            num_dists=3,
            doors_open=True,
            seed=None,
            all_doors=True,
            num_colors=2,
            oracle_mode='call',
            failure_neg=False
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.all_doors = all_doors
        self.num_colors = num_colors
        self.names = ['jack', 'mary']
        self.others_fav = None
        self.oracle_mode = oracle_mode
        self.n_boxes = 2
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )


    def gen_mission(self):
        agent_room_i = self._rand_int(0, self.num_cols)
        agent_room_j = self._rand_int(0, self.num_rows)
        self.place_agent(agent_room_i, agent_room_j)
        locked = False
        if self.all_doors:
            self.add_door(0, 0, door_idx=0, locked=locked)
            self.add_door(1, 0, door_idx=0, locked=locked)
            self.add_door(0, 1, door_idx=0, locked=locked)
            self.add_door(1, 1, door_idx=0, locked=locked)
            self.add_door(0, 2, door_idx=0, locked=locked)
            self.add_door(1, 2, door_idx=0, locked=locked)

            self.add_door(0, 0, door_idx=1, locked=locked)
            self.add_door(1, 0, door_idx=1, locked=locked)
            self.add_door(2, 0, door_idx=1, locked=locked)

            self.add_door(0, 1, door_idx=1, locked=locked)
            self.add_door(1, 1, door_idx=1, locked=locked)
            self.add_door(2, 1, door_idx=1, locked=locked)
        else:
            self.connect_all()


        if self.doors_open:
            self.open_all_doors()
        if self.oracle_mode == 'single_move':
            oracle = Oracle(color='red')
            self.place_in_room(agent_room_i, agent_room_j, oracle)
            self.oracle = oracle


        # ObjInBox
        colors = COLOR_NAMES[:self.n_boxes]
        shuffled_colors = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors)
        shuffled_colors2 = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors2)

        target_id = self._rand_int(0, self.n_boxes)
        toy_owner = self._rand_int(0, 2)
        box_owner = 1 - toy_owner

        for i in range(self.n_boxes):
            ball = Ball(color=shuffled_colors[i])
            box = Box(color=shuffled_colors2[i], contains=ball)
            self.place_in_room(self.np_random.randint(0, 3), self.np_random.randint(0, 3), box)
            if i == target_id:
                self.box_owner = self.names[box_owner]
                self.target_box = box
                self.instrs = FindOrFailInstr(ObjDesc(ball.type, ball.color), surface='find ' + self.names[toy_owner] + ' toy')

                self.useful_answers = ['{} toy is {} ball'.format(self.names[toy_owner], ball.color),
                                       '{} ball in {} suitcase'.format(ball.color, self.names[box_owner]),
                                       '{} suitcase is {} box'.format(self.names[box_owner], self.target_box.color)]



    def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True, num_colors=2):
        """
        Add random objects that can potentially distract/confuse the agent.
        """

        # Collect a list of existing objects
        objs = []
        for row in self.room_grid:
            for room in row:
                for obj in room.objs:
                    objs.append((obj.type, obj.color))

        # List of distractors added
        dists = []

        while len(dists) < num_distractors:
            color = self._rand_elem(COLOR_NAMES[:num_colors])
            type = self._rand_elem(['ball', 'box'])
            obj = (type, color)

            if all_unique and obj in objs:
                continue

            # Add the object to a random room if no room specified
            room_i = i
            room_j = j
            if room_i == None:
                room_i = self._rand_int(0, self.num_cols)
            if room_j == None:
                room_j = self._rand_int(0, self.num_rows)

            dist, pos = self.add_object(room_i, room_j, *obj)

            objs.append(obj)
            dists.append(dist)

        return dists




class Level_DangerObjInBox(RoomGridLevel):
    def __init__(self, seed=None, room_size=7, call=True, failure_neg=False):
        self.lava_colors = ['yellow', 'blue']
        self.n_target = 2
        self.goal_pos = (room_size - 2, room_size - 2)
        self.agent_start_pos = (2, 1)
        self.call = call
        self.n_boxes = 2
        self.names = ['jack', 'mary']
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )

    def gen_mission(self):
        self.target_idx = self.np_random.randint(0, self.n_target)

        # Place obstacles (lava or walls)
        self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
        add_danger_tiles_anti_diag(self)

        if self.call:
            self.grid.set(2, 1, None)
            self.agent_pos = (2, 1)
        else:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.agent_pos = (1, 1)
            self.oracle_pos = (3, 1)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))
        self.agent_dir = 1

        colors = COLOR_NAMES[:self.n_boxes]
        shuffled_colors = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors)
        shuffled_colors2 = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors2)

        target_id = self._rand_int(0, self.n_boxes)
        toy_owner = self._rand_int(0, 2)
        box_owner = 1 - toy_owner
        for i in range(self.n_boxes):
            ball = Ball(color=shuffled_colors[i])
            box = Box(color=shuffled_colors2[i], contains=ball)
            if i == 0:
                self.grid.set(1, self.room_size - 2, box)
            else:
                self.grid.set(self.room_size - 2, self.room_size - 2, box)
            self.get_room(0, 0).objs.append(box)
            if i == target_id:
                self.box_owner = self.names[box_owner]
                self.target_box = box
                self.instrs = FindOrFailInstr(ObjDesc(ball.type, ball.color), surface='Avoid danger zone, find ' + self.names[toy_owner] + ' toy',
                                              danger=True)

                self.useful_answers = ['{} toy is {} ball'.format(self.names[toy_owner], ball.color),
                                       '{} ball in {} suitcase'.format(ball.color, self.names[box_owner]),
                                       '{} suitcase is {} box'.format(self.names[box_owner], self.target_box.color),
                                       'danger zone is {}'.format(self.lava_colors[1 - self.target_idx])]
                #print(self.useful_answers)



class Level_DangerGoToFav(RoomGridLevel):
    def __init__(self, seed=None, room_size=7, call=True, failure_neg=False, anti_danger=True, n_rivers=3):
        self.lava_colors = ['yellow', 'blue']
        self.n_target = 2
        self.goal_pos = (room_size * 2 - 3, room_size - 2)
        self.agent_start_pos = (2, 1)
        self.call = call
        self.names = ['jack', 'mary']
        self.anti_danger = anti_danger
        self.max_n_rivers = n_rivers
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )

    def gen_mission(self):

        self.target_idx = self.np_random.randint(0, self.n_target)

        self.room_grid[0][0].door_pos[0] = (6, 1)
        self.room_grid[0][1].door_pos[2] = (6, 1)
        self.room_grid[0][1].door_pos[0] = (12, 5)
        self.room_grid[0][2].door_pos[2] = (12, 5)
        self.add_door(0, 0, door_idx=0, locked=False)
        self.add_door(1, 0, door_idx=0, locked=False)
        # Place obstacles (lava or walls)
        if self.anti_danger:
            self.num_crossings = self.np_random.randint(1, self.max_n_rivers + 1)
            add_danger_tiles_anti_diag(self)
            self.num_crossings = self.np_random.randint(1, self.max_n_rivers + 1)
            add_danger_tiles_anti_diag(self, offset=12)
        self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
        add_danger_tiles_diag(self)


        if self.call:
            self.grid.set(2, 1, None)
            self.agent_pos = (self.room_size, 1)
        else:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.agent_pos = (1, 1)
            self.oracle_pos = (3, 1)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))
        self.agent_dir = 1
        self.open_all_doors()
        # GoToFav
        self.num_dists = 2
        objs = []
        obj = create_rnd_obj(self)
        place_obj(self, 1, self.room_size - 2, obj)
        objs.append(obj)
        obj = create_rnd_obj(self)
        place_obj(self, (self.room_size - 1) * 3 - 1, 1, obj)
        objs.append(obj)
        target_obj = self._rand_elem(objs)

        self.instrs = FavoriteInstr(ObjDesc(target_obj.type, target_obj.color), name=self.names[0], surface="avoid danger zone, go to {} toy".format(self.names[0]), danger=True)
        room_id = (target_obj.cur_pos[0] // self.room_size) + (target_obj.cur_pos[1] // self.room_size) * self.num_cols
        self.useful_answers = ['{} toy is {} {}'.format(self.names[0], target_obj.color, target_obj.type),
                               '{} {} in room{}'.format(target_obj.color, target_obj.type, room_id),
                               'danger zone is {}'.format(self.lava_colors[1 - self.target_idx])]
        #print(self.useful_answers)


class Level_DangerGoToFavSymmetry(RoomGridLevel):
    def __init__(self, seed=None, room_size=7, call=True, failure_neg=False, n_rivers=3):
        self.lava_colors = ['yellow', 'blue']
        self.n_target = 2
        self.goal_pos = (room_size * 2 - 3, room_size - 2)
        self.agent_start_pos = (2, 1)
        self.call = call
        self.names = ['jack', 'mary']
        self.max_n_rivers = n_rivers
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )

    def gen_mission(self):

        self.target_idx = self.np_random.randint(0, self.n_target)

        self.room_grid[0][0].door_pos[0] = (6, 1)
        self.room_grid[0][1].door_pos[2] = (6, 1)
        self.room_grid[0][1].door_pos[0] = (12, 1)
        self.room_grid[0][2].door_pos[2] = (12, 1)
        self.add_door(0, 0, door_idx=0, locked=False)
        self.add_door(1, 0, door_idx=0, locked=False)
        # Place obstacles (lava or walls)
        self.num_crossings = self.np_random.randint(1, self.max_n_rivers + 1)
        add_danger_tiles_anti_diag(self)


        self.num_crossings = self.np_random.randint(1, 4)
        add_danger_tiles_middle(self, offset=6)

        self.num_crossings = self.np_random.randint(1, (self.max_n_rivers + 1))
        add_danger_tiles_diag(self, offset=12)

        if self.call:
            self.grid.set(2, 1, None)
            self.agent_pos = (self.room_size + 2, 1)
        else:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.agent_pos = (1, 1)
            self.oracle_pos = (3, 1)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))
        self.agent_dir = 1
        self.open_all_doors()
        # GoToFav
        self.num_dists = 2
        objs = []
        obj = create_rnd_obj(self)
        place_obj(self, 1, self.room_size - 2, obj)
        objs.append(obj)
        obj = create_rnd_obj(self)
        place_obj(self, (self.room_size - 1) * 3 - 1, self.room_size - 2, obj)
        objs.append(obj)
        target_obj = self._rand_elem(objs)

        self.instrs = FavoriteInstr(ObjDesc(target_obj.type, target_obj.color), name=self.names[0], surface="avoid danger zone, go to {} toy".format(self.names[0]), danger=True)
        room_id = (target_obj.cur_pos[0] // self.room_size) + (target_obj.cur_pos[1] // self.room_size) * self.num_cols
        self.useful_answers = ['{} toy is {} {}'.format(self.names[0], target_obj.color, target_obj.type),
                               '{} {} in room{}'.format(target_obj.color, target_obj.type, room_id)]


class Level_DangerGoToFav2Room(RoomGridLevel):
    def __init__(self, seed=None, room_size=7, call=True, failure_neg=False, anti_danger=True, n_rivers=3):
        self.lava_colors = ['yellow', 'blue']
        self.n_target = 2
        self.goal_pos = (room_size * 2 - 3, room_size - 2)
        self.agent_start_pos = (2, 1)
        self.call = call
        self.names = ['jack', 'mary']
        self.anti_danger = anti_danger
        self.max_n_rivers = n_rivers
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )

    def gen_mission(self):

        self.target_idx = self.np_random.randint(0, self.n_target)

        self.room_grid[0][0].door_pos[0] = (6, 1)
        self.room_grid[0][1].door_pos[2] = (6, 1)

        self.add_door(0, 0, door_idx=0, locked=False)
        # Place obstacles (lava or walls)

        if self.anti_danger:
            self.num_crossings = self.np_random.randint(1, self.max_n_rivers + 1)
            add_danger_tiles_anti_diag(self)
        self.num_crossings = self.np_random.randint(1, (self.max_n_rivers + 1))
        add_danger_tiles_diag(self)


        if self.call:
            self.grid.set(2, 1, None)
            self.agent_pos = (self.room_size - 1, 1)
        else:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.agent_pos = (1, 1)
            self.oracle_pos = (3, 1)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))
        self.agent_dir = 1
        self.open_all_doors()
        # GoToFav
        self.num_dists = 2
        objs = []
        obj = create_rnd_obj(self)
        place_obj(self, 1, self.room_size - 2, obj)
        objs.append(obj)
        obj = create_rnd_obj(self)
        place_obj(self, (self.room_size - 1) * 2 - 1, self.room_size - 2, obj)
        objs.append(obj)
        target_obj = self._rand_elem(objs)

        self.instrs = FavoriteInstr(ObjDesc(target_obj.type, target_obj.color), name=self.names[0], surface="avoid danger zone, go to {} toy".format(self.names[0]), danger=True)
        room_id = (target_obj.cur_pos[0] // self.room_size) + (target_obj.cur_pos[1] // self.room_size) * self.num_cols
        self.useful_answers = ['{} toy is {} {}'.format(self.names[0], target_obj.color, target_obj.type),
                               '{} {} in room{}'.format(target_obj.color, target_obj.type, room_id),
                               'danger zone is {}'.format(self.lava_colors[1 - self.target_idx])
                               ]





class Level_DangerOpenDoorNeg(Level_DangerOpenDoor):
    def __init__(self, seed=None, room_size=7):
        super().__init__(seed=seed, room_size=room_size, call=True, failure_neg=True)


class Level_DangerGoToFavNeg(Level_DangerGoToFav):
    def __init__(self, seed=None, room_size=7):
        super().__init__(seed=seed, room_size=room_size, call=True, failure_neg=True)

class Level_DangerObjInBoxNeg(Level_DangerObjInBox):
    def __init__(self, seed=None, room_size=7):
        super().__init__(seed=seed, room_size=room_size, call=True, failure_neg=True)


class Level_GoToFavoriteObjInBoxNeg(Level_GoToFavoriteObjInBox):
    def __init__(self, seed=None):
        super().__init__(seed=seed, failure_neg=True)

class Level_OpenDoorObjInBoxNeg(Level_OpenDoorObjInBox):
    def __init__(self, seed=None):
        super().__init__(seed=seed, failure_neg=True)

class Level_DangerGoToFavEasy(Level_DangerGoToFav):
    def __init__(self, seed=None, room_size=7):
        super().__init__(seed=seed, room_size=room_size, call=True, failure_neg=False, anti_danger=False)

class Level_DangerOpenDoorEasy(Level_DangerOpenDoor):
    def __init__(self, seed=None, room_size=7):
        super().__init__(seed=seed, room_size=room_size, call=True, failure_neg=False, anti_danger=False)


class Level_DangerOpenDoor1Cross(Level_DangerOpenDoor):
    def __init__(self, seed=None, room_size=7):
        super().__init__(seed=seed, room_size=room_size, call=True, failure_neg=False, anti_danger=True, n_rivers=1)

class Level_DangerOpenDoor2Cross(Level_DangerOpenDoor):
    def __init__(self, seed=None, room_size=7):
        super().__init__(seed=seed, room_size=room_size, call=True, failure_neg=False, anti_danger=True, n_rivers=2)


class Level_DangerGoToFav1Cross(Level_DangerGoToFav):
    def __init__(self, seed=None, room_size=7):
        super().__init__(seed=seed, room_size=room_size, call=True, failure_neg=False, anti_danger=True, n_rivers=1)

class Level_DangerGoToFav2Cross(Level_DangerGoToFav):
    def __init__(self, seed=None, room_size=7):
        super().__init__(seed=seed, room_size=room_size, call=True, failure_neg=False, anti_danger=True, n_rivers=2)





# Combine 3 tasks

class Level_DangerGoToFavOpenDoor(RoomGridLevel):
    def __init__(self, seed=None, room_size=7, call=True):
        self.lava_colors = ['yellow', 'blue']
        self.n_target = 2
        self.goal_pos = (room_size * 2 - 3, room_size - 2)
        self.agent_start_pos = (2, 1)
        self.call = call
        self.names = ['jack', 'mary']
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):

        self.target_idx = self.np_random.randint(0, self.n_target)

        self.room_grid[0][0].door_pos[0] = (6, 1)
        self.room_grid[0][1].door_pos[2] = (6, 1)
        self.room_grid[0][1].door_pos[0] = (12, 5)
        self.room_grid[0][2].door_pos[2] = (12, 5)
        #self.add_door(0, 0, door_idx=0, locked=False)
        #self.add_door(1, 0, door_idx=0, locked=False)
        # Place obstacles (lava or walls)
        self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
        add_danger_tiles_anti_diag(self)
        self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
        add_danger_tiles_diag(self)
        self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
        add_danger_tiles_anti_diag(self, offset=12)

        if self.call:
            self.grid.set(2, 1, None)
            self.agent_pos = (self.room_size, 1)
        else:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.agent_pos = (1, 1)
            self.oracle_pos = (3, 1)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))
        self.agent_dir = 1


        # GoToFav
        self.num_dists = 2
        objs = []
        obj = create_rnd_obj(self)
        place_obj(self, 1, self.room_size - 2, obj)
        objs.append(obj)
        obj = create_rnd_obj(self)
        place_obj(self, (self.room_size - 1) * 3 - 1, 1, obj)
        objs.append(obj)
        target_obj = self._rand_elem(objs)


        # OpenDoor
        colors = COLOR_NAMES[0:3]
        self.np_random.shuffle(colors)
        n_keys = 2
        # Add a door of color A connecting left and middle room
        locked_door_id = self._rand_int(0, n_keys)
        self.locked_doors = []
        self.locked_doors.append(add_id_door(self, 0, 0, id=0, door_idx=0, color=colors[0], locked=True)[0])
        self.locked_doors.append(add_id_door(self, 1, 0, id=0, door_idx=0, color=colors[1], locked=True)[0])
        agent_room_i, agent_room_j = 1, 0
        self.np_random.shuffle(colors)
        add_id_object_abs(self, self.room_size - 1 + 1, self.room_size - 2, id=0, kind='key', color=colors[0])
        add_id_object_abs(self, 2 * (self.room_size - 1) - 1, self.room_size - 2, id=1, kind='key', color=colors[1])
        self.real_key_color = colors[0]


        self.instrs = FavoriteInstr(ObjDesc(target_obj.type, target_obj.color), name=self.names[0], surface="avoid danger zone, find the key to open the door, go to {} toy".format(self.names[0]), danger=True)
        room_id = (target_obj.cur_pos[0] // self.room_size) + (target_obj.cur_pos[1] // self.room_size) * self.num_cols
        self.useful_answers = ['{} toy is {} {}'.format(self.names[0], target_obj.color, target_obj.type),
                               '{} {} in room{}'.format(target_obj.color, target_obj.type, room_id),
                               'danger zone is {}'.format(self.lava_colors[1 - self.target_idx])
                               ]





class Level_OpenDoorGoToFavoriteObjInBox(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
            self,
            room_size=5,
            num_rows=3,
            num_cols=3,
            num_dists=3,
            doors_open=False,
            seed=None,
            all_doors=True,
            num_colors=2,
            oracle_mode='call'
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.all_doors = all_doors
        self.num_colors = num_colors
        self.names = ['jack', 'mary']
        self.others_fav = None
        self.oracle_mode = oracle_mode
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed
        )


    def gen_mission(self):
        agent_room_i = self._rand_int(0, self.num_cols)
        agent_room_j = self._rand_int(0, self.num_rows)
        self.place_agent(agent_room_i, agent_room_j)
        locked = False
        if self.all_doors:
            add_id_door(self, 0, 0, id=100, door_idx=0, locked=locked)
            add_id_door(self, 0, 1, id=100, door_idx=0, locked=locked)
            add_id_door(self, 1, 1, id=100, door_idx=0, locked=locked)
            add_id_door(self, 0, 2, id=100, door_idx=0, locked=locked)
            add_id_door(self, 1, 2, id=100, door_idx=0, locked=locked)

            add_id_door(self, 0, 0, id=100, door_idx=1, locked=locked)
            add_id_door(self, 1, 0, id=100, door_idx=1, locked=locked)
            add_id_door(self, 2, 0, id=100, door_idx=1, locked=locked)

            add_id_door(self, 0, 1, id=100, door_idx=1, locked=locked)
            add_id_door(self, 1, 1, id=100, door_idx=1, locked=locked)
            add_id_door(self, 2, 1, id=100, door_idx=1, locked=locked)
        else:
            self.connect_all()


        if self.doors_open:
            self.open_all_doors()
        if self.oracle_mode == 'single_move':
            oracle = Oracle(color='red')
            self.place_in_room(agent_room_i, agent_room_j, oracle)
            self.oracle = oracle


        # ObjInBox
        self.n_boxes = 2
        colors = COLOR_NAMES[:self.n_boxes]
        shuffled_colors = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors)
        shuffled_colors2 = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors2)

        target_id = self._rand_int(0, self.n_boxes)
        toy_owner = self._rand_int(0, 2)
        box_owner = 1 - toy_owner
        for i in range(self.n_boxes):
            ball = Ball(color=shuffled_colors[i])
            box = BoxOverLap(color=shuffled_colors2[i], contains=ball)
            room_i, room_j = self.np_random.randint(0, self.num_cols), self._rand_int(0, self.num_rows)
            while (room_i, room_j) == (agent_room_i, agent_room_j):
                room_i, room_j = self.np_random.randint(0, self.num_cols), self._rand_int(0, self.num_rows)
            self.place_in_room(room_i, room_j, box)
            if i == target_id:
                self.box_owner = self.names[box_owner]
                self.target_box = box
                self.instrs = FindOrFailInstr(ObjDesc(ball.type, ball.color), surface='Find the key to the door, find ' + self.names[toy_owner] + ' toy',
                                              danger=True)
                room_id = (box.cur_pos[0] // self.room_size) + (box.cur_pos[1] // self.room_size) * self.num_cols






        # OpenDoor
        colors = COLOR_NAMES[0:3]
        n_keys = 2
        # Add a door of color A connecting left and middle room
        locked_door_id = self._rand_int(0, n_keys)


        for i in range(n_keys):
            obj, _ = add_id_object(self, agent_room_i, agent_room_j, id=i, kind="key", color=colors[i])
            if i == locked_door_id:
                self.real_key_color = colors[i]
        self.target_doors = []
        locked_room_i, locked_room_j = room_id // 3, room_id % 3

        for door in self.room_grid[locked_room_i][locked_room_j].doors:
            if door is not None:
                door.is_locked = True
                door.id = locked_door_id
                self.target_doors.append(door)




class Level_DangerGoToFavObjInBox(RoomGridLevel):
    def __init__(self, seed=None, room_size=7, call=True, failure_neg=False, anti_danger=True, n_rivers=3):
        self.lava_colors = ['yellow', 'blue']
        self.n_target = 2
        self.goal_pos = (room_size * 2 - 3, room_size - 2)
        self.agent_start_pos = (2, 1)
        self.call = call
        self.names = ['jack', 'mary']
        self.anti_danger = anti_danger
        self.max_n_rivers = n_rivers
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )

    def gen_mission(self):

        self.target_idx = self.np_random.randint(0, self.n_target)

        self.room_grid[0][0].door_pos[0] = (6, 1)
        self.room_grid[0][1].door_pos[2] = (6, 1)
        self.add_door(0, 0, door_idx=0, locked=False)
        # Place obstacles (lava or walls)
        if self.anti_danger:
            self.num_crossings = self.np_random.randint(1, self.max_n_rivers + 1)
            add_danger_tiles_anti_diag(self)
        self.num_crossings = self.np_random.randint(1, (self.max_n_rivers + 1))
        add_danger_tiles_diag(self)


        if self.call:
            self.grid.set(2, 1, None)
            self.agent_pos = (self.room_size - 1, 1)
        else:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.agent_pos = (1, 1)
            self.oracle_pos = (3, 1)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))
        self.agent_dir = 1
        self.open_all_doors()

        # ObjInBox
        self.n_boxes = 2
        colors = COLOR_NAMES[:self.n_boxes]
        shuffled_colors = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors)
        shuffled_colors2 = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors2)

        target_id = self._rand_int(0, self.n_boxes)
        toy_owner = self._rand_int(0, 2)
        box_owner = 1 - toy_owner
        for i in range(self.n_boxes):
            ball = Ball(color=shuffled_colors[i])
            box = Box(color=shuffled_colors2[i], contains=ball)
            if i == 0:
                place_obj(self, 1, self.room_size - 2, box)
            else:
                place_obj(self, (self.room_size - 1) * 2 - 1, self.room_size - 2, box)

            if i == target_id:
                self.box_owner = self.names[box_owner]
                self.target_box = box

                self.instrs = FindOrFailInstr(ObjDesc(ball.type, ball.color), surface='Avoid danger zone, find ' + self.names[toy_owner] + ' toy',
                                      danger=True)


class Level_DangerOpenDoorObjInBox(RoomGridLevel):
    def __init__(self, seed=None, room_size=7, call=True, failure_neg=False, anti_danger=True, n_rivers=3):
        self.lava_colors = ['yellow', 'blue']
        self.names = ['jack', 'mary']
        self.n_target = 2
        self.agent_start_pos = (2, 1)
        self.call = call
        self.failure_neg = failure_neg
        self.anti_danger = anti_danger
        self.max_n_rivers = n_rivers
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )


    def gen_mission(self):
        self.target_idx = self.np_random.randint(0, self.n_target)
        self.room_grid[0][0].door_pos[0] = (6, 1)
        self.room_grid[0][1].door_pos[2] = (6, 1)
        self.room_grid[0][1].door_pos[0] = (12, 5)

        door, _ = self.add_door(1, 0, door_idx=0, locked=False)
        door.is_open = True
        # Place obstacles (lava or walls)
        if self.anti_danger:
            self.num_crossings = self.np_random.randint(1, self.max_n_rivers + 1)
            add_danger_tiles_anti_diag(self)
        self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
        add_danger_tiles_diag(self)
        self.num_crossings = self.np_random.randint(1, self.max_n_rivers + 1)
        add_danger_tiles_anti_diag(self, offset=12)

        self.instrs = GoToGoalInstr(door=True)

        if self.call:
            self.grid.set(1, self.room_size - 2, None)
            self.agent_pos = (1, self.room_size - 2)

        else:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.agent_pos = (1, 1)
            self.oracle_pos = (3, 1)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))
        self.agent_dir = 1

        # Open Door Part
        colors = COLOR_NAMES[0:3]
        n_keys = 2
        # Add a door of color A connecting left and middle room
        locked_door_id = self._rand_int(0, n_keys)
        self.locked_door, _ = add_id_door(self, 0, 0, id=locked_door_id, door_idx=0, color=colors[0], locked=True)
        self.target_doors = [self.locked_door]
        for i in range(n_keys):
            obj = KeyWID(i, colors[i])
            if i == 0:
                self.grid.set(self.room_size - 3, 1, obj)
            elif i == 1:
                self.grid.set(self.room_size - 2, 2, obj)
            if i == locked_door_id:
                self.real_key_color = colors[i]

        # ObjInBox
        self.n_boxes = 2
        colors = COLOR_NAMES[:self.n_boxes]
        shuffled_colors = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors)
        shuffled_colors2 = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors2)

        target_id = self._rand_int(0, self.n_boxes)
        toy_owner = self._rand_int(0, 2)
        box_owner = 1 - toy_owner
        for i in range(self.n_boxes):
            ball = Ball(color=shuffled_colors[i])
            box = BoxOverLap(color=shuffled_colors2[i], contains=ball)
            if i == 0:
                place_obj(self, (self.room_size - 1) * 3 - 1, 1, box)
            else:
                place_obj(self, (self.room_size - 1) * 2 - 1, self.room_size - 2, box)

            if i == target_id:
                self.box_owner = self.names[box_owner]
                self.target_box = box
                self.instrs = FindOrFailInstr(ObjDesc(ball.type, ball.color), surface='Avoid danger zone, find the key to the door, find ' + self.names[toy_owner] + ' toy',
                                              danger=True)






# Boss level

class Level_Boss(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
            self,
            room_size=7,
            num_rows=3,
            num_cols=3,
            num_dists=3,
            doors_open=False,
            seed=None,
            all_doors=True,
            num_colors=2,
            oracle_mode='call'
    ):
        self.n_target = 2
        self.num_dists = num_dists
        self.doors_open = doors_open
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.all_doors = all_doors
        self.num_colors = num_colors
        self.names = ['jack', 'mary']
        self.lava_colors = ['yellow', 'blue']
        self.others_fav = None
        self.oracle_mode = oracle_mode
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed
        )


    def gen_mission(self):
        self.target_idx = self.np_random.randint(0, self.n_target)
        agent_room_i = self._rand_int(0, self.num_cols)
        agent_room_j = self._rand_int(0, self.num_rows)
        self.place_agent(agent_room_i, agent_room_j)


        locked = False
        if self.all_doors:
            adjust_door_to_mid(self)
            add_id_door(self, 0, 0, id=100, door_idx=0, locked=locked)
            add_id_door(self, 0, 1, id=100, door_idx=0, locked=locked)
            add_id_door(self, 1, 1, id=100, door_idx=0, locked=locked)
            add_id_door(self, 0, 2, id=100, door_idx=0, locked=locked)
            add_id_door(self, 1, 2, id=100, door_idx=0, locked=locked)

            add_id_door(self, 0, 0, id=100, door_idx=1, locked=locked)
            add_id_door(self, 1, 0, id=100, door_idx=1, locked=locked)
            add_id_door(self, 2, 0, id=100, door_idx=1, locked=locked)

            add_id_door(self, 0, 1, id=100, door_idx=1, locked=locked)
            add_id_door(self, 1, 1, id=100, door_idx=1, locked=locked)
            add_id_door(self, 2, 1, id=100, door_idx=1, locked=locked)
        else:
            self.connect_all()
        add_danger_tiles_circle(self, i_offset=0, j_offset=0)
        add_danger_tiles_circle(self, i_offset=6, j_offset=0)
        add_danger_tiles_circle(self, i_offset=12, j_offset=0)
        add_danger_tiles_circle(self, i_offset=0, j_offset=6)
        add_danger_tiles_circle(self, i_offset=6, j_offset=6)
        add_danger_tiles_circle(self, i_offset=12, j_offset=6)
        add_danger_tiles_circle(self, i_offset=0, j_offset=12)
        add_danger_tiles_circle(self, i_offset=6, j_offset=12)
        add_danger_tiles_circle(self, i_offset=12, j_offset=12)



        if self.doors_open:
            self.open_all_doors()
        if self.oracle_mode == 'single_move':
            oracle = Oracle(color='red')
            self.place_in_room(agent_room_i, agent_room_j, oracle)
            self.oracle = oracle


        # ObjInBox
        self.n_boxes = 2
        colors = COLOR_NAMES[:self.n_boxes]
        shuffled_colors = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors)
        shuffled_colors2 = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors2)

        target_id = self._rand_int(0, self.n_boxes)
        toy_owner = self._rand_int(0, 2)
        box_owner = 1 - toy_owner
        for i in range(self.n_boxes):
            ball = Ball(color=shuffled_colors[i])
            box = BoxOverLap(color=shuffled_colors2[i], contains=ball)
            room_i, room_j = self.np_random.randint(0, self.num_cols), self._rand_int(0, self.num_rows)
            while (room_i, room_j) == (agent_room_i, agent_room_j):
                room_i, room_j = self.np_random.randint(0, self.num_cols), self._rand_int(0, self.num_rows)
            self.place_in_room(room_i, room_j, box)
            if i == target_id:
                self.box_owner = self.names[box_owner]
                self.target_box = box
                self.instrs = FindOrFailInstr(ObjDesc(ball.type, ball.color), surface='Avoid danger zone, find the key to the door, find ' + self.names[toy_owner] + ' toy',
                                              danger=True)
                room_id = (box.cur_pos[0] // self.room_size) + (box.cur_pos[1] // self.room_size) * self.num_cols






        # OpenDoor
        colors = COLOR_NAMES[0:3]
        n_keys = 2
        # Add a door of color A connecting left and middle room
        locked_door_id = self._rand_int(0, n_keys)


        for i in range(n_keys):
            obj, _ = add_id_object(self, agent_room_i, agent_room_j, id=i, kind="key", color=colors[i])
            if i == locked_door_id:
                self.real_key_color = colors[i]
        self.target_doors = []
        locked_room_i, locked_room_j = room_id // 3, room_id % 3

        print(self.room_grid[locked_room_i][locked_room_j].doors)
        for door in self.room_grid[locked_room_i][locked_room_j].doors:
            if door is not None:
                door.is_locked = True
                door.id = locked_door_id
                self.target_doors.append(door)



# Register the levels in this file
register_levels(__name__, globals())
