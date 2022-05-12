import gym
from .verifier import *
from .levelgen import *
from .objects import DoorWID, KeyWID, BoxWID
import random
import copy
from gym_minigrid.envs import Key, Ball, Box

class Oracle(Ball):
    def __init__(self, color):
        super().__init__(color)

    def can_overlap(self):
        return True



class Level_MultiKeys(RoomGridLevel):
    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=3,
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
        self.locked_door, _ = self.add_id_door(0, 0, id=locked_door_id, door_idx=0, color=colors[0], locked=True)

        # Add a door of color B connecting middle and right room
        self.add_id_door(1, 0, id=1, door_idx=0, color=colors[1], locked=False)
        for i in range(n_keys):
            if self._rand_int(0, 2) == 0:
                key_col = 1
            else:
                key_col = 2
            obj, _ = self.add_id_object(key_col, 0, id=i, kind="key", color=colors[i])
            if i == locked_door_id:
                self.real_key_color = colors[i]
        self.place_agent(1, 0)
        self.instrs = OpenInstr(ObjDesc(self.locked_door.type, color=self.locked_door.color))

    def add_id_door(self, i, j, id, door_idx=None, color=None, locked=None):
        """
        Add an id door to a room, connecting it to a neighbor
        """

        room = self.get_room(i, j)

        if door_idx == None:
            # Need to make sure that there is a neighbor along this wall
            # and that there is not already a door
            while True:
                door_idx = self._rand_int(0, 4)
                if room.neighbors[door_idx] and room.doors[door_idx] is None:
                    break

        if color == None:
            color = self._rand_color()

        if locked is None:
            locked = self._rand_bool()

        assert room.doors[door_idx] is None, "door already exists"

        room.locked = locked
        door = DoorWID(color, id=id, is_locked=locked)

        pos = room.door_pos[door_idx]
        self.grid.set(*pos, door)
        door.cur_pos = pos

        neighbor = room.neighbors[door_idx]
        room.doors[door_idx] = door
        neighbor.doors[(door_idx+2) % 4] = door

        return door, pos

    def add_id_object(self, i, j, id, kind=None, color=None):
        """
        Add a new id object to room (i, j)
        """

        if kind == None:
            kind = self._rand_elem(['key', 'ball', 'box'])

        if color == None:
            color = self._rand_color()

        # TODO: we probably want to add an Object.make helper function
        assert kind in ['key', 'ball', 'box']
        if kind == 'key':
            obj = KeyWID(id, color)
        elif kind == 'ball':
            raise NotImplementedError
        elif kind == 'box':
            raise NotImplementedError

        return self.place_in_room(i, j, obj)


class Level_MultiKeysGTAns(Level_MultiKeys):
    def __init__(self, seed=None):
        super(Level_MultiKeysGTAns, self).__init__(
            seed=seed
        )

    def gen_mission(self):
        super().gen_mission()
        self.instrs = OpenInstrGTAns(ObjDesc(self.locked_door.type, color=self.locked_door.color), self.real_key_color + ' key')




class Level_GoToBall(RoomGridLevel):
    """
    Unlock a door A that requires to unlock a door B before
    """

    def __init__(self, seed=None):
        room_size = 6
        max_steps = 8
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            seed=seed,
            #max_steps=max_steps
        )


    def gen_mission(self):
        #colors = self._rand_subset(COLOR_NAMES, 2)
        colors = COLOR_NAMES[0:2]
        # Add a door of color A connecting left and middle room
        self.add_door(0, 0, door_idx=0, color=colors[0], locked=False)

        # Add a door of color B connecting middle and right room
        self.add_door(1, 0, door_idx=0, color=colors[0], locked=False)

        if self._rand_int(0, 2) == 0:
            ball_col = 0
        else:
            ball_col = 2
        obj, _ = self.add_object(ball_col, 0, kind="ball")

        self.place_agent(1, 0)
        self.instrs = GoToInstr(ObjDesc(obj.type))


class Level_GoToBallGTAns(RoomGridLevel):
    """
    Unlock a door A that requires to unlock a door B before
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        #colors = self._rand_subset(COLOR_NAMES, 2)
        colors = COLOR_NAMES[0:2]
        # Add a door of color A connecting left and middle room
        self.add_door(0, 0, door_idx=0, color=colors[0], locked=False)

        # Add a door of color B connecting middle and right room
        self.add_door(1, 0, door_idx=0, color=colors[0], locked=False)

        if self._rand_int(0, 2) == 0:
            ball_col = 0
        else:
            ball_col = 2
        obj, _ = self.add_object(ball_col, 0, kind="ball")

        self.place_agent(1, 0)

        #self.instrs = PickupInstr(ObjDesc(obj.type))
        #self.instrs = GoToGTAns(ObjDesc(obj.type), 'Room{} Room{} Room{} Room{}'.format(ball_col, ball_col, ball_col, ball_col))
        self.instrs = GoToGTAns(ObjDesc(obj.type), 'Ball in Room{}'.format(ball_col))


class Level_GoToTwoBall(RoomGridLevel):
    """
    Unlock a door A that requires to unlock a door B before
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        colors = COLOR_NAMES[0:2]
        # Add a door of color A connecting left and middle room
        self.add_door(0, 0, door_idx=0, color=colors[0], locked=False)

        # Add a door of color B connecting middle and right room
        self.add_door(1, 0, door_idx=0, color=colors[0], locked=False)
        if self._rand_int(0, 2) == 0:
            c1, c2 = colors[0], colors[1]
        else:
            c1, c2 = colors[1], colors[0]
        if self._rand_int(0, 2) == 0:
            ball1_col = 0
        else:
            ball1_col = 2
        if self._rand_int(0, 2) == 0:
            ball2_col = 0
        else:
            ball2_col = 2
        ball1, _ = self.add_object(ball1_col, 0, kind="ball", color=c1)
        ball2, _ = self.add_object(ball2_col, 0, kind="ball", color=c2)
        target_ball = random.choice([ball1, ball2])
        self.place_agent(1, 0)
        self.instrs = GoToInstr(ObjDesc(target_ball.type, color=target_ball.color))



class Level_General(RoomGridLevel):
    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            seed=seed
        )
        self.real_key_color = None


    def gen_mission(self):

        if self._rand_int(0, 2) == 0:
            task = 'key'
        else:
            task = 'ball'

        colors = COLOR_NAMES[0:3]
        n_keys = 3
        # Add a door of color A connecting left and middle room
        locked_door_id = self._rand_int(0, n_keys)
        self.locked_door, _ = self.add_id_door(0, 0, id=locked_door_id, door_idx=0, color=colors[0], locked=True if task == 'key' else False)

        # Add a door of color B connecting middle and right room
        self.add_id_door(1, 0, id=1, door_idx=0, color=colors[1], locked=False)
        for i in range(n_keys):
            if self._rand_int(0, 2) == 0:
                key_col = 1
            else:
                key_col = 2
            obj, _ = self.add_id_object(key_col, 0, id=i, kind="key", color=colors[i])
            if i == locked_door_id:
                self.real_key_color = colors[i]

        if self._rand_int(0, 2) == 0:
            ball_col = 0
        else:
            ball_col = 2
        obj, _ = self.add_object(ball_col, 0, kind="ball")

        if ball_col == 0:
            loc = 'left'
        else:
            loc = 'right'

        self.place_agent(1, 0)
        if task == 'ball':
            self.instrs = GoToGTAns(ObjDesc(obj.type), loc + ' Room')
        else:
            self.instrs = OpenInstrGTAns(ObjDesc(self.locked_door.type, color=self.locked_door.color), self.real_key_color + ' key')

    def add_id_door(self, i, j, id, door_idx=None, color=None, locked=None):
        """
        Add an id door to a room, connecting it to a neighbor
        """

        room = self.get_room(i, j)

        if door_idx == None:
            # Need to make sure that there is a neighbor along this wall
            # and that there is not already a door
            while True:
                door_idx = self._rand_int(0, 4)
                if room.neighbors[door_idx] and room.doors[door_idx] is None:
                    break

        if color == None:
            color = self._rand_color()

        if locked is None:
            locked = self._rand_bool()

        assert room.doors[door_idx] is None, "door already exists"

        room.locked = locked
        door = DoorWID(color, id=id, is_locked=locked)

        pos = room.door_pos[door_idx]
        self.grid.set(*pos, door)
        door.cur_pos = pos

        neighbor = room.neighbors[door_idx]
        room.doors[door_idx] = door
        neighbor.doors[(door_idx+2) % 4] = door

        return door, pos

    def add_id_object(self, i, j, id, kind=None, color=None):
        """
        Add a new id object to room (i, j)
        """

        if kind == None:
            kind = self._rand_elem(['key', 'ball', 'box'])

        if color == None:
            color = self._rand_color()

        # TODO: we probably want to add an Object.make helper function
        assert kind in ['key', 'ball', 'box']
        if kind == 'key':
            obj = KeyWID(id, color)
        elif kind == 'ball':
            raise NotImplementedError
        elif kind == 'box':
            raise NotImplementedError

        return self.place_in_room(i, j, obj)


class Level_GoToBallMaze(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
            self,
            room_size=5,
            num_rows=3,
            num_cols=3,
            num_dists=1,
            doors_open=True,
            seed=None,
            all_doors=True
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.all_doors = all_doors
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
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
        num_colors = 2
        obj, _ = self.add_object(self._rand_int(0, self.num_cols), self._rand_int(0, self.num_rows), 'ball', color=self._rand_elem(COLOR_NAMES[:num_colors]))
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
        if self.doors_open:
            self.open_all_doors()


class Level_GoToBallMazeS5N1A0(Level_GoToBallMaze):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_dists=1, all_doors=False, seed=seed)


class Level_GoToBallMazeS8N1A0(Level_GoToBallMaze):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=1, all_doors=False, seed=seed)


class Level_GoToBallMazeS5N1A1O0(Level_GoToBallMaze):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_dists=1, all_doors=True, doors_open=False, seed=seed)


class Level_GoToBallMazeS8N1A1O0(Level_GoToBallMaze):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=1, all_doors=True, doors_open=False, seed=seed)


class Level_GoToObjMaze2(RoomGridLevel):
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
            num_colors=2
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.all_doors = all_doors
        self.num_colors = num_colors
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
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

        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=True, num_colors=self.num_colors)
        obj = self._rand_elem(objs)
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
        if self.doors_open:
            self.open_all_doors()

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
            type = self._rand_elem(['key', 'ball', 'box'])
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



class Level_GoToFavorite(RoomGridLevel):
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

        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=True, num_colors=self.num_colors)
        obj = self._rand_elem(objs)
        self.check_objs_reachable()
        random.shuffle(self.names)
        self.instrs = FavoriteInstr(ObjDesc(obj.type, obj.color), name=self.names[0])
        room_id = (obj.cur_pos[0] // self.room_size) + (obj.cur_pos[1] // self.room_size) * self.num_cols
        self.useful_answers = ['{} toy is {} {}'.format(self.names[0], obj.color, obj.type),
                               '{} {} in room{}'.format(obj.color, obj.type, room_id)]
        #print('Useful answer:', self.useful_answers)
        self.others_fav = self._rand_elem(objs)
        if self.doors_open:
            self.open_all_doors()
        if self.oracle_mode == 'single_move':
            oracle = Oracle(color='red')
            self.place_in_room(agent_room_i, agent_room_j, oracle)
            self.oracle = oracle


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


class Level_GoToFavoriteSingleMove(Level_GoToFavorite):
    def __init__(self, seed=None):
        super().__init__(seed=seed, oracle_mode='single_move')



class Level_ObjInLockedBox(RoomGridLevel):
    def __init__(self, seed=None, n_keys=3, room_size=9):
        room_size = room_size
        self.real_key_color = None
        self.n_keys = n_keys
        self.instrs = None
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        #colors = self._rand_subset(COLOR_NAMES, 2)
        colors = COLOR_NAMES[:self.n_keys]
        shuffled_colors = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors)
        shuffled_colors2 = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors2)
        target_id = self._rand_int(0, self.n_keys)

        for i in range(self.n_keys):
            key = KeyWID(color=colors[i], id=i)
            self.place_in_room(0, 0, key)
            ball = Ball(color=shuffled_colors[i])
            box = BoxWID(color=shuffled_colors2[i], id=i, contains=ball)
            self.place_in_room(0, 0, box)
            if i == target_id:
                self.instrs = FindInstr(ObjDesc(ball.type, ball.color))
                self.target_key_color = key.color
        assert self.instrs
        self.place_agent(0, 0)


class Level_ObjInLockedBoxOne(Level_ObjInLockedBox):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            n_keys=1)

class Level_ObjInLockedBoxTwo(Level_ObjInLockedBox):
    def __init__(self, seed=None, room_size=8):
        super().__init__(
            seed=seed,
            n_keys=2)


class Level_ObjInLockedBoxThree(Level_ObjInLockedBox):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            n_keys=3)


class Level_ObjInBox(RoomGridLevel):
    def __init__(self, seed=None, n_boxes=2, room_size=9):
        room_size = room_size
        self.instrs = None
        self.n_boxes = n_boxes
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        colors = COLOR_NAMES[:self.n_boxes + 1]
        shuffled_colors = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors)
        shuffled_colors2 = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors2)
        target_id = self._rand_int(0, self.n_boxes)

        for i in range(self.n_boxes):
            ball = Ball(color=shuffled_colors[i])
            box = Box(color=shuffled_colors2[i], contains=ball)
            self.place_in_room(0, 0, box)
            if i == target_id:
                self.instrs = FindOrFailInstr(ObjDesc(ball.type, ball.color))
        assert self.instrs
        self.place_agent(0, 0)


class Level_ObjInBoxV2(RoomGridLevel):
    def __init__(self, seed=None, n_boxes=2, room_size=9, enough_info=False, oracle_mode='single_call', failure_neg=False):
        room_size = room_size
        self.instrs = None
        self.n_boxes = n_boxes
        self.names = ['jack', 'mary']
        self.enough_info = enough_info
        self.oracle_mode = oracle_mode # single_call, multi_call, single_move, multi_move
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed,
            failure_neg=failure_neg
        )

    def gen_mission(self):
        if self.oracle_mode == 'multi_move':
            oracle1 = Oracle(color='red')
            _, self.mary_pos = self.place_in_room(0, 0, oracle1)
            oracle2 = Oracle(color='yellow')
            _, self.jack_pos = self.place_in_room(0, 0, oracle2)
        elif self.oracle_mode == 'single_move':
            oracle = Oracle(color='red')
            _, self.mary_pos = self.place_in_room(0, 0, oracle)
            self.jack_pos = self.mary_pos

        colors = COLOR_NAMES[:self.n_boxes]
        shuffled_colors = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors)
        shuffled_colors2 = copy.deepcopy(colors)
        self.np_random.shuffle(shuffled_colors2)

        target_id = self._rand_int(0, self.n_boxes)
        toy_owner = self._rand_int(0, 2)
        box_owner = 1 - toy_owner
        #self.np_random.shuffle(self.names)
        #print(self.names)
        for i in range(self.n_boxes):
            ball = Ball(color=shuffled_colors[i])
            box = Box(color=shuffled_colors2[i], contains=ball)
            self.place_in_room(0, 0, box)
            if i == target_id:
                self.box_owner = self.names[box_owner]
                self.target_box = box
                if self.enough_info:
                    self.instrs = FindOrFailInstr(ObjDesc(ball.type, ball.color), surface='find ' + self.names[toy_owner] + ' toy, which is in the {} box'.format(self.target_box.color))
                else:
                    self.instrs = FindOrFailInstr(ObjDesc(ball.type, ball.color), surface='find ' + self.names[toy_owner] + ' toy')

                self.useful_answers = ['{} toy is {} ball'.format(self.names[toy_owner], ball.color),
                                   '{} ball in {} suitcase'.format(ball.color, self.names[box_owner]),
                                   '{} suitcase is {} box'.format(self.names[box_owner], self.target_box.color)]
        assert self.instrs
        self.place_agent(0, 0)


class Level_ObjInBoxV2SingleMove(Level_ObjInBoxV2):
    def __init__(self, seed=None, n_boxes=2, room_size=9):
        super().__init__(seed=seed, n_boxes=n_boxes, room_size=room_size, enough_info=False, oracle_mode='single_move')

class Level_ObjInBoxV2MultiMove(Level_ObjInBoxV2):
    def __init__(self, seed=None, n_boxes=2, room_size=9):
        super().__init__(seed=seed, n_boxes=n_boxes, room_size=room_size, enough_info=False, oracle_mode='multi_move')

class Level_ObjInBoxMultiSufficient(Level_ObjInBoxV2):
    def __init__(self, seed=None, n_boxes=2, room_size=9):
        super().__init__(seed=seed, n_boxes=n_boxes, room_size=room_size, enough_info=True)

class Level_ObjInBoxMultiMultiOracle(Level_ObjInBoxV2):
    def __init__(self, seed=None, n_boxes=2, room_size=9):
        super().__init__(seed=seed, n_boxes=n_boxes, room_size=room_size, enough_info=False, oracle_mode='multi_move')

class Level_ObjInBoxMultiMultiOracleSufficient(Level_ObjInBoxV2):
    def __init__(self, seed=None, n_boxes=2, room_size=9):
        super().__init__(seed=seed, n_boxes=n_boxes, room_size=room_size, enough_info=True, oracle_mode='multi_move')


class Level_ObjInBoxV2Neg(Level_ObjInBoxV2):
    def __init__(self, seed=None):
        super().__init__(seed=seed, failure_neg=True)



# Register the levels in this file
register_levels(__name__, globals())
