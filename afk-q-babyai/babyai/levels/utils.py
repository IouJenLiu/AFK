from gym_minigrid.minigrid import fill_coords, point_in_rect, point_in_line, WorldObj
import itertools as itt
from gym_minigrid.envs import Key, Ball, Box
from .objects import DoorWID, KeyWID

class BoxOverLap(Box):
    def __init__(self, color, contains):
        super(BoxOverLap, self).__init__(color, contains)
    def can_overlap(self):
        return True


class Lava(WorldObj):
    def __init__(self, color):
        super().__init__('lava', color)
        assert color in ['blue', 'yellow']
        self.color = color

    def can_overlap(self):
        return True

    def render(self, img):
        if self.color == 'yellow':
            c = (255, 128, 0)
        elif self.color == 'blue':
            c = (0, 0, 255)
        else:
            raise NotImplementedError

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0,0,0))


def create_rnd_obj(self):
    kind = self._rand_elem(['key', 'ball', 'box'])
    color = self._rand_color()
    if kind == 'key':
        return Key(color)
    elif kind == 'ball':
        return Ball(color)
    elif kind == 'box':
        return Box(color)


def get_room_xy(room_size, i, j):
    room_x = i // room_size
    room_y = j // room_size
    return room_x, room_y


def place_obj(self, x, y, obj):
    self.grid.set(x, y, obj)
    obj.cur_pos = (x, y)
    self.get_room(*get_room_xy(self.room_size, x, y)).objs.append(obj)


def on_danger_zone(env):
    # return 1 if the agent is on the danger zone
    linear_idx = env.agent_pos[1] * env.grid.width + env.agent_pos[0]
    cur_cell = env.grid.grid[linear_idx]
    if cur_cell and cur_cell.type == 'lava' and cur_cell.color != env.lava_colors[env.target_idx]:
        return 1
    return 0



def add_danger_tiles_anti_diag(self, offset=0):

    height, width = self.room_size, self.room_size
    v, h = object(), object()  # singleton `vertical` and `horizontal` objects

    # Lava rivers or walls specified by direction and position in grid
    rivers = [(v, i) for i in range(2, height - 2, 2)]
    rivers += [(h, j) for j in range(2, width - 2, 2)]
    self.np_random.shuffle(rivers)

    # debug
    #self.num_crossings = 2
    #rivers = [(h, 2), (v, 4)]
    ##
    rivers = rivers[:self.num_crossings]  # sample random rivers
    rivers_v = sorted([pos + offset for direction, pos in rivers if direction is v])
    rivers_h = sorted([pos for direction, pos in rivers if direction is h])


    obstacle_pos = itt.chain(
        itt.product(range(1 + offset, width - 1 + offset), rivers_h),
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
    #limits_v = [0] + rivers_v + [height - 1]
    #limits_h = [0] + rivers_h + [width - 1]
    #limits_v = [height - 1 + offset] + rivers_v[::-1] + [0 + offset]
    #limits_h = [width - 1] + rivers_h[::-1] + [0]


    limits_v = [0 + offset] + rivers_v + [height - 1 + offset]
    limits_h = [width - 1] + rivers_h[::-1] + [0]

    room_i, room_j = 0, 0
    openings = set()
    for direction in path:
        if direction is h:
            i = limits_v[room_i + 1]
            #j = self.np_random.choice(
            #    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
            j = self.np_random.choice(
                range(limits_h[room_j + 1] + 1, limits_h[room_j]))
            room_i += 1
        elif direction is v:
            i = self.np_random.choice(
                range(limits_v[room_i] + 1, limits_v[room_i + 1]))
            #i = self.np_random.choice(
            #    range(limits_v[room_i + 1] + 1, limits_v[room_i]))
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


def add_danger_tiles_diag(self, offset=6):
    height, width = self.room_size, self.room_size
    v, h = object(), object()
    rivers = [(v, i) for i in range(2, height - 2, 2)]
    rivers += [(h, j) for j in range(2 , width - 2, 2)]
    self.np_random.shuffle(rivers)
    rivers = rivers[:self.num_crossings]  # sample random rivers
    rivers_v = sorted([pos + offset for direction, pos in rivers if direction is v])
    rivers_h = sorted([pos for direction, pos in rivers if direction is h])
    obstacle_pos = itt.chain(
        itt.product(range(1 + offset, width - 1 + offset), rivers_h),
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
    limits_v = [0 + offset ] + rivers_v + [height - 1 + offset]
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


def add_danger_tiles_middle(self, offset=6):
    height, width = self.room_size, self.room_size
    v, h = object(), object()
    rivers = [(v, 2), (v, 4), (h, 4)]
    self.np_random.shuffle(rivers)
    rivers = rivers[:self.num_crossings]  # sample random rivers
    rivers_v = sorted([pos + offset for direction, pos in rivers if direction is v])
    rivers_h = sorted([pos for direction, pos in rivers if direction is h])
    obstacle_pos = itt.chain(
        itt.product(range(1 + offset, width - 1 + offset), rivers_h),
        itt.product(rivers_v, range(1, height - 1)),
    )

    colored_tiles = [set(), set()]
    for i, j in obstacle_pos:
        color_idx = self.np_random.randint(0, 2)
        self.grid.set(i, j, Lava(self.lava_colors[color_idx]))
        colored_tiles[color_idx].add((i, j))
    for river_i in [2, 4]:
        if (v, river_i) in rivers:
            i, j = river_i + offset, self.np_random.choice(range(1, 4))
            self.grid.set(i, j, Lava(self.lava_colors[self.target_idx]))


def add_danger_tiles_circle(self, i_offset=0, j_offset=0):
    height, width = self.room_size, self.room_size
    v, h = object(), object()
    rivers = [(v, i) for i in range(2, height - 2, 2)]
    rivers += [(h, j) for j in range(2 , width - 2, 2)]
    self.np_random.shuffle(rivers)
    self.num_crossings = self.np_random.randint(1, (self.room_size - 3))
    #self.num_crossings = 4
    rivers = rivers[:self.num_crossings]  # sample random rivers
    rivers_v = sorted([pos + i_offset for direction, pos in rivers if direction is v])
    rivers_h = sorted([pos + j_offset for direction, pos in rivers if direction is h])
    obstacle_pos = itt.chain(
        itt.product(range(1 + i_offset, width - 1 + i_offset), rivers_h),
        itt.product(rivers_v, range(1 + j_offset, height - 1 + j_offset)),
    )

    colored_tiles = [set(), set()]
    for i, j in obstacle_pos:
        color_idx = self.np_random.randint(0, 2)
        self.grid.set(i, j, Lava(self.lava_colors[color_idx]))
        colored_tiles[color_idx].add((i, j))

    path = [h] * len(rivers_v) + [v] * len(rivers_h)
    #self.np_random.shuffle(path)

    # Create openings
    limits_v = [0 + i_offset] + rivers_v + [height - 1 + i_offset]
    limits_h = [0 + j_offset] + rivers_h + [width - 1 + j_offset]
    room_i, room_j = 0, 0
    openings = set()

    def set_lave(i, j):
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
        set_lave(i, j)

    path_back = [h] * len(rivers_v) + [v] * len(rivers_h)

    for direction in path_back:
        if direction is h:
            i = limits_v[room_i]
            j = self.np_random.choice(
                range(limits_h[room_j] + 1, limits_h[room_j + 1]))
            room_i -= 1
        elif direction is v:
            i = self.np_random.choice(
                range(limits_v[room_i] + 1, limits_v[room_i + 1]))
            j = limits_h[room_j]
            room_j -= 1
        else:
            assert False
        set_lave(i, j)





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


def add_id_object_abs(self, i, j, id, kind=None, color=None):
    """
    Add a new id object to cell (i, j)
    """
    obj = create_id_obj(self, id, kind, color)
    self.grid.set(i, j, obj)
    return obj


def create_id_obj(self, id, kind=None, color=None):
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
    return obj


def adjust_door_to_mid(self):
    for j in range(0, self.num_rows):
        # For each column of rooms
        for i in range(0, self.num_cols):
            room = self.room_grid[j][i]
            x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
            x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)
            # Door positions, order is right, down, left, up
            if i < self.num_cols - 1:
                room.neighbors[0] = self.room_grid[j][i+1]
                #room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                room.door_pos[0] = (x_m, (y_l + y_m) // 2)
            if j < self.num_rows - 1:
                room.neighbors[1] = self.room_grid[j+1][i]
                #room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                room.door_pos[1] = ((x_l + x_m) // 2, y_m)
            if i > 0:
                room.neighbors[2] = self.room_grid[j][i-1]
                room.door_pos[2] = room.neighbors[2].door_pos[0]
            if j > 0:
                room.neighbors[3] = self.room_grid[j-1][i]
                room.door_pos[3] = room.neighbors[3].door_pos[1]

