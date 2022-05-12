from gym_minigrid.minigrid import Door, Key, Box


class DoorWID(Door):
    def __init__(self, color, id, is_open=False, is_locked=False):
        super(DoorWID, self).__init__(color, is_open, is_locked)
        self.id = id

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, KeyWID) and env.carrying.id == self.id:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True


class KeyWID(Key):
    def __init__(self, id, color='blue'):
        super(KeyWID, self).__init__(color)
        self.id = id

    def can_overlap(self):
        return True


class BoxWID(Box):
    def __init__(self, color, id, contains=None, locked=True):
        super(BoxWID, self).__init__(color)
        self.contains = contains
        self.is_locked = locked
        self.id = id

    def can_pickup(self):
        return True

    def toggle(self, env, pos):
        if self.is_locked:
            if isinstance(env.carrying, KeyWID) and env.carrying.id == self.id:
                self.is_locked = False
                self.is_open = True
                return True
            return False
        else:
            # Replace the box by its contents
            env.grid.set(*pos, self.contains)
        return True