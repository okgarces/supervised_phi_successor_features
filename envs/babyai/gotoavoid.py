from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, PickupInstr
from minigrid.core.world_object import Ball, Box, Key, WorldObj

import numpy as np

from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class OverlapableObject(WorldObj):

    def __init__(self, object: WorldObj):
        super().__init__(object.type, object.color)
        self.object = object

    def can_overlap(self) -> bool:
        return True

    def can_pickup(self) -> bool:
        return self.object.can_pickup()

    def render(self, r: np.ndarray) -> np.ndarray:
        return self.object.render(r)


def reject_next_to(env: MiniGridEnv, pos: tuple[int, int]):
    """
    Function to filter out object positions that are right next to
    the agent's starting point
    """

    sx, sy = env.agent_pos
    x, y = pos
    d = abs(sx - x) + abs(sy - y)
    return d < 2

class PickupAndAvoid(RoomGridLevel):
    """
    ## Description
    Pick up an object, the object may be in another room.
    ## Mission Space
    "pick up a {color} {type}"
    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".
    {type} is the type of the object. Can be "ball", "box" or "key".
    ## Action Space
    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    ## Observation Encoding
    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked
    ## Rewards
    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.
    ## Termination
    1. This env does not terminate only truncated due to timeout.
    ## Truncation
    1. Timeout (see `max_steps`).
    """

    def __init__(self, number_of_elements, vector_to_reward, use_pick_action = True, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(num_rows=1, num_cols=1, room_size=10, **kwargs)
        # Set valid objects as: (key, red), (key, blue), (key, green), (key, yellow)
        self.object_kind = 'ball'
        self.object_colors_vector = ['red', 'blue', 'green', 'yellow']
        self.valid_elements = [(self.object_kind, color) for color in self.object_colors_vector]
        
        self.number_of_elements = number_of_elements
        self.vector_to_reward = vector_to_reward # [1,0,0,0] means pickup red key. [0,1,0,0] means pickup blue key.
        self.action_space.n = 4 if use_pick_action else 3 # Only actions 0 to 3
        self.use_pick_action = use_pick_action
        
    def place_in_room(self, i: int, j: int, obj: WorldObj, append=True) -> tuple[WorldObj, tuple[int, int]]:
        """
        Add an existing object to room (i, j). This method is called by add_object in room grid level.
        """
        room = self.get_room(i, j)
        top = (room.top[0] + 1, room.top[1] + 1)
        size = (room.size[0] - 1, room.size[1] - 1)
        pos = self.place_obj(obj, top, size, reject_fn=reject_next_to, max_tries=1_000_000)

        if append:
            room.objs.append(obj)

        return obj, pos

    def _make_valid_object(self, obj: tuple):
        obj_type, obj_color = obj
        
        if obj_type == "key":
            obj = Key(obj_color)
        elif obj_type == "ball":
            obj = Ball(obj_color)
        elif obj_type == "box":
            obj = Box(obj_color)
        else:
            raise ValueError(
                f"{obj_type} object kind is not available in this environment."
            )

        if not self.use_pick_action:
            obj = OverlapableObject(obj)

        return obj
    
    def _add_valid_objects(self):     
        objs = []
        
        for n in range(self.number_of_elements):
            room_i = self._rand_int(0, self.num_cols)
            room_j = self._rand_int(0, self.num_rows)
            
            obj = self.valid_elements[n % len(self.valid_elements)]
            obj = self._make_valid_object(obj)
            objs.append(self.place_in_room(room_i, room_j, obj))
            
        return objs

    def gen_mission(self):
        # Assert vector to reward and valid elements
        assert len(self.vector_to_reward) == len(self.valid_elements)
        
        self.place_agent()
        self.connect_all()
        
        self.list_of_valid_objects = self._add_valid_objects()
        # self.check_objs_reachable() # do we need to check reachability for all objects?
        
        # Generate mission instruction from vector to reward
        self.instrs = PickupInstr(ObjDesc(self.object_kind))

    def is_object_reached(self):
        if self.use_pick_action:
            return self.carrying is not None
        else:
            is_reached = False
            fwd_cell = self.grid.get(*self.agent_pos)

            if fwd_cell is not None and fwd_cell.can_pickup():
                is_reached = True
                self.carrying = fwd_cell
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
            return is_reached

    def step(self, action):

        if not self.use_pick_action:
            if action == 3:
                action = 5

        # If action == 4. It's a not action
        obs, env_reward, terminated, truncated, info = super().step(action)
        
        # total reward
        reward = 0
        
        # Only for fixed feature
        feature = np.zeros(len(self.object_colors_vector))
        
        if (self.is_object_reached()) or terminated:
            # calculate the reward
            idx_object_in_valid_obj = self.object_colors_vector.index(self.carrying.color)
            
            # compute fixed features
            feature[idx_object_in_valid_obj] = 1
            
            reward_to_compute = self.vector_to_reward[idx_object_in_valid_obj]
            
            reward = reward_to_compute

            terminated = 0
            # self.add_object(0, 0, self.carrying.type, self.carrying.color)
            obj = self._make_valid_object((self.carrying.type, self.carrying.color))
            self.place_in_room(0, 0, obj, append=False)

            self.carrying = None
        
        info['features'] = feature
        info['env_reward'] = env_reward 
        
        return obs, reward, terminated, truncated, info

    def get_no_op_action(self):
        return 4

if __name__ == '__main__':
    env = PickupAndAvoid(12,[1,1,1,1], render_mode="human", max_steps=200)

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
