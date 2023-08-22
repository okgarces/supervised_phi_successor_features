from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, PickupInstr
import numpy as np
import random

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
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |
    ## Observation Encoding
    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked
    ## Rewards
    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.
    ## Termination
    The episode ends if any one of the following conditions is met:
    1. The agent picks up the object.
    2. Timeout (see `max_steps`).
    """

    def __init__(self, number_of_elements, vector_to_reward, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(num_rows=1, num_cols=1, room_size=10, **kwargs)
        # Set valid objects as: (key, red), (key, blue), (key, green), (key, yellow)
        self.object_kind = 'key'
        self.object_colors_vector = ['red', 'blue', 'green', 'yellow']
        self.valid_elements = [(self.object_kind, color) for color in self.object_colors_vector]
        
        self.number_of_elements = number_of_elements
        self.vector_to_reward = vector_to_reward # [1,0,0,0] means pickup red key. [0,1,0,0] means pickup blue key.
        self.action_space.n = 4 # Only actions 0 to 3
        
        
    def _add_valid_objects(self):     
        objs = []
        
        for n in range(self.number_of_elements):
            room_i = self._rand_int(0, self.num_cols)
            room_j = self._rand_int(0, self.num_rows)
            
            obj = self.valid_elements[n % len(self.valid_elements)]
            obj, _ = self.add_object(room_i, room_j, *obj)
            objs.append(obj)
            
        return objs

    def gen_mission(self):
        # Assert vector to reward and valid elements
        assert len(self.vector_to_reward) == len(self.valid_elements)
        
        self.place_agent()
        self.connect_all()
        
        self.list_of_valid_objects = self._add_valid_objects()
        self.check_objs_reachable()
        
        # Generate mission instruction from vector to reward
        self.instrs = PickupInstr(ObjDesc(self.object_kind))
        
    def step(self, action):
        obs, env_reward, terminated, truncated, info = super().step(action)
        
        # total reward
        reward = 0
        
        # Only for fixed feature
        feature = np.zeros(len(self.object_colors_vector))
        
        if (action == 4 and (self.carrying is not None)) or terminated:
            # calculate the reward
            idx_object_in_valid_obj = self.object_colors_vector.index(self.carrying.color)
            
            # compute fixed features
            feature[idx_object_in_valid_obj] = 1
            
            reward_to_compute = self.vector_to_reward[idx_object_in_valid_obj]
            
            reward = reward_to_compute
            
            # if reward_to_compute >= 0:
            #     reward = env_reward * reward_to_compute
            # else:
            #     reward = np.abs(env_reward) * reward_to_compute
                
            self.add_object(0, 0, self.carrying.type, self.carrying.color)
            self.carrying = None
        
        info['features'] = feature
        info['env_reward'] = env_reward 
        
        return obs, reward, terminated, truncated, info