from typing import Optional

from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from minigrid.manual_control import ManualControl


class CartpoleDissimilar(CartPoleEnv):
    def __init__(self, pole_length = 0.5, render_mode: Optional[str] = None):
        # actually half the pole's length
        super().__init__(render_mode)

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = pole_length  # actually half the pole's length
        self.polemass_length = self.masspole * self.length


if __name__ == '__main__':
    env = CartpoleDissimilar(12, render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()