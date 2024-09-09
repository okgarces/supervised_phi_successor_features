import torch


class QNetwork(torch.nn.Module):

    def __init__(self, input_shape, outputs):
        super().__init__()

        q_layers = [
            torch.nn.Linear(input_shape, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, outputs)
        ]
        self._q_network = torch.nn.Sequential(*q_layers)

    def forward(self, inputs):
        q_values = self._q_network(inputs)
        return q_values
