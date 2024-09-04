import torch


class DQNConvolutionalNetwork(torch.nn.Module):

    def __init__(self, input_shape, outputs):
        super().__init__()

        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_shape[1])))
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_shape[2])))

        linear_input_size = convw * convh * 32 # output channels
        self.fc1 = torch.nn.Linear(linear_input_size, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, outputs)

        layers = [
            torch.nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
        ]
        q_layers = [
            torch.nn.Linear(linear_input_size, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, outputs)
        ]
        self._vision = torch.nn.Sequential(*layers)
        self._q_network = torch.nn.Sequential(*q_layers)

    def conv2d_size_out(self, size, kernel_size=3, stride=2):
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, inputs):
        q_values = self._vision(inputs)
        q_values = q_values.reshape(inputs.size(0), -1)
        q_values = self._q_network(q_values)
        return q_values
