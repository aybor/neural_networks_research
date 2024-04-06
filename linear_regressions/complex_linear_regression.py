from torch import nn


class ComplexLinearRegressionNN(nn.Module):
    def __init__(self, input_size):
        super(ComplexLinearRegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.dropout1(x)
        x = self.leakyrelu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout2(x)
        x = self.fc4(x)
        return x