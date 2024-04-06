from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from linear_regressions import SimpleLinearRegressionNN, ComplexLinearRegressionNN, EnhancedLinearRegressionNN
from utils import generate_dataset


base_path = Path(__file__).parent


if __name__ == '__main__':
    input_size = 128  # 64 точки на две координаты

    torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    simpe_model = SimpleLinearRegressionNN(input_size)
    complex_model = ComplexLinearRegressionNN(input_size)
    enhanced_model = EnhancedLinearRegressionNN(input_size)

    criterion = nn.MSELoss()

    simple_optimizer = optim.Adam(simpe_model.parameters(), lr=0.001)
    complex_optimizer = optim.Adam(complex_model.parameters(), lr=0.001)
    enhanced_optimizer = optim.Adam(enhanced_model.parameters(), lr=0.001)

    eras = 10
    epochs = 500

    for era in range(eras):
        inputs, targets = generate_dataset(num_samples=4096)
        dataset = TensorDataset(inputs, targets)

        for epoch in range(epochs):
            batch_size = 64
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                simple_optimizer.zero_grad()
                complex_optimizer.zero_grad()
                enhanced_optimizer.zero_grad()

                simple_outputs = simpe_model(inputs)
                complex_outputs = complex_model(inputs)
                enhanced_outputs = enhanced_model(inputs)

                simple_loss = criterion(simple_outputs, targets)
                complex_loss = criterion(complex_outputs, targets)
                enhanced_loss = criterion(enhanced_outputs, targets)

                simple_loss.backward()
                complex_loss.backward()
                enhanced_loss.backward()

                simple_optimizer.step()
                complex_optimizer.step()
                enhanced_optimizer.step()

                if batch_idx + 1 == len(dataloader):
                    print(f'Era [{era}/{eras}], Epoch [{epoch + 1}/{epochs}], simple_loss: {simple_loss.item():.4f}, enhanced_loss: {enhanced_loss.item():.4f}, complex_loss: {complex_loss.item():.4f}')

    torch.save(simpe_model.state_dict(), base_path / 'models' / 'simple_model_.pth')
    torch.save(complex_model.state_dict(), base_path / 'models' / 'complex_model_.pth')
    torch.save(enhanced_model.state_dict(), base_path / 'models' / 'enhanced_model_.pth')
