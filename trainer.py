import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from network.neural_net import NeuralNet
from data.dataset import HamiltonianDataset  

def train(train_csv):
   
    train_dataset = HamiltonianDataset(train_csv)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = NeuralNet().float()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for i, (batch_pq, batch_H) in enumerate(train_dataloader):
            batch_pq = batch_pq.float()
            batch_H = batch_H.float()

             
            outputs = model(batch_pq)
            loss = criterion(outputs.squeeze(), batch_H)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.6f}')


    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train("C:\\Users\\brend\\OneDrive\\Desktop\\HamiltonianBNN\\data\\train_data.csv")