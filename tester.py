import torch
from torch.utils.data import DataLoader
from network.neural_net import NeuralNet  
from data.dataset import HamiltonianDataset  

def test(test_csv, tolerance=0.1):
    
    test_dataset = HamiltonianDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = NeuralNet().float()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    n_correct = 0
    n_samples = len(test_loader.dataset)
    total_error = 0

    with torch.no_grad():
        for i, (batch_pq, batch_H) in enumerate(test_loader):
            batch_pq = batch_pq.float()
            batch_H = batch_H.float()

            outputs = model(batch_pq).squeeze()
            n_correct += torch.sum(torch.abs(outputs - batch_H) < tolerance).item()
            
            criterion = torch.nn.MSELoss()
            loss = criterion(outputs.squeeze(), batch_H)
            
            total_error += torch.sum(torch.abs(outputs - batch_H)).item()
            
            print(f'Batch {i+1}, Accumulated error: {total_error:.4f}, N_correct {n_correct}')
    

    acc = n_correct / n_samples
    avg_error = total_error / len(test_loader.dataset)
    print(f'Accuracy: {acc * 100:.2f}%, Average error: {avg_error}')

if __name__ == "__main__":
    test('C:\\Users\\brend\\OneDrive\\Desktop\\HamiltonianBNN\\data\\test_data.csv')
