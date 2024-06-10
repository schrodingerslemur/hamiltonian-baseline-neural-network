import torch
import sys
sys.path.append('../')
from HamiltonianBNN.network.neural_net import NeuralNet # type: ignore

def finite(qx, qy, px, py, method, change=0.01):
    input = torch.tensor([[qx, qy, px, py]])
    if method == 'qx':
        delta_minus = torch.tensor([[qx-change, qy, px, py]])
        delta_plus = torch.tensor([[qx+change, qy, px, py]])
    elif method == 'qy':
        delta_minus = torch.tensor([[qx, qy-change, px, py]])
        delta_plus = torch.tensor([[qx, qy+change, px, py]])
    elif method == 'px':
        delta_minus = torch.tensor([[qx, qy, px-change, py]])
        delta_plus = torch.tensor([[qx, qy, px+change, py]])
    elif method == 'py':
        delta_minus = torch.tensor([[qx, qy, px, py-change]])
        delta_plus = torch.tensor([[qx, qy, px, py+change]])
    
    model = NeuralNet()
    output_minus = model(delta_minus)
    output_plus = model(delta_plus)

    derivative = (output_plus.item() - output_minus.item())/change

    print(f"del_H/del_{method} at (qx, qy, px, py): ({qx}, {qy}, {px}, {py}) is {derivative:.10f}")

if __name__ == '__main__':
    finite(1,1,1,1, 'qy')
