import os
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    print('\nTorch Version: {}\n'.format(torch.__version__))
    
    x = torch.rand(5, 3)
    print(x)