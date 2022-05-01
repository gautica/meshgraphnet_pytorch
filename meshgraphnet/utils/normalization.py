import torch
from torch import nn


class Normalizer(nn.Module):
    def __init__(self, size, name, max_accumulation=10**6, std_epsilon=1e-8):
        super(Normalizer, self).__init__()
        self.name = name
        self.max_accumulation = nn.Parameter(torch.tensor(max_accumulation), requires_grad=False)
        self.std_epsilon = nn.Parameter(torch.tensor(std_epsilon), requires_grad=False)

        self.register_buffer('acc_count', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('num_acc', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('acc_sum', torch.zeros(size, dtype=torch.float32))
        self.register_buffer('acc_sum_squared', torch.zeros(size, dtype=torch.float32))
        
    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate and self.num_acc < self.max_accumulation:
            self._accumulate(batched_data)
        return (batched_data - self._mean()) / self._std_with_eps()
    
    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_eps() + self._mean()
        
    def _accumulate(self, batched_data):
        batch_size, length_traj, _ = batched_data.shape
        batched_data = batched_data.reshape(batch_size*length_traj, -1)

        data_sum = torch.sum(batched_data, dim=0)
        data_sum_squared = torch.sum(batched_data**2, dim=0)
        self.acc_sum += data_sum
        self.acc_sum_squared += data_sum_squared
        self.acc_count += torch.tensor(batch_size*length_traj).to(self.acc_count.device)
        self.num_acc += torch.tensor(1.0).to(self.num_acc.device)
        
    def _mean(self):
        safe_count = torch.maximum(self.acc_count, torch.tensor(1.0).to(self.acc_count.device))
        return self.acc_sum / safe_count
    
    def _std_with_eps(self):
        safe_count = torch.maximum(self.acc_count, torch.tensor(1.0).to(self.acc_count.device))
        std = torch.sqrt(self.acc_sum_squared / safe_count - self._mean()**2)
        return torch.maximum(std, self.std_epsilon)
