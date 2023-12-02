import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple


ExpertData = namedtuple('ExpertData', ('states', 'actions'))


class ExpertDataset(Dataset):
    def __init__(self, expert_data):
        self.states = expert_data.states
        self.actions = expert_data.actions
        
    def __len__(self):
        return self.states.size(0)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        
        return state, action
    
    def add_data(self, data):
        self.states = torch.cat([self.states, data.states], dim=0)
        self.actions = torch.cat([self.actions, data.actions], dim=0)


def get_dataloader(dataset, args):
    small_dset = dataset[:args.num_dataset_samples]
    small_states, small_actions = small_dset
    small_dset = ExpertDataset(ExpertData(small_states, small_actions))
    return DataLoader(small_dset, batch_size=args.batch_size, shuffle=True)
    