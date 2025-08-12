import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, init_type='kaiming_normal'):
        super(BaseModel, self).__init__()
        self.init_type = init_type
    
    def init_weights(self):
        if self.init_type == 'gaussian_random':
            self._init_gaussian_random()
        elif self.init_type == 'kaiming_normal':
            self._init_kaiming_normal()
        elif self.init_type == 'kaiming_uniform':
            self._init_kaiming_uniform()
        elif self.init_type == 'xavier_normal':
            self._init_xavier_normal()
        elif self.init_type == 'xavier_uniform':
            self._init_xavier_uniform()
        elif self.init_type == 'orthogonal':
            self._init_orthogonal()
        else:
            raise NotImplementedError(f'Initialization {self.init_type} is not implemented.')

    def _init_gaussian_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _init_kaiming_normal(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _init_kaiming_uniform(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _init_xavier_normal(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _init_xavier_uniform(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _init_orthogonal(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
