from torch.nn import Linear
from torch.nn import functional as F
from torch.nn import init
import math
from torch.nn.init import xavier_uniform_

class Linear_masked_weight(Linear):

    def forward(self, input, mask):
        maskedW=self.weight*mask
        return F.linear(input, maskedW, self.bias)
