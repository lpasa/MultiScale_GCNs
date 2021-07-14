from torch.nn import Linear
from torch.nn import functional as F
from torch.nn import init
import math
from torch.nn.init import xavier_uniform_

class Linear_masked_weight(Linear):

    def forward(self, input, mask):
        #maskedW=(self.weight.t() * mask).t()
        maskedW=self.weight*mask
        return F.linear(input, maskedW, self.bias)

    # def reset_parameters(self):
    #     xavier_uniform_(self.weight)
    #     if self.bias is not None:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         init.uniform_(self.bias, -bound, bound)