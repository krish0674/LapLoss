import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from utils.models.archs.LPTN_paper_arch import LPTNPaper


net = LPTNPaper(nrb_low=4,
                nrb_high=4,
                num_high=2,
                device='cpu')

# Using list of tuples for multiple inputs
def prepare_input(resolution):
    x1 = torch.FloatTensor(1, *resolution)
    x2 = torch.FloatTensor(1, *resolution)
    return dict(real_A_full=x1, LR_thermal=x2)
input_shapes = (1, 224, 224)

macs, params = get_model_complexity_info(net, input_shapes, as_strings=True,
                                            print_per_layer_stat=True, verbose=True, input_constructor=prepare_input)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
