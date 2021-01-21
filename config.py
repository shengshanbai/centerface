from easydict import EasyDict
import torch

g_config=EasyDict()
g_config.input_res=800
g_config.output_res=200
g_config.dev43ice=torch.device('cuda' if torch.cuda.is_available() else 'cpu')