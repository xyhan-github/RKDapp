import re
import torch
import torch.nn as nn

from network_app import Network
from collections import OrderedDict

def replace(string, substitutions):

    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)


def load_model(obj, prefix=''):
    net = obj.net
    if prefix == '':
        model = Network().construct(net, obj).to(obj.device)
    else:
        model = Network().construct(net, obj, prefix=prefix).to(obj.device)
        
    print("Loading checkpoint:")
    print(obj.checkpoint)

    state_dict = torch.load(obj.checkpoint, map_location=lambda storage, loc: storage)

    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    if hasattr(obj, 'double'):
        if obj.double:
            model = model.double()
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    if hasattr(obj,'cuda'):
        if obj.cuda:
            model = model.cuda()

    if hasattr(obj,'mode'):
        if obj.mode == 'eval':
            model.eval()
        elif obj.mode == 'train':
            model.train()
        else:
            raise Exception
    else:
        model.eval()
    
    return model

