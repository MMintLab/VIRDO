import sys, os
from torchmeta.modules.module import MetaModule
from torchmeta.modules.container import MetaSequential
from torchmeta.modules.utils import get_subdict
import numpy as np
from torch import nn
from collections import OrderedDict
import math
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import diff_operators
    

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        batchsize = x.size()[0]
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float))).view(1,9).repeat(batchsize,1)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 64, 1)
        

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STN3d(k=64)


        
    def forward(self, x, epoch):
      
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans).float()
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None                
        pointfeat = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.mean(x, 2, keepdim=True)[0]
        x = x.view(-1, 64)
        return x
    

    
class force_mlp(nn.Module):
    def __init__(self, d_cnt_code , d_force_emb ):
        super(force_mlp, self).__init__()
        self.fc1 = nn.Linear(64, d_cnt_code)
        self.fc2 = nn.Linear(d_cnt_code+3,  d_force_emb)
       
        self.relu = nn.ReLU()
    def forward(self, ft, contact_force):
        ft = self.relu(self.fc1(ft))
        x = torch.cat([ft, contact_force], axis=1).float()
        x = self.fc2(x)
        return x, ft

    def forward_infer(self, ft, contact_force):
        x = torch.cat([ft, contact_force], axis=1).float()
        x = self.fc2(x)
        return x, ft
     

class PointNetCls(nn.Module):
    def __init__(self, d_cnt_code , d_force_emb ):
        super(PointNetCls, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.force_mlp = force_mlp(d_cnt_code , d_force_emb )
        print(self)
                
#         Xavier_init(self)

    def forward(self, x, contact_force, epoch):
        x  = self.feat(x, epoch)
        x, self.cnt_ft  = self.force_mlp(x, contact_force)
        return x
    
    def forward_infer(self, x, contact_force):
        x = self.force_mlp.forward_infer(x, contact_force)
        return x
 

class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__


    def forward(self, input, params=None, layer_num=None):
        if params is None:
            print("BatchLinear params activated")
            params = OrderedDict(self.named_parameters())
        if layer_num != None:
            key_init = 'net.'+str(layer_num)+'.0.'
            bias = params.get(key_init+'bias', None)
            weight = params[key_init+'weight']

            output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
            output += bias.unsqueeze(-2)
        else:
            bias = params.get('bias', None)
            weight = params['weight']

            output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
           
            if bias.size() == output.size():
                output += bias
            else:
                output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear, nonlinearity='relu', weight_init=None, drop_out = False, **kwargs):
        super().__init__()

        self.latent_in = False
        self.first_layer_init = None
        for key, value in kwargs.items():
            self.__dict__[key] = value


        nls_and_inits = {'relu':(nn.ReLU(inplace=True), init_weights_normal, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]
        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        # single layer
        self.net = []
        self.net.append(MetaSequential(BatchLinear(in_features, hidden_features), nl))
        
        for i in range(num_hidden_layers):
            if i != self.latent_in - 1:
                self.net.append(MetaSequential(
                    BatchLinear(hidden_features, hidden_features), nl))
            else:
                self.net.append(MetaSequential(
                    BatchLinear(hidden_features + in_features, hidden_features), nl))

        if outermost_linear:
            self.net.append(
                MetaSequential(BatchLinear(hidden_features, out_features), nl))
        else:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net[0].apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)


    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output


    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations



class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode=None, hidden_features=512, num_hidden_layers=5,outermost_linear=False, drop_out= False, input_encoder = None, **kwargs):
        super().__init__()
        self.mode = mode
        self.num_hidden_layers = num_hidden_layers
        self.outermost_linear = outermost_linear
        self.drop_out = drop_out
        self.input_encoder = input_encoder

        itm = 0
        self.latent_in = False
        for key, value in kwargs.items():
            self.__dict__[key] = value
            itm  = itm +1

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear= self.outermost_linear, nonlinearity=type, latent_in = self.latent_in,
                           drop_out = self.drop_out)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())


        ## HypoNetwork
        if 'model_out' in model_input:
            coords_org = model_input['coords']
            input = model_input['model_out']
        else:
            coords_org = model_input['coords'].clone().detach().requires_grad_(True)
            input = coords_org


        for layer in range(self.num_hidden_layers + 2):
            if layer == 0:
                x = self.net.net[0][0](input, params= get_subdict(params,'net'), layer_num= layer)
                try:
                    x = self.net.net[0][1](x)
                except:
                    pass
                if self.drop_out:
                    x = self.net.net[0][2](x)

            elif layer == self.num_hidden_layers + 1:

                x = self.net.net[layer][0](x, params= get_subdict(params,'net'), layer_num= layer)

                if self.outermost_linear:
                    x = self.net.net[layer][1](x)
                    if self.drop_out:
                        x = self.net.net[layer][2](x)
            else:
                x = self.net.net[layer][0](x, params=get_subdict(params, 'net'), layer_num=layer)
                x = self.net.net[layer][1](x)
                if self.drop_out:
                    x = self.net.net[layer][2](x)
                      
        return {'model_in': coords_org, 'model_out': x}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}




















########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


###################
# Complex operators
def compl_conj(x):
    y = x.clone()
    y[..., 1::2] = -1 * y[..., 1::2]
    return y


def compl_div(x, y):
    ''' x / y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = (a * c + b * d) / (c ** 2 + d ** 2)
    outi = (b * c - a * d) / (c ** 2 + d ** 2)
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


def compl_mul(x, y):
    '''  x * y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = a * c - b * d
    outi = (a + b) * (c + d) - a * c - b * d
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out
