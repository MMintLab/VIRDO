import sys
import modules
import torch
from torch import nn
from collections import OrderedDict



class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module, drop_out):
        '''
        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = modules.FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                 num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                                 outermost_linear=False, drop_out= drop_out, nonlinearity='relu')
            self.nets.append(hn)

            if 'weight' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))

            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)
        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


class virdo_hypernet(nn.Module):
    '''A canonical 2D representation hypernetwork mapping 2D coords to out_features.'''
    def __init__(self, in_features, out_features, hyper_in_features = 256, hl = 2, **kwargs):
        super().__init__()
        self.hl = hl
        for key, value in kwargs.items():
            self.__dict__[key] = value

        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='relu', mode='hypo',in_features= in_features,
                                             hidden_features=256, num_hidden_layers=self.hl, outermost_linear= False, drop_out = False)
            

        self.hyper_net = HyperNetwork(hyper_in_features=hyper_in_features, hyper_hidden_layers=0, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net, drop_out = False)



    def forward(self, model_input):
        if 'model_out' in model_input.keys():
            input = {'coords' : model_input['coords'], 'model_out' : model_input['model_out']}
        else:
            input = {'coords' : model_input['coords']}
        hypo_params = self.hyper_net(model_input['embedding'])
        output = self.hypo_net(input, params=hypo_params)

        return {'model_in':output['model_in'], 'model_out':output['model_out'], 'latent_vec':model_input['embedding'],
                'hypo_params':hypo_params}



############################
# Initialization schemes
def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2


    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2


    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)
