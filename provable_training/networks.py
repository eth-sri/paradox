import torch
import torch.nn as nn

"""
    A set of commonly used network architectures built on top of 
    network classes that encapsulate torch layers
"""


def create_net(net_type, input_shape, device, load_model=None, use_relu6=False, normalize=True, dataset=None):
    # An ugly monkeypatch for pre-Flatten torch versions (needed for loss landscape)
    try:
        dummy_flatten = nn.Flatten()
    except AttributeError:
        print(f'Flatten does not exist in torch version {torch.__version__}, making our own')
        class Flatten(nn.Module):
            def forward(self, input):
                return input.view(input.size(0), -1)
        nn.Flatten = Flatten

    # Create a net based on the name
    if net_type == 'dm-small' or net_type == 'CONVPLUS':
        convs = [(16, 4, 2, 0), (32, 4, 1, 0)]
        fcs = [100, 10]
        net = Conv(device, input_shape, convs, fcs, use_relu6=use_relu6, normalize=normalize, dataset=dataset).to(device)
    elif net_type == 'dm-medium':
        convs = [(32, 3, 1, 0), (32, 4, 2, 0), (64, 3, 1, 0), (64, 4, 2, 0)]
        fcs = [512, 512, 10]
        net = Conv(device, input_shape, convs, fcs, use_relu6=use_relu6, normalize=normalize, dataset=dataset).to(device)
    elif net_type == 'dm-large':
        convs = [(64, 3, 1, 1), (64, 3, 1, 1), (128, 3, 2, 1), (128, 3, 1, 1), (128, 3, 1, 1)]
        fcs = [512, 10]
        net = Conv(device, input_shape, convs, fcs, use_relu6=use_relu6, normalize=normalize, dataset=dataset).to(device)
    elif net_type == 'fc1':
        net = FullyConnected(device, input_shape, [100, 10], use_relu6=use_relu6).to(device)
    elif net_type == 'fc2':
        net = FullyConnected(device, input_shape, [50, 50, 10], use_relu6=use_relu6).to(device)
    elif net_type == 'fc3' or net_type == 'FC-SMALL':
        net = FullyConnected(device, input_shape, [100, 100, 10], use_relu6=use_relu6, dataset=dataset).to(device)
    elif net_type == 'fc4':
        net = FullyConnected(device, input_shape, [100, 100, 100, 10], use_relu6=use_relu6).to(device)
    elif net_type == 'fc5' or net_type == 'FC':
        net = FullyConnected(device, input_shape, [400, 200, 100, 100, 10], use_relu6=use_relu6, normalize=normalize, dataset=dataset).to(device)
    elif net_type == 'conv0' or net_type == 'CONV':
        net = Conv(device, input_shape, [(16, 4, 2, 1)], [100, 10], 10, use_relu6=use_relu6, normalize=normalize, dataset=dataset).to(device)
    else:
        raise RuntimeError(f'Unknown net: {net_type}')

    # Load state dict
    if load_model is not None:
        state_dict = torch.load(load_model, map_location=device)

        # hack: register_buffer requires these to be present
        if type(net.layers[0]) == Normalization:
            state_dict['layers.0.mean'] = net.layers[0].mean 
            state_dict['layers.0.sigma'] = net.layers[0].sigma 

        net.load_state_dict(state_dict)

    return net


class Normalization(nn.Module):

    def __init__(self, device, dataset):
        super(Normalization, self).__init__()
        # Register buffer so .to(device) works
        # Note: make sure to remove layers.0.mean and layers.0.sigma from old model files
        
        if dataset == 'mnist':
            self.register_buffer('mean', torch.FloatTensor([0.1307]))  
            self.register_buffer('sigma', torch.FloatTensor([0.3081])) 
        elif dataset == 'fashion-mnist':
            self.register_buffer('mean', torch.FloatTensor([0.2861]))  
            self.register_buffer('sigma', torch.FloatTensor([0.3530])) 
        elif dataset == 'cifar10':
            self.register_buffer('mean', torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1))) 
            self.register_buffer('sigma', torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1)))
        elif dataset == 'svhn':
            self.register_buffer('mean', torch.FloatTensor([0.4377, 0.4438, 0.4728]).view((1, 3, 1, 1))) 
            self.register_buffer('sigma', torch.FloatTensor([0.1980, 0.2011, 0.1970]).view((1, 3, 1, 1)))
        else:
            raise RuntimeError(f'Invalid dataset supplied to Normalization: {dataset}')
        self.mean.to(device)
        self.sigma.to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class FullyConnected(nn.Module):

    def __init__(self, device, input_shape, fc_layers, normalize=True, dataset=None, use_relu6=False, linear=False):
        super(FullyConnected, self).__init__()
        self.skip_norm = not normalize

        self.use_relu6 = use_relu6
        if self.use_relu6:
            print('(!) Using ReLU6 in FC!')

        layers = []
        if normalize:
            layers.append(Normalization(device, dataset))
        layers.append(nn.Flatten())

        prev_fc_size = input_shape.prod().item()
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers) and not linear:
                if self.use_relu6:
                    layers += [nn.ReLU6()]
                else:
                    layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            if self.skip_norm and isinstance(layer, Normalization):
                continue
            x = layer(x)
        return x

    def printnet(self, f):
        print('Normalize mean=[0.1307] std=[0.3081]', file=f)
        for layer in self.layers[1:]:
            print(layer)
            if isinstance(layer, nn.Linear):
                if layer.out_features == 10:
                    print('Affine', file=f)
                else:
                    print('ReLU', file=f)
            elif isinstance(layer, nn.ReLU):
                pass
            elif isinstance(layer, nn.ReLU6):
                pass
            elif isinstance(layer, nn.Flatten):
                pass
            else:
                assert False
        exit(0)


class Conv(nn.Module):

    def __init__(self, device, input_shape, conv_layers, fc_layers, n_class=10, normalize=True, dataset=None, use_relu6=False):
        super(Conv, self).__init__()

        self.use_relu6 = use_relu6

        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]

        self.input_size = input_shape[1]
        self.n_class = n_class

        layers = []
        if normalize:
            layers.append(Normalization(device, dataset))
            
        prev_channels = input_shape[0]
        img_dim = self.input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding)
            ]
            if self.use_relu6:
                layers += [nn.ReLU6()]
            else:
                layers += [nn.ReLU()]
            prev_channels = n_channels
            img_dim = (img_dim - kernel_size + 2 * padding) // stride + 1
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                if self.use_relu6:
                    layers += [nn.ReLU6()]
                else:
                    layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def printnet(self, f):
        print('Normalize mean=[0.1307] std=[0.3081]', file=f)
        dim, in_channels = 28, 1
        for layer in self.layers[1:]:
            print(layer)
            if isinstance(layer, nn.Linear):
                if layer.out_features == 10:
                    print('Affine', file=f)
                else:
                    print('ReLU', file=f)
                #print(h.printListsNumpy(list(layer.weight.data)), file=f)
                #print(h.printNumpy(layer.bias), file=f)
            elif isinstance(layer, nn.ReLU):
                pass
            elif isinstance(layer, nn.ReLU6):
                pass
            elif isinstance(layer, nn.Flatten):
                pass
            elif isinstance(layer, nn.Conv2d):
                print("Conv2D", file = f)
                sz = [dim, dim, in_channels]
                print('ReLU' + ", filters={}, kernel_size={}, input_shape={}, stride={}, padding={}".format(
                    layer.out_channels, [layer.kernel_size[0], layer.kernel_size[1]], sz, [layer.stride[0], layer.stride[1]], layer.padding[0]), file=f)
                dim = dim // layer.stride[0]
                in_channels = layer.out_channels
            else:
                assert False
        exit(0)
