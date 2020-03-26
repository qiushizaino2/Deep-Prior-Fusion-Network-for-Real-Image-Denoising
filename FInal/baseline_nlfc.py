
import torch 

from torch.nn import init

import torch.nn as nn 

import torch.nn.functional as F 

from torch.autograd import Variable 


def _make_layer(channels, kernelsize=3,stride=1,padding=1, block_num=1):
    # shortcut = nn.Sequential(
        # nn.Conv2d(in_channel, out_channel, 1, stride),
        # nn.BatchNorm2d(out_channel))
    layers = list()
    layers.append(ResBlock(channels, kernelsize,stride,padding))
        
    for i in range(1, block_num):
        layers.append(ResBlock(channels, kernelsize,stride,padding))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, chls,ks=3, strd=1,pad=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.PReLU(chls),
            nn.Conv2d(chls, chls,(ks,ks),stride=strd,padding=pad),
            nn.PReLU(chls),
            nn.Conv2d(chls, chls,(ks,ks),stride=strd,padding=pad)
        )

    def forward(self, x):
        out = self.left(x)
        residual = x  
        out += residual
        return out


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.EncConv1_P=nn.PReLU(6)
        self.EncConv1 = nn.Conv2d(6, 512,(1, 1),stride=1,padding=0)  
        self.Encblock1 = _make_layer(512,3,1,1)  

        self.EncConv2_P=nn.PReLU(512+6)
        self.EncConv2 = nn.Conv2d(512+6, 512, (1, 1),stride=1,padding=0)  
        self.Encblock2 = _make_layer(512,3,1,1)  

        self.EncConv3_P=nn.PReLU(512+6)
        self.EncConv3 = nn.Conv2d(512+6, 512, (1, 1),stride=1,padding=0)  
        self.Encblock3 =_make_layer(512,3,1,1)  
        
        self.EncConv4_P=nn.PReLU(512+6)
        self.EncConv4 = nn.Conv2d(512+6, 512, (1, 1),stride=1,padding=0)  
        self.Encblock4 =_make_layer(512,3,1,1)  

        self.EncConv5_P=nn.PReLU(512+6)
        self.EncConv5 = nn.Conv2d(512+6, 512, (1, 1),stride=1,padding=0)  
        self.Encblock5 =_make_layer(512,3,1,1)  

        self.DeConv4_P=nn.PReLU(512)
        self.DeConv4 = nn.ConvTranspose2d(512, 512, (3, 3), stride=2, padding=1,output_padding=1)
        self.Decblock4 = _make_layer(512,3,1,1)  
        
        self.DeConv3_P=nn.PReLU(512+6)
        self.DeConv3 = nn.ConvTranspose2d(512+6, 512, (3, 3), stride=2, padding=1,output_padding=1)
        self.Decblock3 =_make_layer(512,3,1,1)  

        self.DeConv2_P=nn.PReLU(512+6)
        self.DeConv2 = nn.ConvTranspose2d(512+6, 512, (3, 3), stride=2, padding=1,output_padding=1)
        self.Decblock2 =_make_layer(512,3,1,1)  

        self.DeConv1_P=nn.PReLU(512+6)
        self.DeConv1 = nn.ConvTranspose2d(512+6, 512, (3, 3), stride=2, padding=1,output_padding=1)
        self.Decblock1 =_make_layer(512,3,1,1)  
    
        self.DecConv_P=nn.PReLU(512)
        self.DecConv = nn.Conv2d(512, 4, (3, 3),stride=1,padding=1)  


    def forward(self, x):
        xo=x 
        x0=x[:,0:4,:,:]
        x=self.EncConv1_P(x)
        x=self.EncConv1(x)
        x1=self.Encblock1(x)

        x=F.max_pool2d(x1, (2, 2))
        x=torch.cat([x, F.avg_pool2d(xo, (2, 2))], dim=1) 
        x=self.EncConv2_P(x)
        x=self.EncConv2(x)
        x2=self.Encblock2(x)

        
        x=F.max_pool2d(x2, (2, 2))
        x=torch.cat([x, F.avg_pool2d(xo, (4, 4))], dim=1) 
        x=self.EncConv3_P(x)
        x=self.EncConv3(x)
        x3=self.Encblock3(x)

        x=F.max_pool2d(x3, (2, 2))
        x=torch.cat([x, F.avg_pool2d(xo, (8, 8))], dim=1) 
        x=self.EncConv4_P(x)
        x=self.EncConv4(x)
        x4=self.Encblock4(x)

        x=F.max_pool2d(x4, (2, 2))
        x=torch.cat([x, F.avg_pool2d(xo, (16, 16))], dim=1) 
        x=self.EncConv5_P(x)
        x=self.EncConv5(x)
        x5=self.Encblock5(x)

        x=self.DeConv4_P(x5)
        x=self.DeConv4(x)
        x=x+x4
        x=self.Decblock4(x)

        x=torch.cat([x, F.avg_pool2d(xo, (8, 8))], dim=1) 
        x=self.DeConv3_P(x)
        x=self.DeConv3(x)
        x=x+x3
        x=self.Decblock3(x)

        x=torch.cat([x, F.avg_pool2d(xo, (4, 4))], dim=1) 
        x=self.DeConv2_P(x)
        x=self.DeConv2(x)
        x=x+x2
        x=self.Decblock2(x)

        x=torch.cat([x, F.avg_pool2d(xo, (2, 2))], dim=1) 
        x=self.DeConv1_P(x)
        x=self.DeConv1(x)
        x=x+x1
        x=self.Decblock1(x)
        x=self.DecConv(x)
        x=x+x0

        return x

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net
 

def output_name_and_params(net):

    for name, parameters in net.named_parameters():

        print('name: {}, param: {}'.format(name, parameters))

 

if __name__ == '__main__':


    net = Baseline()

    init_net(net, init_type = 'xavier')

    # print('net: {}'.format(net))

    # params = net.parameters()   # generator object

    # print('params: {}'.format(params)) # params: <generator object Module.parameters at 0x0000025B0356B7C8>

    output_name_and_params(net)

    input_image = torch.zeros(1, 4, 128, 128)+1


 

    output = net(input_image)

    print('output: {}'.format(output))
    # print('output.size: {}'.format(output.size()))
