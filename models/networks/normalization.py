
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        
        nhidden = 128

        pw = ks // 2

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, fmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        #segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        #actv = self.mlp_shared(segmap)
        
        gamma = self.mlp_gamma(fmap)
        beta = self.mlp_beta(fmap)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADE2(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, no_instance=True):
        super().__init__()
        self.no_instance = no_instance
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        nhidden = 128

        pw = ks // 2
        

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        self.class_specified_affine = ClassAffine(label_nc, norm_nc)
        self.attention_affine = AttentionAffine(label_nc, norm_nc)

        if not no_instance:
            self.inst_conv = nn.Conv2d(1, 1, kernel_size=1, padding=0)

    def forward(self, x, fmap, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        gamma1 = self.mlp_gamma(fmap)
        beta1 = self.mlp_beta(fmap)

        # apply scale and bias
        out1 = normalized * (1 + gamma1) + beta1

        # Part 2. scale the segmentation mask
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        if not self.no_instance:
            inst_map = torch.unsqueeze(segmap[:,-1,:,:],1)
            segmap = segmap[:,:-1,:,:]

        # Part 3. class affine with noise
        out2 = self.class_specified_affine(normalized, segmap)

        if not self.no_instance:
            inst_feat = self.inst_conv(inst_map)
            out2 = torch.cat((out2, inst_feat), dim=1)

        attention_global, attention_local = self.attention_affine(segmap)

        out = out1 * attention_global + out2 * attention_local

        return out

class ClassAffine(nn.Module):
    def __init__(self, label_nc, affine_nc):
        super(ClassAffine, self).__init__()
        self.affine_nc = affine_nc
        self.label_nc = label_nc
        self.weight = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        self.bias = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def affine_embed(self, mask):
        arg_mask = torch.argmax(mask, 1).long() # [n, h, w]
        class_weight = F.embedding(arg_mask, self.weight).permute(0, 3, 1, 2) # [n, c, h, w]
        class_bias = F.embedding(arg_mask, self.bias).permute(0, 3, 1, 2) # [n, c, h, w]
        return class_weight, class_bias

    def forward(self, input, mask, input_dist=None):
        # class_weight, class_bias = self.affine_gather(input, mask)
        # class_weight, class_bias = self.affine_einsum(mask) 
        class_weight, class_bias = self.affine_embed(mask)
        x = input * class_weight + class_bias
        return x

class AttentionAffine(nn.Module):
    def __init__(self, label_nc, affine_nc):
        super(AttentionAffine, self).__init__()
        self.affine_nc = affine_nc
        self.label_nc = label_nc
        self.attention = nn.Parameter(torch.Tensor(self.label_nc, 2))
        self.softmax_ = torch.nn.Softmax(dim=1)

    def forward(self, mask):
        arg_mask = torch.argmax(mask, 1).long() # [n, h, w]
        att = F.embedding(arg_mask, self.attention).permute(0, 3, 1, 2) # [n, c, h, w]
        
        result_attention = self.softmax_(att)

        attention_global = result_attention[:, 0:1, :, :]
        attention_local = result_attention[:, 1:2, :, :]
        
        #attention_global = attention_local.repeat(1, self.affine_nc, 1, 1)
        #attention_local = attention_global.repeat(1, self.affine_nc, 1, 1)

        return attention_global, attention_local
