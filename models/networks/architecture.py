
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.normalization import SPADE2


class Unet(nn.Module):
    def __init__(self, fin, opt):
        super().__init__()
        self.opt=opt
        fmiddle=128
        fup=256
        norm_layer = get_nonspade_norm_layer(opt, 'spectralsync_batch')
        self.down_conv_00 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_01 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_10 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_11 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_20 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_21 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_30 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_31 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_40 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_41 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        if opt.num_upsampling_layers == 'more' or opt.num_upsampling_layers == 'most':
            self.down_conv_50 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
            self.down_conv_51 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
            if opt.num_upsampling_layers == 'most':
                self.down_conv_60 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
                self.down_conv_61 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

        self.up_conv_00 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_01 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_10 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_11 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        if opt.num_upsampling_layers == 'more' or opt.num_upsampling_layers == 'most':
            self.up_conv_20 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        else:
            self.up_conv_20 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_21 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_30 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_31 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_40 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_41 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_50 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_51 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_60 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_61 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        if opt.num_upsampling_layers == 'most':
            self.up_conv_70 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
            self.up_conv_71 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        

    def forward(self, seg):
        concat=[]
        #result=[]
        down0=self.down_conv_01(self.down_conv_00(seg))
        concat.append(down0)
        down1=self.down_conv_11(self.down_conv_10(self.down(down0)))
        concat.append(down1)
        down2=self.down_conv_21(self.down_conv_20(self.down(down1)))
        concat.append(down2)
        down3=self.down_conv_31(self.down_conv_30(self.down(down2)))
        concat.append(down3)
        down4=self.down_conv_41(self.down_conv_40(self.down(down3)))
        concat.append(down4)
        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            down5=self.down_conv_51(self.down_conv_50(self.down(down4)))
            concat.append(down5)
            if self.opt.num_upsampling_layers == 'most':
                down6=self.down_conv_61(self.down_conv_60(self.down(down5)))
                concat.append(down6)
        i=-1
        up0=self.up_conv_00(self.down(concat[i]))
        up1=self.up_conv_01(up0)
        up2=self.up_conv_10(torch.cat([self.up(up1),concat[i]],dim=1))
        i=i-1
        up3=self.up_conv_11(up2)
        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            up4=self.up_conv_20(torch.cat([self.up(up3),concat[i]],dim=1))
            i=i-1
        else:
            up4=self.up_conv_20(up3)
        up5=self.up_conv_21(up4)
        up6=self.up_conv_30(torch.cat([self.up(up5),concat[i]],dim=1))
        i=i-1
        up7=self.up_conv_31(up6)
        up8=self.up_conv_40(torch.cat([self.up(up7),concat[i]],dim=1))
        i=i-1
        up9=self.up_conv_41(up8)
        up10=self.up_conv_50(torch.cat([self.up(up9),concat[i]],dim=1))
        i=i-1
        up11=self.up_conv_51(up10)
        up12=self.up_conv_60(torch.cat([self.up(up11),concat[i]],dim=1))
        i=i-1
        up13=self.up_conv_61(up12)
        if self.opt.num_upsampling_layers == 'most':
            up14=self.up_conv_70(torch.cat([self.up(up13),concat[i]],dim=1))
            i=i-1
            up15=self.up_conv_71(up14)

        if self.opt.num_upsampling_layers == 'most':
            return up0,up1,up2,up3,up4,up5,up6,up7,up8,up9,up10,up11,up12,up13,up14,up15

        return up0,up1,up2,up3,up4,up5,up6,up7,up8,up9,up10,up11,up12,up13

class Unet_D(nn.Module):
    def __init__(self, fin, opt):
        super().__init__()
        self.opt=opt
        fmiddle=64
        fup=128
        norm_layer = get_nonspade_norm_layer(opt, 'spectralsync_batch')
        self.down_conv_0 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_1 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_2 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_3 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.down_conv_4 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

        self.up_conv_0 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.up_conv_1 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.to_result0 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=4, padding=2)), nn.LeakyReLU(0.2, True))
        self.up_conv_2 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.to_result1 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=4, padding=2)), nn.LeakyReLU(0.2, True))
        self.up_conv_3 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.to_result2 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=4, padding=2)), nn.LeakyReLU(0.2, True))
        self.up_conv_4 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        self.to_result4 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=4, padding=2)), nn.LeakyReLU(0.2, True))
        self.up_conv_5 = nn.Sequential(norm_layer(nn.Conv2d(fup, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))
        #self.to_result3 = nn.Sequential(norm_layer(nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)), nn.LeakyReLU(0.2, True))

    def forward(self, seg):
        concat=[]
        #result=[]
        down0=self.down_conv_0(seg)
        concat.append(down0)
        down1=self.down_conv_1(self.down(down0))
        concat.append(down1)
        down2=self.down_conv_2(self.down(down1))
        concat.append(down2)
        down3=self.down_conv_3(self.down(down2))
        concat.append(down3)
        down4=self.down_conv_4(self.down(down3))
        concat.append(down4)
        i=-1
        up0=self.up_conv_0(self.down(concat[i]))
        up1=self.up_conv_1(torch.cat([self.up(up0),concat[i]],dim=1))
        i=i-1
        result0=self.to_result0(up1)
        up2=self.up_conv_2(torch.cat([self.up(up1),concat[i]],dim=1))
        i=i-1
        result1=self.to_result1(up2)
        up3=self.up_conv_3(torch.cat([self.up(up2),concat[i]],dim=1))
        i=i-1
        result2=self.to_result2(up3)
        up4=self.up_conv_4(torch.cat([self.up(up3),concat[i]],dim=1))
        i=i-1
        result3=up4
        result4=self.to_result4(up4)
        up5=self.up_conv_5(torch.cat([self.up(up4),concat[i]],dim=1))
        i=i-1
        result5=up5

        return result0,result1,result2,result3,result4,result5



class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE2(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = SPADE2(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE2(spade_config_str, fin, opt.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, fmap0, fmap1, seg):
        x_s = self.shortcut(x, fmap0, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, fmap0, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, fmap1, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, fmap, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, fmap, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
