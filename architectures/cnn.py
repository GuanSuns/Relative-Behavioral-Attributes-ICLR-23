import torch.nn as nn
import torch


class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batch_norm=True, non_linearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batch_norm is True:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features), non_linearity)
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features), non_linearity)

    def forward(self, x):
        return self.model(x)


class Conv_Layer(nn.Module):
    def __init__(self, n_in, n_out):
        super(Conv_Layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class Encoder(nn.Module):
    def __init__(self, dim, n_channel=1):
        super(Encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = Conv_Layer(n_channel, nf)
        # state size. (nf) x 32 x 32
        self.c2 = Conv_Layer(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = Conv_Layer(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = Conv_Layer(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
            nn.Conv2d(in_channels=nf * 8, out_channels=dim, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.Tanh()
        )

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class Upconv_Layer(nn.Module):
    def __init__(self, n_in, n_out):
        super(Upconv_Layer, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_in, out_channels=n_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    def __init__(self, dim, n_channel=1):
        super(Decoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=dim, out_channels=nf * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (nf*8) x 4 x 4
        self.upc2 = Upconv_Layer(nf * 8, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = Upconv_Layer(nf * 4, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = Upconv_Layer(nf * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf, out_channels=n_channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        d1 = self.upc1(x.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        return output
