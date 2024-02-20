
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

import torch.fft as afft

class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.
    Args:
        output_dim: output dimension for compact bilinear pooling.
        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.
        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.
        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.
        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.
        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, cuda=True,
                 rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sparse_sketch_matrix1 = Variable(self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim))

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        self.sparse_sketch_matrix2 = Variable(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim))

        if cuda:
            self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.cuda()
            self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.cuda()

    def forward(self, bottom1, bottom2):
        """
        bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width].
        """
        
        assert bottom1.size(1) == self.input_dim1 and \
            bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)

        fft1 = afft.fft(sketch_1)
        fft2 = afft.fft(sketch_2)

        fft_product = fft1 * fft2

        cbp_flat = afft.ifft(fft_product).real

        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert(rand_h.ndim == 1 and rand_s.ndim ==
               1 and len(rand_h) == len(rand_s))
        assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()


class MCB_baseline(nn.Module):
    def __init__(self, im_feat_size, text_feat_size, mcb_out=1000, mcb_factor=5, dropout=0.3):
        super(MCB_baseline, self).__init__()

        self.MCB_output_dim = mcb_out * mcb_factor
        self.im_feat_size = im_feat_size
        self.text_feat_size = text_feat_size
        self.mcb_out = mcb_out
        self.mcb_factor = mcb_factor

        # self.pool2d = AvgPool2d(global_avg_pool_size, stride=1)
        self.Dropout = nn.Dropout(p=dropout, )
        self.mcb = CompactBilinearPooling(self.im_feat_size, self.text_feat_size, self.MCB_output_dim)

    def forward(self, text_feat, img_feat):
        # pooling image features

        # img_feat_resh = img_feat.permute(0, 3, 1, 2)        # N x w x w x C -> N x C x w x w
        # img_feat_pooled = self.pool2d(img_feat_resh)        # N x C x 1 x 1
        # img_feat_sq = img_feat_pooled.squeeze()             # N x C

        # N x C -> N x C x 1 x 1
        img_feat = img_feat.unsqueeze(-1)
        img_feat = img_feat.unsqueeze(-1)

        text_feat = text_feat.unsqueeze(-1)
        text_feat = text_feat.unsqueeze(-1)

        # # ques_embed                                         N x T x embedding_size
        # ques_embed_resh = ques_embed.permute(1, 0, 2)       #T x N x embedding_size
        # lstm_out, (hn, cn) = self.LSTM(ques_embed_resh)
        # ques_lstm = lstm_out[-1]                            # N x lstm_units
        # ques_lstm = self.Dropout(ques_lstm)

        iq_feat = self.mcb(img_feat, text_feat)  # N x 5000
        iq_feat = self.Dropout(iq_feat)

        iq_resh = iq_feat.view(-1, 1, self.mcb_out, self.mcb_factor)  # N x 1 x 1000 x 5
        iq_sumpool = torch.sum(iq_resh, 3)  # N x 1 x 1000 x 1
        iq_sumpool = torch.squeeze(iq_sumpool)  # N x 1000

        iq_sqrt = torch.sqrt(F.relu(iq_sumpool)) - torch.sqrt(F.relu(-iq_sumpool))
        iq_norm = F.normalize(iq_sqrt)

        return iq_norm


class MFB_baseline(nn.Module):
    def __init__(self, im_feat_size, text_feat_size,
                 mfb_out=1000, mfb_factor=5, dropout=0.3):
        super(MFB_baseline, self).__init__()

        self.mfb_output_dim = mfb_out*mfb_factor
        self.im_feat_size = im_feat_size
        self.text_feat_size = text_feat_size
        self.mfb_out = mfb_out
        self.mfb_factor = mfb_factor

        # self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=LSTM_units,
        #                     num_layers=LSTM_layers, batch_first=False)
        # self.pool2d = AvgPool2d(global_avg_pool_size, stride=1)
        self.Dropout = nn.Dropout(p=dropout, )
        self.Linear_img_proj = nn.Linear(im_feat_size, self.mfb_output_dim)
        self.Linear_ques_proj = nn.Linear(im_feat_size, self.mfb_output_dim)
        # self.Linear_predict = nn.Linear(self.mfb_out, ans_vocab_size)
        # self.Softmax = nn.Softmax()

    def forward(self, text_feat, img_feat):
        ques_feat = self.Linear_ques_proj(text_feat)  # N x 5000
        img_feat = self.Linear_img_proj(img_feat)  # N x 5000

        iq_feat = torch.mul(ques_feat, img_feat)  # N x 5000
        iq_feat = self.Dropout(iq_feat)

        iq_resh = iq_feat.view(-1, 1, self.mfb_out, self.mfb_factor)  # N x 1 x 1000 x 5
        iq_sumpool = torch.sum(iq_resh, 3)  # N x 1 x 1000 x 1
        iq_sumpool = torch.squeeze(iq_sumpool)  # N x 1000

        iq_sqrt = torch.sqrt(F.relu(iq_sumpool)) - torch.sqrt(F.relu(-iq_sumpool))
        iq_norm = F.normalize(iq_sqrt)

        # pred = self.Linear_predict(iq_norm)
        # pred = self.Softmax(pred)

        return iq_norm


class MutanFusion(nn.Module):
    def __init__(self, input_dim=1024, out_dim=5000, num_layers=5):
        super(MutanFusion, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)

            hv.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.image_transformation_layers = nn.ModuleList(hv)
        #
        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.ques_transformation_layers = nn.ModuleList(hq)

    def forward(self, ques_emb, img_emb):
        # Pdb().set_trace()
        batch_size = img_emb.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)

            x_hq = ques_emb
            x_hq = self.ques_transformation_layers[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))
        #
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        x_mm = F.tanh(x_mm)
        return x_mm


class MUTAN_baseline(nn.Module):
    def __init__(self, im_feat_size, text_feat_size, mutan_out=1000, mutan_factor=5,
                 dropout=0.3):
        super(MUTAN_baseline, self).__init__()


        self.im_feat_size = im_feat_size
        self.text_feat_size = text_feat_size
        self.mutan_out = mutan_out

        self.mutan = MutanFusion(text_feat_size, self.mutan_out, mutan_factor)
        # self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=LSTM_units,
        #                     num_layers=LSTM_layers, batch_first=False)
        # self.pool2d = AvgPool2d(global_avg_pool_size, stride=1)
        self.Dropout = nn.Dropout(p=dropout, )
        # self.Linear_predict = nn.Linear(self.mutan_out, ans_vocab_size)
        # self.Softmax = nn.Softmax()

    def forward(self, text_feat, img_feat):
        # # pooling image features
        # img_feat_resh = img_feat.permute(0, 3, 1, 2)        # N x w x w x C -> N x C x w x w
        # img_feat_pooled = self.pool2d(img_feat_resh)        # N x C x 1 x 1
        # img_feat_sq = img_feat_pooled.squeeze()             # N x C

        # # ques_embed                                         N x T x embedding_size
        # ques_embed_resh = ques_embed.permute(1, 0, 2)       #T x N x embedding_size
        # lstm_out, (hn, cn) = self.LSTM(ques_embed_resh)
        # ques_lstm = lstm_out[-1]                            # N x lstm_units
        # ques_lstm = self.Dropout(ques_lstm)

        iq_feat = self.mutan(img_feat, text_feat)

        iq_sqrt = torch.sqrt(F.relu(iq_feat)) - torch.sqrt(F.relu(-iq_feat))
        iq_norm = F.normalize(iq_sqrt)

        # pred = self.Linear_predict(iq_norm)
        # pred = self.Softmax(pred)

        return iq_norm