import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np




class FastTF(nn.Module):
    def __init__(self, seq_len, in_features, out_features, pred_len, seg_num_x, seg_num_y,
                 num_sampling, cut_freq, flinear_sparse_num, in_sparse_freq, group,
                 in_group_freq, flinear_individual, linear_individual):
        super(FastTF, self).__init__()

        # 赋值所有必要的属性
        self.seq_len = seq_len
        self.in_features = in_features
        self.out_features = out_features
        self.pred_len = pred_len
        self.seg_num_x = seg_num_x
        self.seg_num_y = seg_num_y
        self.num_sampling = num_sampling
        self.down_sampling = (self.seq_len // self.seg_num_x) // self.num_sampling
        self.cut_freq = cut_freq
        self.flinear_sparse_num = flinear_sparse_num
        self.in_sparse_freq = in_sparse_freq
        self.group = 1
        self.in_group_freq = 1
        self.flinear_individual = flinear_individual
        self.linear_individual = linear_individual
        self.enc_in = self.in_features


        if self.flinear_individual:
            self.flinear_weight = nn.ParameterList([
                nn.Parameter(torch.randn(self.flinear_sparse_num, self.in_sparse_freq, self.in_sparse_freq))
                for _ in range(self.enc_in)
            ])
        else:
            self.linear_sin_weight = nn.Parameter(
                torch.randn(self.seg_num_x, self.cut_freq))
            self.linear_cos_weight = nn.Parameter(
                torch.randn(self.seg_num_x, self.cut_freq))

        if self.linear_individual:
            self.linear_weight = nn.ParameterList([
                nn.Parameter(torch.randn(1, self.seg_num_y))
                for _ in range(self.enc_in)
            ])
        else:
            self.linear_sin_weight = nn.Parameter(
                torch.randn(self.seg_num_x, self.cut_freq))
            self.linear_cos_weight = nn.Parameter(
                torch.randn(self.seg_num_x, self.cut_freq))

    def forward(self, x):
        """
        x: [batch_size, seq_len, num_nodes, hidden_channels]
        """
        batch_size, seq_len, num_nodes, hidden_channels = x.shape
        # print(f"FastTF Input shape: {x.shape}, total elements: {x.numel()}")


        assert seq_len % self.seg_num_x == 0,


        x = x.reshape(batch_size * num_nodes * hidden_channels, self.seg_num_x, self.num_sampling, self.down_sampling)
        # print(f"After reshape for segmentation: {x.shape}, total elements: {x.numel()}")


        x = x.permute(0, 1, 3, 2)
        # print(f"After permute for FFT: {x.shape}, total elements: {x.numel()}")

        x = torch.fft.rfft(x, dim=3)
        # print(f"After rFFT: {x.shape}, total elements: {x.numel()}")


        x = x[:, :, :, :self.cut_freq]
        # print(f"After cutting frequency: {x.shape}, total elements: {x.numel()}")


        actual_cut_freq = x.size(-1)

        # SFM 处理
        if self.flinear_individual:
            x = x.reshape(batch_size, num_nodes, self.flinear_sparse_num, self.in_sparse_freq, self.down_sampling,
                          actual_cut_freq)
            x = torch.einsum('bncsft,cfet->bncsfe', x, self.flinear_weight) + x
        else:
            x_sin = x.imag
            x_cos = x.real

            linear_sin_weight = self.linear_sin_weight[:, :actual_cut_freq]
            linear_cos_weight = self.linear_cos_weight[:, :actual_cut_freq]


            tmp_sin = x_sin * linear_sin_weight.view(1, self.seg_num_x, 1, actual_cut_freq) + x_sin
            tmp_cos = x_cos * linear_cos_weight.view(1, self.seg_num_x, 1, actual_cut_freq) + x_cos
            x = torch.view_as_complex(torch.stack((tmp_cos, tmp_sin), dim=-1))

        x = x.contiguous()


        total_elements = x.numel()
        calculated_elements = batch_size * num_nodes * self.seg_num_x * self.down_sampling * actual_cut_freq

        if calculated_elements > 0:
            self.seg_num_y = total_elements // calculated_elements
            if total_elements % calculated_elements != 0:
                raise ValueError("Mismatch between total_elements and calculated_elements for reshaping.")
        else:
            raise ValueError("Calculated elements is zero. Please check the configuration parameters.")


        if self.seg_num_y <= 0:
            raise ValueError("seg_num_y calculated to be non-positive. Check your input parameters and configuration.")


        x = x.reshape(batch_size, num_nodes, self.seg_num_x, self.down_sampling, actual_cut_freq, self.seg_num_y)


        # print(f"After reshape: x.shape = {x.shape}, total elements: {x.numel()}")


        tmp2 = torch.zeros(
            [x.size(0), x.size(1), x.size(2), self.down_sampling, self.num_sampling // 2 + 1, self.seg_num_y],
            dtype=x.dtype, device=x.device
        )
        tmp2[:, :, :, :, :actual_cut_freq, :] = x

        y = tmp2.permute(0, 1, 2, 4, 3, 5)


        y = torch.fft.irfft(y, dim=5).permute(0, 1, 2, 3, 5, 4)

        total_elements = y.numel()
        elements_per_sample = num_nodes * hidden_channels
        remaining_length = total_elements // (batch_size * elements_per_sample)


        if total_elements % (batch_size * remaining_length * num_nodes * hidden_channels) != 0:

            valid_elements = batch_size * remaining_length * num_nodes * hidden_channels
            y = y.flatten()[:valid_elements].reshape(batch_size, remaining_length, num_nodes, hidden_channels)
        else:
            y = y.reshape(batch_size, remaining_length, num_nodes, hidden_channels)


        return y  # [batch_size, pred_len, num_nodes, hidden_channels]


class GGRU_FastTF(nn.Module):
    def __init__(self, config):
        super(GGRU_FastTF, self).__init__()


        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_nodes = config.model.num_nodes
        self.hidden_channels = config.model.hidden_channels
        self.in_features = config.model.in_features
        self.output_size = config.model.output_size
        self.dropout = config.model.dropout
        self.bidirectional = config.model.bidirectional



        self.gru = nn.GRU(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=True
        )


        self.seg_num_x = config.model.seg_num_x
        self.seg_num_y = config.model.seg_num_y
        self.num_sampling = config.model.num_sampling
        self.down_sampling = config.model.down_sampling
        self.cut_freq = config.model.cut_freq
        self.flinear_sparse_num = config.model.flinear_sparse_num
        self.in_sparse_freq = config.model.in_sparse_freq
        self.group = config.model.group
        self.in_group_freq = config.model.in_group_freq
        self.flinear_individual = config.model.flinear_individual
        self.linear_individual = config.model.linear_individual


        self.fast_tf = FastTF(
            seq_len=self.input_len,
            in_features=self.in_features,
            out_features=self.output_size,
            pred_len=self.output_len,
            seg_num_x=self.seg_num_x,
            seg_num_y=self.seg_num_y,
            num_sampling=self.num_sampling,
            cut_freq=self.cut_freq,
            flinear_sparse_num=self.flinear_sparse_num,
            in_sparse_freq=self.in_sparse_freq,
            group=self.group,
            in_group_freq=self.in_group_freq,
            flinear_individual=self.flinear_individual,
            linear_individual=self.linear_individual
        )


        self.gru = nn.GRU(
            input_size=self.in_features,
            hidden_size=self.hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout,
        )


        self.conv1 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)


        self.bn1 = nn.BatchNorm1d(self.hidden_channels)
        self.bn2 = nn.BatchNorm1d(self.hidden_channels)


        self.seq_len = config.input_len
        self.seg_num_x = config.model.seg_num_x
        self.num_sampling = config.model.num_sampling
        self.cut_freq = config.model.cut_freq
        self.down_sampling = (self.seq_len // self.seg_num_x) // self.num_sampling

        self.fc1 = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, self.output_size)

    def fast_tf_process(self, x):

        return self.fast_tf(x)

    def forward(self, x, edge_index):

        batch_size, num_nodes, seq_len, features = x.shape


        if edge_index.dim() == 2:
            max_index = edge_index.max().item()
            if max_index >= num_nodes:

                mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                edge_index = edge_index[:, mask]


        x = x.reshape(batch_size * num_nodes, seq_len, features)
        gru_out, _ = self.gru(x)
        x = gru_out[:, -1, :]

        edge_index = edge_index.to(x.device)


        assert x.size(0) == batch_size * num_nodes, f"Expected size {batch_size * num_nodes}, got {x.size(0)}"


        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)


        x = x.view(batch_size, num_nodes, -1)


        x = x.unsqueeze(2)  # [batch_size, num_nodes, 1, hidden_channels]
        x = x.expand(-1, -1, self.seq_len, -1)  # [batch_size, num_nodes, seq_len, hidden_channels]
        x = x.permute(0, 2, 1, 3)  # [batch_size, seq_len, num_nodes, hidden_channels]


        x = self.fast_tf_process(x)


        batch_size, pred_len, num_nodes, hidden_channels = x.shape
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_nodes, pred_len, hidden_channels]


        x = x.reshape(batch_size * num_nodes * pred_len, hidden_channels)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)


        x = x.view(batch_size, num_nodes, pred_len, -1)
        x = x.mean(dim=2)  # [batch_size, num_nodes, output_size]

        return x