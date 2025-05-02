# wpf_dataset.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import yaml
import time
import datetime
from torch_geometric.data import Data, Batch
from easydict import EasyDict as edict  # 新增导入
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO)

def time2obj(time_sj):
    return time.strptime(time_sj, "%H:%M")

def time2int(time_sj):
    return int(time.mktime(time2obj(time_sj)))

def int2time(t):
    timestamp = datetime.datetime.fromtimestamp(t)
    return timestamp.strftime('"%H:%M"')

def func_add_t(x):
    time_strip = 600
    time_obj = time2obj(x)
    time_e = ((time_obj.tm_sec + time_obj.tm_min * 60 + time_obj.tm_hour * 3600) // time_strip) % 288
    return time_e

def func_add_h(x):
    time_obj = time2obj(x)
    return time_obj.tm_hour

def load_config(config_path):

    with open(config_path, 'r', encoding='utf-8') as f:
        config = edict(yaml.safe_load(f))
    return config

def data_augment(X, y, p=0.8, alpha=0.5, beta=0.5):
    """Regression SMOTE for data augmentation"""
    device = X.device

    fix_X, X_var = X[:, :, :, :2], X[:, :, :, 2:]
    fix_y, y_var = y[:, :, :2], y[:, :, 2:]

    batch_size, num_nodes, seq_len, feature_dim = X_var.shape



    random_values = torch.rand(batch_size, device=device)
    idx_to_change = random_values < p  # [batch_size]


    num_selected = idx_to_change.sum().item()


    if num_selected == 0:
        return torch.cat([fix_X, X_var], dim=-1), torch.cat([fix_y, y_var], dim=-1)


    np_betas = np.random.beta(alpha, beta, num_selected) / 2 + 0.5
    random_betas = torch.tensor(np_betas, dtype=torch.float32, device=device)

    index_permute = torch.arange(X.size(0), device=device)


    random_betas_X = random_betas.view(-1, 1, 1, 1)  # [num_selected, 1, 1, 1]
    random_betas_y = random_betas.view(-1, 1, 1)     # [num_selected, 1, 1]


    X_var_aug = X_var[idx_to_change] * random_betas_X + X_var[index_permute] * (1 - random_betas_X)
    y_var_aug = y_var[idx_to_change] * random_betas_y + y_var[index_permute] * (1 - random_betas_y)


    X_var[idx_to_change] = X_var_aug
    y_var[idx_to_change] = y_var_aug


    return torch.cat([fix_X, X_var], dim=-1), torch.cat([fix_y, y_var], dim=-1)

def custom_collate_fn(batch, p=0.8, alpha=0.5, beta=0.5):

    seq_x, seq_y, graphs, sub_graphs = zip(*batch)


    seq_x = torch.stack(seq_x)
    seq_y = torch.stack(seq_y)


    seq_x, seq_y = data_augment(seq_x, seq_y, p=p, alpha=alpha, beta=beta)


    batched_graphs = Batch.from_data_list(graphs)
    batched_sub_graphs = Batch.from_data_list(sub_graphs)

    return seq_x, seq_y, batched_graphs, batched_sub_graphs


class PGL4WPFDataset(Dataset):
    def __init__(self, data_path, filename='wtbdata_245days.csv', flag='train', size=None,
                 capacity=134, day_len=24 * 6, train_days=153, val_days=16, test_days=15,
                 total_days=184, theta=0.9, window_size=36, alpha=0.5, threshold=0.5,
                 outlier_method="z_score", z_score_threshold=3, iqr_multiplier=1.5):
        super().__init__()
        self.window_size = window_size
        self.unit_size = day_len
        if size is None:

            self.input_len =72
            self.output_len = 3
        else:
            self.input_len = size[0]
            self.output_len = size[1]

        self.start_col = 0
        self.capacity = capacity
        self.theta = theta
        self.alpha = alpha
        self.threshold = threshold
        self.outlier_method = outlier_method
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.data_path = data_path
        self.filename = filename


        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))

        self.original_data_shape = df_raw.shape
        self.total_size = self.original_data_shape[0] // self.capacity

        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        self.test_size = test_days * self.unit_size


        self.__read_data__()

    def _create_valid_mask(self, data):
        nan_cond = pd.isna(data).any(axis=1)
        invalid_cond = (data['Patv'] < 0) | \
                       ((data['Patv'] == 0) & (data['Wspd'] > 2.5)) | \
                       ((data['Pab1'] > 89) | (data['Pab2'] > 89) | (data['Pab3'] > 89)) | \
                       ((data['Wdir'] < -180) | (data['Wdir'] > 180) | (data['Ndir'] < -720) |
                        (data['Ndir'] > 720))
        return ~(invalid_cond | nan_cond)



    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        df_location = pd.read_csv(os.path.join(self.data_path, 'turb_location.csv'))


        df_raw = df_raw.merge(df_location, on='TurbID', how='left')


        df_data, raw_df_data = self.data_preprocess(df_raw)


        self.valid_mask = self._create_valid_mask(df_data)


        self.df_data = df_data
        self.raw_df_data = raw_df_data


        self.data_x, self.graph = self.build_graph_data(df_data)

    def get_valid_mask(self):
        return self.valid_mask

    def __getitem__(self, index):

        if index < 0 or index >= self.__len__():
            print(f"无效的索引: {index}, 数据集长度: {self.__len__()}")
            raise IndexError("索引超出范围")

        try:

            seq_start = index
            seq_end = seq_start + self.input_len
            target_end = seq_end + self.output_len

            # 确保不会超出数据范围
            if target_end > self.data_x.shape[1]:
                raise IndexError(
                    f"索引超出范围: seq_start={seq_start}, seq_end={seq_end}, target_end={target_end}, data_shape={self.data_x.shape}")


            seq_x = self.data_x[:, seq_start:seq_end, :]  # [num_nodes, input_len, features]
            seq_y = self.data_x[:, seq_end:target_end, -1]  # [num_nodes, output_size] # [output_size, num_nodes]

            graph = self.graph
            batch_node_indices = list(range(self.capacity))
            sub_graph = self.create_subgraph(batch_node_indices)

            return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), graph, sub_graph

        except Exception as e:
            print(f"在 __getitem__ 中发生错误: {e}, index={index}")
            raise

    def create_subgraph(self, node_indices):
        graph = self.graph


        mask = (graph.edge_index[0] < len(node_indices)) & (graph.edge_index[1] < len(node_indices))
        sub_edge_index = graph.edge_index[:, mask]
        sub_edge_attr = graph.edge_attr[mask] if graph.edge_attr is not None else None

        sub_graph = Data(
            num_nodes=len(node_indices),
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr
        )

        return sub_graph

    def __len__(self):

        return self.data_x.shape[1] - self.input_len - self.output_len + 1

    def data_preprocess(self, df_data):
        feature_name = [
            n for n in df_data.columns
            if "Patv" not in n and 'Day' not in n and 'Tmstamp' not in n and 'TurbID' not in n
        ]
        feature_name.append("Patv")


        new_df_data = df_data[feature_name].copy()


        t = df_data['Tmstamp'].apply(func_add_t)
        new_df_data.insert(0, 'time', t)

        weekday = df_data['Day'].apply(lambda x: x % 7)
        new_df_data.insert(0, 'weekday', weekday)


        for column in new_df_data.columns:
            if column in ['time', 'weekday']:
                continue

            if self.outlier_method == "z_score":
                mean = new_df_data[column].mean()
                std = new_df_data[column].std()
                outlier_condition = (new_df_data[column] > mean + self.z_score_threshold * std) | \
                                    (new_df_data[column] < mean - self.z_score_threshold * std)
            elif self.outlier_method == "iqr":
                Q1 = new_df_data[column].quantile(0.25)
                Q3 = new_df_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
                outlier_condition = (new_df_data[column] < lower_bound) | (new_df_data[column] > upper_bound)
            else:
                raise ValueError(f"未知的异常值检测方法: {self.outlier_method}")


            new_df_data.loc[outlier_condition, column] = np.nan


        new_df_data.interpolate(method='linear', inplace=True, limit_direction='both')


        new_df_data = new_df_data.ffill()
        new_df_data = new_df_data.bfill()


        self.check_for_nulls(new_df_data, "处理后")

        scaler = MinMaxScaler()

        numeric_features = new_df_data.columns.difference(['time', 'weekday'])
        new_df_data[numeric_features] = scaler.fit_transform(new_df_data[numeric_features])


        self.data_min = scaler.data_min_
        self.data_max = scaler.data_max_


        raw_df_data = new_df_data.copy()

        return new_df_data, raw_df_data

    def build_graph_data(self, df_data):
        cols_data = df_data.columns[self.start_col:]
        df_data = df_data[cols_data]
        raw_df_data = self.raw_df_data[cols_data]

        data = df_data.values
        raw_data = raw_df_data.values
        valid_mask = self.valid_mask.values

        try:
            data = np.reshape(data, [self.capacity, self.total_size, len(cols_data)])
            raw_data = np.reshape(raw_data, [self.capacity, self.total_size, len(cols_data)])
            valid_mask = np.reshape(valid_mask, [self.capacity, self.total_size])
        except ValueError as e:
            raise e


        border1s = [
            0,
            self.train_size - self.input_len - self.output_len + 1,
            self.train_size + self.val_size - self.input_len - self.output_len + 1
        ]
        border2s = [
            self.train_size, self.train_size + self.val_size,
                             self.train_size + self.val_size + self.test_size
        ]


        y = data[:, :, -1]
        self.y_all = torch.tensor(y, dtype=torch.float32).view(-1)
        self.y_mean = self.y_all.mean()
        self.y_scale = self.y_all.std()


        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.raw_df = []
        self.valid_mask_list = []
        for turb_id in range(self.capacity):
            self.raw_df.append(
                pd.DataFrame(
                    data=raw_data[turb_id, border1 + self.input_len:border2],
                    columns=cols_data
                )
            )
            self.valid_mask_list.append(valid_mask[turb_id, border1 + self.input_len:border2])

        data_x = data[:, border1:border2, :]


        turbine_locations = pd.read_csv(os.path.join(self.data_path, 'turb_location.csv'))
        coords = turbine_locations[['x', 'y']].values


        dist_matrix = cdist(coords, coords)

        power_data = data_x[:, :, -1]
        power_data = power_data.reshape(self.capacity, -1)
        correlation_matrix = np.corrcoef(power_data)
        correlation_matrix = np.abs(correlation_matrix)


        correlation_matrix = np.nan_to_num(correlation_matrix)


        scaler = MinMaxScaler()
        correlation_scaled = scaler.fit_transform(correlation_matrix)


        distance_weights = 1 / (dist_matrix + 1e-6)
        distance_weights = distance_weights / distance_weights.max()


        combined_metric = self.alpha * correlation_scaled + (1 - self.alpha) * distance_weights


        adj_matrix = (combined_metric > self.threshold).astype(float) * combined_metric
        np.fill_diagonal(adj_matrix, 0)


        edges = np.where(adj_matrix > 0)
        assert np.all(edges[0] < self.capacity) and np.all(edges[1] < self.capacity), "Invalid edge indices detected"

        edge_index = torch.tensor(np.stack([edges[0], edges[1]], axis=0), dtype=torch.long)
        edge_weights = torch.tensor(adj_matrix[edges[0], edges[1]], dtype=torch.float32).unsqueeze(1)


        # Modify the edge creation part
        edges = np.where(adj_matrix > 0)

        # Add validation checks
        edge_source = edges[0]
        edge_target = edges[1]

        # Ensure all indices are within bounds
        valid_edges_mask = (edge_source < self.capacity) & (edge_target < self.capacity)
        edge_source = edge_source[valid_edges_mask]
        edge_target = edge_target[valid_edges_mask]

        edge_index = torch.tensor(np.stack([edge_source, edge_target], axis=0), dtype=torch.long)
        edge_weights = torch.tensor(adj_matrix[edge_source, edge_target], dtype=torch.float32).unsqueeze(1)

        if edge_index.numel() == 0:
            logging.warning("Empty edge index detected. Creating default edges.")

            source_nodes = torch.arange(self.capacity - 1)
            target_nodes = torch.arange(1, self.capacity)
            edge_index = torch.stack([
                torch.cat([source_nodes, target_nodes]),
                torch.cat([target_nodes, source_nodes])
            ])
            edge_weights = torch.ones(edge_index.size(1), 1)  # 默认权重为1
        else:

            max_index = edge_index.max(dim=1)[0].max().item()
            min_index = edge_index.min(dim=1)[0].min().item()

            if max_index >= self.capacity or min_index < 0:
                logging.warning(f"Invalid edge indices detected. max: {max_index}, min: {min_index}")

                valid_edges_mask = (edge_index[0] < self.capacity) & (edge_index[1] < self.capacity) & \
                                   (edge_index[0] >= 0) & (edge_index[1] >= 0)
                edge_index = edge_index[:, valid_edges_mask]
                edge_weights = edge_weights[valid_edges_mask]

        # Create graph with validated edges
        optimal_graph = Data(
            num_nodes=self.capacity,
            edge_index=edge_index,
            edge_attr=edge_weights
        )

        return data_x, optimal_graph

    def get_raw_df(self):
        return self.raw_df

    def get_valid_mask_list(self):
        return self.valid_mask_list

    def get_raw_df_lst(self):

        return self.raw_df

    def get_y_mean_scale(self):

        return self.y_mean, self.y_scale

if __name__ == "__main__":

    config = load_config('config.yaml')
    data_path = config['data_path']
    outlier_config = config.get('outlier', {})
    outlier_method = outlier_config.get('method', 'z_score')
    z_score_threshold = outlier_config.get('z_score_threshold', 3)
    iqr_multiplier = outlier_config.get('iqr_multiplier', 1.5)


    pyg_dataset = PGL4WPFDataset(
        data_path,
        filename="wtbdata_245days.csv",
        alpha=0.5,
        threshold=0,
        outlier_method=outlier_method,
        z_score_threshold=z_score_threshold,
        iqr_multiplier=iqr_multiplier
    )


    train_loader = DataLoader(
        pyg_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, p=0.8, alpha=0.5, beta=0.5)
    )


