import torch
import argparse
import logging
import torch.nn as nn
import yaml
import numpy as np
from easydict import EasyDict as edict
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pandas as pd
from wpf_dataset import PGL4WPFDataset, data_augment
from wpf_model16 import GGRU_FastTF
import optimization as optim
from metrics import mae, rmse, regressor_detailed_scores
from utils import save_model, _create_if_not_exist, load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import r2_score
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)

def get_step_scheduler(optimizer, config):
    return StepLR(
        optimizer,
        step_size=config.scheduler.step_size,
        gamma=config.scheduler.gamma
    )


def train_and_evaluate(config, train_data, valid_data, test_data):

    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epsilon = 1e-8

    graph = train_data.graph.to(device)
    # DataLoader setup
    train_data_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    valid_data_loader = DataLoader(valid_data, batch_size=config['batch_size'], shuffle=False, drop_last=False)
    test_data_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, drop_last=False)


    input_channels = config.model.input_channels
    hidden_channels = config.model.hidden_channels
    output_size = config.model.output_size
    kernel_size = config.model.kernel_size
    dropout = config.model.dropout

    model = GGRU_FastTF(input_channels, hidden_channels, output_size, kernel_size, dropout).to(device)

    # Loss function and optimizer
    loss_fn = getattr(nn, config['loss']['name'])(reduction=config['loss']['reduction'])
    optimizer, clip_gradients = optim.get_optimizer(model=model, learning_rate=config['lr'])
    scheduler = optim.get_step_scheduler(optimizer, config)

    _create_if_not_exist(config['output_path'])
    global_step = 0
    test_records = []
    train_losses = []
    valid_losses = []

    patience_counter = 0
    best_valid_score = float('inf')
    best_test_results = None
    best_test_predictions = None
    best_epoch = -1
    best_metrics = None
    best_valid_loss = float('inf')

    logging.info("Starting training...")

    for epoch in range(config['epoch']):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch + 1}"):
            batch_x, batch_y, _, _ = batch

            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()

            batch_size, num_nodes, seq_len, feature_dim = batch_x.shape


            base_edge_index = graph.edge_index.to(device)


            num_graphs = batch_size * seq_len


            offsets = torch.arange(num_graphs, device=device) * num_nodes


            edge_index = base_edge_index.unsqueeze(1).repeat(1, num_graphs, 1).reshape(2, -1)
            edge_index[0] += offsets.repeat_interleave(base_edge_index.size(1))
            edge_index[1] += offsets.repeat_interleave(base_edge_index.size(1))


            max_index = batch_size * seq_len * num_nodes
            if edge_index.max() >= max_index:
                raise ValueError(f"edge_index contains indices >= {max_index}")

            # Reshape batch_x to [batch_size, seq_len, num_nodes, feature_dim]
            # No need to reshape here as GFastTF handles it
            # Pass directly to model
            pred_y = model(batch_x)
            #print(f"pred_y shape: {pred_y.shape}")
            # print(f"Adjusted pred_y shape: {pred_y.shape}, batch_y shape: {batch_y.shape}")

            #print(f"Initial pred_y shape: {pred_y.shape}, batch_y shape: {batch_y.shape}")


            if pred_y.dim() == 4 and batch_y.dim() == 3:
                pred_y = pred_y[:, :, :batch_y.size(2)]
                #print(f"Adjusted pred_y shape: {pred_y.shape}, batch_y shape: {batch_y.shape}")
            elif pred_y.dim() != batch_y.dim():
                raise ValueError(f"Shape mismatch: pred_y shape {pred_y.shape}, batch_y shape {batch_y.shape}")

            loss = loss_fn(pred_y, batch_y)

            epoch_loss += loss.item()
            batch_count += 1

            if torch.isnan(loss):
                logging.warning("NaN loss detected. Skipping this batch.")
                continue

            optimizer.zero_grad()
            loss.backward()
            clip_gradients()
            optimizer.step()
            global_step += 1


        torch.cuda.empty_cache()


        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        logging.info("Epoch %d Average Loss: %.6f", epoch + 1, avg_loss)


        patience = config.get('early_stopping', {}).get('patience', 10)
        min_delta = config.get('early_stopping', {}).get('min_delta', 1e-4)


        valid_results = evaluate(valid_data_loader, valid_data.get_raw_df(), model, loss_fn, config, graph, device)
        valid_score = valid_results['score']
        valid_losses.append(valid_results['loss'])


        logging.info("\nValidation Results for Epoch %d:", epoch + 1)
        logging.info("Validation Loss: %.4f", valid_results['loss'])
        logging.info("Validation MAE: %.4f", valid_results['mae'])
        logging.info("Validation RMSE: %.4f", valid_results['rmse'])
        logging.info("Validation Total MAE: %.4f", valid_results['total_mae'])
        logging.info("Validation Total RMSE: %.4f", valid_results['total_rmse'])
        logging.info("Validation R2 Score: %.4f", valid_results['r2'])
        logging.info("Validation Score: %.4f\n", valid_score)

        if valid_score < best_valid_score - min_delta:

            test_results = evaluate(test_data_loader, test_data.get_raw_df(), model, loss_fn, config, graph, device)


            best_valid_score = valid_score
            best_epoch = epoch
            best_metrics = {
                'validation_score': valid_score,
                'test_mae': test_results['mae'],
                'test_rmse': test_results['rmse'],
                'test_total_mae': test_results['total_mae'],
                'test_total_rmse': test_results['total_rmse'],
                'test_r2': test_results['r2']
            }
            best_test_results = test_results
            patience_counter = 0


            save_model(
                output_path=config.output_path,
                model=model,
                epoch=epoch,
                best_score=best_valid_score,
                current_score=valid_score
            )


            total_power_predictions = test_results['total_power_predictions']
            total_power_targets = test_results['total_power_targets']

            if len(total_power_predictions) > 0 and len(total_power_targets) > 0:

                flattened_predictions = total_power_predictions.flatten()
                flattened_targets = total_power_targets.flatten()


                raw_df_lst = test_data.get_raw_df()


                if raw_df_lst and len(raw_df_lst) > 0:

                    timestamp_col = None
                    potential_cols = ['Tmstamp', 'time', 'date', 'timestamp', 'Date', 'Time']
                    for col in potential_cols:
                        if col in raw_df_lst[0].columns:
                            timestamp_col = col
                            break


                    if timestamp_col is not None:
                        timestamps = raw_df_lst[0][timestamp_col].values


                        min_length = min(len(flattened_predictions), len(flattened_targets), len(timestamps))
                        timestamps = timestamps[:min_length]
                        flattened_predictions = flattened_predictions[:min_length]
                        flattened_targets = flattened_targets[:min_length]
                    else:

                        min_length = min(len(flattened_predictions), len(flattened_targets))
                        timestamps = [f"Step_{i + 1}" for i in range(min_length)]
                        flattened_predictions = flattened_predictions[:min_length]
                        flattened_targets = flattened_targets[:min_length]
                else:

                    min_length = min(len(flattened_predictions), len(flattened_targets))
                    timestamps = [f"Step_{i + 1}" for i in range(min_length)]
                    flattened_predictions = flattened_predictions[:min_length]
                    flattened_targets = flattened_targets[:min_length]


                results_df = pd.DataFrame({
                    'Timestamp': timestamps,
                    'Predicted_Power': flattened_predictions,
                    'Real_Power': flattened_targets
                })


                result_path = os.path.join(config['output_path'], f'best_model_total_power_epoch_{epoch + 1}.csv')
                results_df.to_csv(result_path, index=False)
                logging.info(f"Saved best model total power predictions to {result_path}")


            test_results = evaluate(test_data_loader, test_data.get_raw_df(), model, loss_fn, config, graph, device)
            best_test_results = test_results




        else:
            patience_counter += 1
            logging.info(f"No improvement in validation score. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs")

            logging.info("\n" + "=" * 50)
            logging.info("Best Results Summary:")
            logging.info(f"Best Epoch: {best_epoch + 1}")
            logging.info(f"Best Validation Score: {best_valid_score:.4f}")
            if best_metrics:
                logging.info(f"Best Test MAE: {best_metrics['test_mae']:.4f}")
                logging.info(f"Best Test RMSE: {best_metrics['test_rmse']:.4f}")
                logging.info(f"Best Test Total MAE: {best_metrics['test_total_mae']:.4f}")
                logging.info(f"Best Test Total RMSE: {best_metrics['test_total_rmse']:.4f}")
                logging.info(f"Best Test R2 Score: {best_metrics['test_r2']:.4f}")
            logging.info("=" * 50 + "\n")
            break

        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logging.info(f"Current Learning Rate: {current_lr:.6f}")


        test_results = evaluate(test_data_loader, test_data.get_raw_df(), model, loss_fn, config, graph, device)
        logging.info("Test Results: %s", dict(test_results))
        test_records.append(test_results)


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(config['output_path'], 'loss_plot.png')
    plot_dir = os.path.dirname(plot_path)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(plot_path)
    plt.close()

def evaluate(data_loader, raw_df_lst, model, loss_fn, config, graph, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    total_power_predictions = []
    total_power_targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch_x, batch_y = batch[:2]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            batch_size, num_nodes, seq_len, num_features = batch_x.shape


            edge_index = graph.edge_index.clone()
            edge_index[0] = edge_index[0] % num_nodes + torch.div(edge_index[0], num_nodes, rounding_mode='floor') * num_nodes
            edge_index[1] = edge_index[1] % num_nodes + torch.div(edge_index[1], num_nodes, rounding_mode='floor') * num_nodes


            mask = (edge_index[0] < batch_size * num_nodes) & (edge_index[1] < batch_size * num_nodes)
            edge_index = edge_index[:, mask]


            outputs = model(batch_x)

            if len(outputs.shape) == 4:
                outputs = outputs.squeeze(-1)
            if len(batch_y.shape) == 4:
                batch_y = batch_y.squeeze(-1)

            loss = loss_fn(outputs, batch_y)
            total_loss += loss.item()

            batch_total_power_pred = outputs.sum(dim=1).cpu().numpy()
            batch_total_power_true = batch_y.sum(dim=1).cpu().numpy()

            total_power_predictions.extend(batch_total_power_pred)
            total_power_targets.extend(batch_total_power_true)

            predictions.append(outputs.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)


    total_power_predictions = np.array(total_power_predictions)
    total_power_targets = np.array(total_power_targets)
    r2 = r2_score(total_power_targets.flatten(), total_power_predictions.flatten())


    mae_value = mae(predictions, targets)
    rmse_value = rmse(predictions, targets)
    total_mae, total_rmse = regressor_detailed_scores(predictions, targets, raw_df_lst, config.capacity, config.output_size)

    score = (mae_value + rmse_value) / 2
    total_score = (total_mae + total_rmse) / 2

    return {
        "loss": avg_loss,
        "mae": mae_value,
        "rmse": rmse_value,
        "total_mae": total_mae,
        "total_rmse": total_rmse,
        "score": score,
        "total_score": total_score,
        "r2": r2,
        "total_power_predictions": np.array(total_power_predictions),
        "total_power_targets": np.array(total_power_targets)
    }


def custom_collate_fn(batch, p=0.8, alpha=0.5, beta=0.5):
    X_list, y_list, graphs, _ = zip(*batch)
    X = torch.stack(X_list)
    y = torch.stack(y_list)


    X_aug, y_aug = data_augment(X, y, p=p, alpha=alpha, beta=beta)


    return X_aug, y_aug, graphs[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    args = parser.parse_args()

    with open(args.conf, 'r', encoding='utf-8') as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))

    # Data loading
    size = [config.input_len, config.output_len]

    train_data = PGL4WPFDataset(config.data_path, filename=config.filename, size=size, flag='train',
                                total_days=config.total_days, train_days=config.train_days, val_days=config.val_days,
                                test_days=config.test_days, alpha=config.alpha, threshold=config.threshold)
    valid_data = PGL4WPFDataset(config.data_path, filename=config.filename, size=size, flag='val',
                                total_days=config.total_days, train_days=config.train_days, val_days=config.val_days,
                                test_days=config.test_days, alpha=config.alpha, threshold=config.threshold)
    test_data = PGL4WPFDataset(config.data_path, filename=config.filename, size=size, flag='test',
                               total_days=config.total_days, train_days=config.train_days, val_days=config.val_days,
                               test_days=config.test_days, alpha=config.alpha, threshold=config.threshold)

    # Train and evaluate
    results = train_and_evaluate(config, train_data, valid_data, test_data)

