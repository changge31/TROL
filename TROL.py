# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import expit
import pandas as pd
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import math
import os
import cProfile
import pickle
import random
from torch.utils.data import DataLoader
from momentfm import MOMENTPipeline
import time
from tqdm import tqdm


class Parameters:

    # Training parameters
    TRAINING_PARAMETERS = {
        "budget": 110,  # 64.9 for gotrack, 107.8 for fishing
        "batch_size": 64,
        "epoch_per_purchase": 20,
        "initial_reward_rUCB": 1,
        "epsilon_wUCB": 2,
        "epsilon_rUCB": 2,
        "epsilon_LinUCB": 2,
        "context_length_wUCB": 10,
        "context_length_rUCB": 10,
        "context_length_LinUCB": 10,
        "reward_computation": "relative",   # "absolute", "relative", "normalised"
        "train_epochs": 200,
    }

    # Data parameters
    DATA_PARAMETERS = {
        "dataset": "bjtaxi",
        "num_bids": 100, # 59 for gotrack, 98 for fishing
        "bid_generation": "uniform", # "uniform" or "normal"
        "length_per_epoch": 9,
        "p_noise": 0.10,
    }

    # path parameters
    PATH_PARAMETERS = {
        "path": 'results/bjtaxi/7-5-2',
        "data_path": 'datasets/bjtaxi/',
        "train_data_path": "datasets/bjtaxi/train_noise_10.txt",
    }

    ATTACK_PARAMETERS = {
        "mode": 1,
    }


class DataProcessor:
    def __init__(self, num_seq, input_length=8, output_length=1, batch_size=3, input_channels=2):
        self.data = [torch.empty((0, input_channels)) for _ in range(num_seq)]
        self.input_length = input_length
        self.output_length = output_length
        self.batch_size = Parameters.TRAINING_PARAMETERS["batch_size"]
        self.input_channels = input_channels
        # self.input_output_pairs = self._generate_input_output_pairs()
        self.input_output_pairs = []
        self.processed_lengths = [0] * num_seq
        self.interval = 4

    def _generate_input_output_pairs(self, idx, incremental=False):
        # print("self.data: ", self.data)
        # for idx, sample in enumerate(self.data):
        #     if len(sample.shape) != 2 or sample.shape[1] != self.input_channels:
        #         raise ValueError(f"Each data sample must have the shape (?, {self.input_channels})")
        
        sample = self.data[idx]
        # print("processed_lengths: ", self.processed_lengths[idx])
        start_pos = max(0, self.processed_lengths[idx])
        # print("start_pos: ", start_pos)
        end_pos = sample.shape[0]
        # print("end condition: ", end_pos - self.input_length - self.output_length + 1)

        # print("start_pos: ", start_pos)
        # print("end_pos: ", end_pos)
        # print(end_pos - self.input_length - self.output_length + 1)

        for start in range(start_pos, end_pos - self.input_length - self.output_length + 1, self.interval):
            # print("start: ", start)
            input_segment = sample[start:start + self.input_length, :].permute(1, 0)
            output_segment = sample[start + self.input_length:start + self.input_length + self.output_length, :].permute(1, 0)
            self.input_output_pairs.append((input_segment, output_segment))

        if end_pos - start_pos >= self.input_length + self.output_length:
            self.processed_lengths[idx] = start + self.interval
        else:
            self.processed_lengths[idx] = end_pos
            
        # self.processed_lengths[idx] = start + self.interval if start + self.interval <= end_pos else end_pos

        # print("pairs: ", self.input_output_pairs)

    def update_data(self, new_data, data_index):
        if type(new_data) != torch.tensor:
            new_data = torch.tensor(new_data)
        if new_data.shape[1] != self.input_channels:
            raise ValueError(f"New data sample must have the shape (?, {self.input_channels})")

        self.data[data_index] = torch.cat([self.data[data_index], new_data], dim=0)
        self._generate_input_output_pairs(data_index, incremental=True)
        random.shuffle(self.input_output_pairs)

    def get_batches(self):
        for i in range(0, len(self.input_output_pairs), self.batch_size):
            batch = self.input_output_pairs[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                inputs, outputs = zip(*batch)
                yield torch.stack(inputs), torch.stack(outputs)
            else:
                # If the last batch is smaller than the other batches, pad it with zeros
                # print("ELSE: ")
                inputs, outputs = zip(*batch)
                # print("inputs: ", inputs)
                # print("outputs: ", outputs)
                padded_inputs = torch.zeros((self.batch_size, self.input_channels, self.input_length))
                # print("padded_inputs: ", padded_inputs)
                padded_outputs = torch.zeros((self.batch_size, self.input_channels, self.output_length))
                # print("padded_outputs: ", padded_outputs)
                padded_inputs[:len(inputs), :, :] = torch.stack(inputs)
                # print("padded_inputs: ", padded_inputs)
                padded_outputs[:len(outputs), :, :] = torch.stack(outputs)
                # print("padded_outputs: ", padded_outputs)
                yield padded_inputs, padded_outputs

    def __len__(self):
        # Return the number of batches
        # return len(self.input_output_pairs) // self.batch_size
        # print(math.ceil(len(self.input_output_pairs) / self.batch_size))
        return math.ceil(len(self.input_output_pairs) / self.batch_size)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        batch = self.input_output_pairs[start_idx:end_idx]
        inputs, outputs = zip(*batch)
        return torch.stack(inputs), torch.stack(outputs)


def load_dataset(data_path):
    all_data = []

    with open(data_path, 'r') as file:
        for line in file:
            data_sequence = eval(line.strip())
            all_data.append(data_sequence)
    
    return all_data

def sigmoid(x, k=0.1):
    return 1 / (1 + np.exp(-k * x))


def validate_epoch(moment, model, dataset, loss_function, device):
    # print("validate_epoch")
    model.eval()
    total_loss = 0  # sum of the loss of all the sequences in the dataset for one epoch
    with torch.no_grad():
        for src, tgt in dataset.get_batches():
            # print("src: ", src)
            src, tgt = src.to(device), tgt.to(device)
            embeddings = moment(src).embeddings
            output = model(embeddings)
            tgt = tgt.squeeze(-1)
            # print("output: ", output)
            # print("tgt: ", tgt)
            # print("output: ", output[:5])
            # print("tgt: ", tgt[:5])

            mask = (tgt != 0)
            # print("mask: ", mask)
            output = output[mask]
            tgt = tgt[mask]
            # print("output: ", output)
            # print("tgt: ", tgt)
            loss = loss_function(output, tgt)
            # print("loss: ", loss.item())            
            total_loss += loss.item()
    # print(total_loss / len(dataset))
    # print("val finished")
    return total_loss / len(dataset) 

def train_epoch(moment, model, dataset, loss_function, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in dataset.get_batches():
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        embeddings = moment(src).embeddings
        output = model(embeddings)
        tgt = tgt.squeeze(-1)
        mask = (tgt != 0)
        output = output[mask]
        tgt = tgt[mask]
        loss = loss_function(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)


def set_seed(seed):
    torch.manual_seed(seed)  # Set seed for PyTorch CPU operations
    torch.cuda.manual_seed(seed)  # Set seed for CUDA operations (single GPU)
    torch.cuda.manual_seed_all(seed)  # Set seed for CUDA operations (all GPUs)
    np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)  # Set seed for Python's random module


def experiment(exp):
# %%
    random_allocation=True
    rUCB=True
    LinUCB=True
    wUCB=True

    # %%
    # initialise

    output_path = Parameters.PATH_PARAMETERS["path"] + "-" + str(exp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # model = nn.Linear(1024, 2)
    # # store the model in the folder
    # torch.save(model.state_dict(), "datasets/geolife" + '/init_pre_model.pth')

    # bids = np.sort(np.random.rand(100))
    # np.save("datasets/geolife/geolife_bids.npy", bids)
    # bids = np.sort(np.random.rand(Parameters.DATA_PARAMETERS["num_bids"]))
    # # generate random bids using normal distribution
    # r_bids = np.sort(np.random.randn(Parameters.DATA_PARAMETERS["num_bids"]))
    # bids = (r_bids - np.min(r_bids)) / (np.max(r_bids) - np.min(r_bids))
    # bids = np.clip(bids, bids[1]-1e-6, bids[-2]+1e-6)
    # # print("Bids: ", bids)
    # np.save('datasets/bids100_normal.npy', bids)

    # bids = np.full(Parameters.DATA_PARAMETERS["num_bids"], 0.3)
    # load the bids
    bids = np.load(Parameters.PATH_PARAMETERS["data_path"] + 'bids.npy')
    # bids = np.full(Parameters.DATA_PARAMETERS["num_bids"], 1)

    if Parameters.TRAINING_PARAMETERS["budget"] < np.sum(bids) + bids[-1]:
        raise ValueError("Budget is not enough to purchase all the data.")

    # load dataset
    if Parameters.ATTACK_PARAMETERS["mode"] == 0:
        code = "zero"
        train_data_path = Parameters.PATH_PARAMETERS["data_path"] + "train_" + code + "_" + str(round(Parameters.DATA_PARAMETERS["num_bids"]*Parameters.DATA_PARAMETERS["p_noise"])) + ".txt"
    elif Parameters.ATTACK_PARAMETERS["mode"] == 1:
        code = "noise"
        train_data_path = Parameters.PATH_PARAMETERS["data_path"] + "train_" + code + "_" + str(round(Parameters.DATA_PARAMETERS["num_bids"]*Parameters.DATA_PARAMETERS["p_noise"])) + ".txt"
    else:
        code = "attack"
        train_data_path = Parameters.PATH_PARAMETERS["train_data_path"]
    
    train_data = load_dataset(train_data_path)
    val_data = load_dataset(Parameters.PATH_PARAMETERS["data_path"]+"val.txt")
    test_data = load_dataset(Parameters.PATH_PARAMETERS["data_path"]+"test.txt")

    val_dataset = DataProcessor(len(val_data))
    for i in range(len(val_data)):
        if len(val_data[i]) >= 72:
            val_dataset.update_data(val_data[i][:72], i)
        else:
            val_dataset.update_data(val_data[i], i)

    test_dataset = DataProcessor(len(test_data))
    for i in range(len(test_data)):
        test_dataset.update_data(test_data[i], i)


    # store the parameters in a CSV file
    df_parameters = pd.DataFrame()
    # store all parameters in class parameters in the dataframe
    for key, value in Parameters.TRAINING_PARAMETERS.items():
        df_parameters[key] = [value]
    for key, value in Parameters.DATA_PARAMETERS.items():
        df_parameters[key] = [value]
    for key, value in Parameters.PATH_PARAMETERS.items():
        df_parameters[key] = [value]
    df_parameters.to_csv(output_path + '/parameters.csv')
    df_bids = pd.DataFrame()
    df_bids["bids"] = bids.tolist()
    df_bids.to_csv(output_path + '/bids.csv')


    # initialise model

    # load the pretrained model
    moment = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={'task_name': 'embedding'},
    )
    moment.init()
    print("Model loaded.")
    model = nn.Linear(1024, 2)
    # store the model in the folder
    # torch.save(model.state_dict(), output_path + '/model.pth')
    torch.load(Parameters.PATH_PARAMETERS["data_path"] + 'init_pre_model.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moment.to(device)
    model.to(device)

    # open a file to store some useful information
    f = open(output_path + '/info.txt', 'w')


    # %%
    if random_allocation:
        print("Random allocation...")
        model_random = nn.Linear(1024, 2)
        model_random.load_state_dict(model.state_dict())
        model_random.to(device)
        optimizer_random = optim.Adam(model_random.parameters(), lr=0.001)
        loss_function_random = nn.MSELoss()

        selection_history_random = []
        total_payment_random = 0
        payments_random = []
        
        purchase_history_random = np.full(Parameters.DATA_PARAMETERS["num_bids"], 0, dtype=int) 
        # purchased_train_data_random = [[] for _ in range(Parameters.DATA_PARAMETERS["num_bids"])]
        current_train_data_random = DataProcessor(Parameters.DATA_PARAMETERS["num_bids"])
        train_loss_history_random = []
        val_loss_history_random = []

        # empty_val_loss_random = True

        for i in range(Parameters.DATA_PARAMETERS["num_bids"]):
            if total_payment_random + bids[i+1] > Parameters.TRAINING_PARAMETERS["budget"]:
                break
            else:
                while bids[i+1] + total_payment_random <= Parameters.TRAINING_PARAMETERS["budget"] and purchase_history_random[i] < len(train_data[i]):
                    selection_history_random.append(i)
                    if i == (Parameters.DATA_PARAMETERS["num_bids"]-1):
                        total_payment_random += bids[i]
                        payments_random.append(bids[i])
                    else:
                        total_payment_random += bids[i+1]
                        payments_random.append(bids[i+1])
                    
                    if purchase_history_random[i] + Parameters.DATA_PARAMETERS["length_per_epoch"] >= len(train_data[i]):
                        current_train_data_random.update_data(train_data[i][purchase_history_random[i]:], i)
                        purchase_history_random[i] = len(train_data[i])
                    else:
                        current_train_data_random.update_data(train_data[i][purchase_history_random[i]:purchase_history_random[i]+Parameters.DATA_PARAMETERS["length_per_epoch"]], i)
                        purchase_history_random[i] += Parameters.DATA_PARAMETERS["length_per_epoch"]

        for eps in range(Parameters.TRAINING_PARAMETERS["train_epochs"]):
            train_loss = train_epoch(moment, model_random, current_train_data_random, loss_function_random, optimizer_random, device)
            train_loss_history_random.append(train_loss)
            val_loss = validate_epoch(moment, model_random, val_dataset, loss_function_random, device)
            val_loss_history_random.append(val_loss)
        
        torch.cuda.empty_cache()
        
        plt.figure()
        plt.plot(train_loss_history_random, label='train_loss_random')
        plt.plot(val_loss_history_random, label='val_loss_random')
        plt.legend()
        plt.savefig(output_path + '/loss_random.png')
        plt.close()

        df_random_selection = pd.DataFrame()
        df_random_selection["selection_history"] = selection_history_random
        df_random_selection["payments"] = payments_random
        df_random_selection.to_csv(output_path + '/selection_random.csv')

        df_random_loss = pd.DataFrame()
        df_random_loss["train_loss"] = train_loss_history_random
        df_random_loss["val_loss"] = val_loss_history_random
        df_random_loss.to_csv(output_path + '/loss_random.csv')

        test_loss_random = validate_epoch(moment, model_random, test_dataset, loss_function_random, device)
        torch.save(model_random.state_dict(), output_path + '/model_random.pth')
        torch.save(current_train_data_random, output_path + '/current_train_data_random.pth')
        f.write(f"Length of random train loss history: {len(selection_history_random)}\n")
        f.write(f"Test loss of random: {test_loss_random}\n")
        print("Random model test loss: ", test_loss_random)

        model_random.to('cpu')
        print("Random allocation done.")

        # control_model = nn.Linear(1024, 2)
        # control_model.load_state_dict(model.state_dict())
        # control_model.to(device)
        # optimizer_control = optim.Adam(control_model.parameters(), lr=0.001)
        # loss_function_control = nn.MSELoss()

        # for eps in range(Parameters.TRAINING_PARAMETERS["train_epochs"]):
        #     train_loss = train_epoch(moment, control_model, val_dataset, loss_function_control, optimizer_control, device)

        # test_loss_control = validate_epoch(moment, control_model, test_dataset, loss_function_control, device)
        # print("Control model test loss: ", test_loss_control)

    # %%
    if wUCB:
        print("wUCB allocation...")
        model_wUCB = nn.Linear(1024, 2)
        model_wUCB.load_state_dict(model.state_dict())
        model_wUCB.to(device)
        optimizer_wUCB = optim.Adam(model_wUCB.parameters(), lr=0.001)
        loss_function_wUCB = nn.MSELoss()

        selection_history_wUCB = []
        total_payment_wUCB = 0
        payments_wUCB = []

        purchase_history_wUCB = np.full(Parameters.DATA_PARAMETERS["num_bids"], 0, dtype=int)
        current_train_data_wUCB = DataProcessor(Parameters.DATA_PARAMETERS["num_bids"])
        train_loss_history_wUCB = []
        val_loss_history_wUCB = []
        
        rewards_wUCB = []
        # unit_rewards_wUCB = np.zeros(Parameters.DATA_PARAMETERS["num_bids"])
        reward_history_wUCB = [[] for _ in range(Parameters.DATA_PARAMETERS["num_bids"])]
        order_history_wUCB = []

        num_exp_wUCB = 0

        ep_per_purchase = Parameters.TRAINING_PARAMETERS["epoch_per_purchase"]

        # we play each arm once at the beginning
        random_indices_wUCB_explore = np.random.permutation(Parameters.DATA_PARAMETERS["num_bids"])
        for idx in random_indices_wUCB_explore:
            selection_history_wUCB.append(idx)
            payments_wUCB.append(1)
            total_payment_wUCB += 1

            model_wUCB_current = nn.Linear(1024, 2)
            model_wUCB_current.load_state_dict(model_wUCB.state_dict())
            model_wUCB_current.to(device)
            optimizer_wUCB_current = optim.Adam(model_wUCB_current.parameters(), lr=0.001)
            loss_function_wUCB_current = nn.MSELoss()

            if num_exp_wUCB == 0:
                val_loss = validate_epoch(moment, model_wUCB_current, val_dataset, loss_function_wUCB_current, device)
                for _ in range(Parameters.TRAINING_PARAMETERS["epoch_per_purchase"]):
                    val_loss_history_wUCB.append(val_loss)
                    train_loss_history_wUCB.append(None)

            num_exp_wUCB += 1

            current_train_data_wUCB.update_data(train_data[idx][:Parameters.DATA_PARAMETERS["length_per_epoch"]], idx)
            purchase_history_wUCB[idx] += Parameters.DATA_PARAMETERS["length_per_epoch"]

            for _ in range(Parameters.TRAINING_PARAMETERS["epoch_per_purchase"]):
                train_loss_history_wUCB.append(train_epoch(moment, model_wUCB_current, current_train_data_wUCB, loss_function_wUCB_current, optimizer_wUCB_current, device))
                val_loss = validate_epoch(moment, model_wUCB_current, val_dataset, loss_function_wUCB_current, device)
                val_loss_history_wUCB.append(val_loss)

            current_loss_diff = val_loss_history_wUCB[-ep_per_purchase-1] - val_loss
            current_reward = current_loss_diff / (val_loss + 1e-8)

            reward_history_wUCB[idx].append(current_reward)
            rewards_wUCB.append(current_reward)
            order_history_wUCB.append([])
            torch.cuda.empty_cache()
        
        # for eps in range(Parameters.TRAINING_PARAMETERS["train_epochs"]):
        #     train_loss = train_epoch(moment, model_wUCB_current, current_train_data_wUCB, loss_function_wUCB, optimizer_wUCB, device)
        #     val_loss = validate_epoch(moment, model_wUCB_current, val_dataset, loss_function_wUCB, device)

        # test_loss = validate_epoch(moment, model_wUCB_current, test_dataset, loss_function_wUCB, device)
        # print("wUCB model test loss: ", test_loss)
        
        print("Initial exploration done.")

        avg_rewards_wUCB_list = []
        confidence_wUCB_list = []
        purchase_count_wUCB = np.ones(Parameters.DATA_PARAMETERS["num_bids"], dtype=int)
        while total_payment_wUCB < Parameters.TRAINING_PARAMETERS["budget"]: # budget is not used up
            avg_rewards_wUCB = [np.average(history) for history in reward_history_wUCB]
            avg_rewards_wUCB_list.append(avg_rewards_wUCB)
            # print("avg_rewards_wUCB: ", avg_rewards_wUCB)
            # print("reward_history_wUCB: ", reward_history_wUCB)
            epsilon_wUCB = Parameters.TRAINING_PARAMETERS["epsilon_wUCB"]
            confidence_wUCB = np.sqrt(epsilon_wUCB * np.log(num_exp_wUCB) / purchase_count_wUCB)
            confidence_wUCB_list.append(confidence_wUCB)
            # print("confidence_wUCB: ", confidence_wUCB)
            wUCB_values = avg_rewards_wUCB + confidence_wUCB
            # print("wUCB_values: ", wUCB_values)
            unit_wUCB_values = wUCB_values / bids
            # print("unit_wUCB_values: ", unit_wUCB_values)
            sorted_wUCB_indices = np.lexsort((bids, -unit_wUCB_values))
            # print("sorted_wUCB_indices: ", sorted_wUCB_indices)
            order_history_wUCB.append(sorted_wUCB_indices)
            exit_flag_wUCB = False
            
            while (not exit_flag_wUCB) and len(sorted_wUCB_indices) > 0:
                idx = sorted_wUCB_indices[0]
                if purchase_history_wUCB[idx] >= len(train_data[idx]):
                    sorted_wUCB_indices = np.delete(sorted_wUCB_indices, 0)
                    continue
                if len(sorted_wUCB_indices) == 1:
                    current_compensation = bids[idx]
                else:
                    if unit_wUCB_values[sorted_wUCB_indices[1]] <= 0:
                        current_compensation = bids[idx]
                    else:
                        current_compensation = max(bids[idx], wUCB_values[idx] / unit_wUCB_values[sorted_wUCB_indices[1]])

                if current_compensation + total_payment_wUCB > Parameters.TRAINING_PARAMETERS["budget"]:
                    exit_flag_wUCB = True
                    break
                else:
                    selection_history_wUCB.append(idx)
                    total_payment_wUCB += current_compensation
                    payments_wUCB.append(current_compensation)
                    num_exp_wUCB += 1
                    purchase_count_wUCB[idx] += 1

                    model_wUCB_current = nn.Linear(1024, 2)
                    model_wUCB_current.load_state_dict(model_wUCB.state_dict())
                    model_wUCB_current.to(device)
                    optimizer_wUCB_current = optim.Adam(model_wUCB_current.parameters(), lr=0.001)
                    loss_function_wUCB_current = nn.MSELoss()
                    
                    if purchase_history_wUCB[idx] + Parameters.DATA_PARAMETERS["length_per_epoch"] >= len(train_data[idx]):
                        current_train_data_wUCB.update_data(train_data[idx][purchase_history_wUCB[idx]:], idx)
                        purchase_history_wUCB[idx] = len(train_data[idx])
                    else:
                        current_train_data_wUCB.update_data(train_data[idx][purchase_history_wUCB[idx]:purchase_history_wUCB[idx]+Parameters.DATA_PARAMETERS["length_per_epoch"]], idx)
                        purchase_history_wUCB[idx] += Parameters.DATA_PARAMETERS["length_per_epoch"]
                    
                    for _ in range(Parameters.TRAINING_PARAMETERS["epoch_per_purchase"]):
                        train_loss_history_wUCB.append(train_epoch(moment, model_wUCB_current, current_train_data_wUCB, loss_function_wUCB_current, optimizer_wUCB_current, device))
                        val_loss_history_wUCB.append(validate_epoch(moment, model_wUCB_current, val_dataset, loss_function_wUCB_current, device))

                    current_loss_diff = val_loss_history_wUCB[-ep_per_purchase-1] - val_loss_history_wUCB[-1]
                    current_reward = current_loss_diff / (val_loss_history_wUCB[-1] + 1e-8)

                    reward_history_wUCB[idx].append(current_reward)
                    rewards_wUCB.append(current_reward)

                    torch.cuda.empty_cache()
                    break            
                
            if exit_flag_wUCB:
                break
        
        print("wUCB allocation done.")
        
        train_loss_final_wUCB = []
        val_loss_final_wUCB = []
        model_wUCB_final = nn.Linear(1024, 2)
        model_wUCB_final.load_state_dict(model_wUCB.state_dict())
        model_wUCB_final.to(device)
        optimizer_wUCB_final = optim.Adam(model_wUCB_final.parameters(), lr=0.001)
        loss_function_wUCB_final = nn.MSELoss()
    
        for _ in range(Parameters.TRAINING_PARAMETERS["train_epochs"]):
            train_loss_final_wUCB.append(train_epoch(moment, model_wUCB_final, current_train_data_wUCB, loss_function_wUCB_final, optimizer_wUCB_final, device))
            val_loss_final_wUCB.append(validate_epoch(moment, model_wUCB_final, val_dataset, loss_function_wUCB_final, device))

        torch.cuda.empty_cache()

        plt.figure()
        plt.plot(train_loss_final_wUCB, label='train_loss_wUCB')
        plt.plot(val_loss_final_wUCB, label='val_loss_wUCB')
        plt.legend()
        plt.savefig(output_path + '/loss_wUCB.png')
        plt.close()

        test_loss_wUCB = validate_epoch(moment, model_wUCB_final, test_dataset, loss_function_wUCB_final, device)
        torch.save(model_wUCB_final.state_dict(), output_path + '/model_wUCB.pth')
        torch.save(current_train_data_wUCB, output_path + '/current_train_data_wUCB.pth')
        f.write(f"number of selections: {len(selection_history_wUCB)}\n")
        f.write(f"Test loss of wUCB: {test_loss_wUCB}\n")
        model_wUCB_final.to('cpu')

        output_selection_history_wUCB = []
        output_payments_wUCB = []
        output_rewards_wUCB = []
        output_order_history_wUCB = []
        for selection, payment, reward, order in zip(selection_history_wUCB, payments_wUCB, rewards_wUCB, order_history_wUCB):
            output_selection_history_wUCB.extend([selection] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
            output_payments_wUCB.extend([payment] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
            output_rewards_wUCB.extend([reward] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
            output_order_history_wUCB.extend([order] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
        
        selection_df_wUCB = pd.DataFrame()
        selection_df_wUCB["selection_history"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_selection_history_wUCB
        selection_df_wUCB["payments"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_payments_wUCB
        selection_df_wUCB["rewards"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_rewards_wUCB
        selection_df_wUCB["order_history"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_order_history_wUCB
        selection_df_wUCB["train_loss"] = train_loss_history_wUCB
        selection_df_wUCB["val_loss"] = val_loss_history_wUCB
        selection_df_wUCB.to_csv(output_path + '/selection_wUCB.csv')

        df_wUCB_loss = pd.DataFrame()
        df_wUCB_loss["train_loss"] = train_loss_final_wUCB
        df_wUCB_loss["val_loss"] = val_loss_final_wUCB
        df_wUCB_loss.to_csv(output_path + '/loss_wUCB.csv')

        print("wUCB done.")
        
    # %%
    if rUCB:
        print("rUCB allocation...")
        model_rUCB = nn.Linear(1024, 2)
        model_rUCB.load_state_dict(model.state_dict())    
        model_rUCB.to(device)
        optimizer_rUCB = optim.Adam(model_rUCB.parameters(), lr=0.001)
        loss_function_rUCB = nn.MSELoss()

        selection_history_rUCB = []
        total_payment_rUCB = 0
        payments_rUCB = []

        # the index of the first unpurchased data in each sequence, or the length of purchase data
        purchase_history_rUCB = np.full(Parameters.DATA_PARAMETERS["num_bids"], 0, dtype=int)
        current_train_data_rUCB = DataProcessor(Parameters.DATA_PARAMETERS["num_bids"])
        train_loss_history_rUCB = []
        val_loss_history_rUCB = []
        
        rewards_rUCB = []
        # unit_rewards_UCB = np.zeros(Parameters.DATA_PARAMETERS["num_bids"])
        init_reward = Parameters.TRAINING_PARAMETERS["initial_reward_rUCB"]
        reward_history_rUCB = [[init_reward] for _ in range(Parameters.DATA_PARAMETERS["num_bids"])]
        order_history_rUCB = []

        num_exp_rUCB = Parameters.DATA_PARAMETERS["num_bids"]
        purchase_count_rUCB = np.ones(Parameters.DATA_PARAMETERS["num_bids"], dtype=int)

        ep_per_purchase = Parameters.TRAINING_PARAMETERS["epoch_per_purchase"]
        avg_rewards_rUCB_list = []
        confidence_rUCB_list = []

        while total_payment_rUCB < Parameters.TRAINING_PARAMETERS["budget"]: # budget is not used up
            avg_rewards_rUCB = [np.average(history) for history in reward_history_rUCB]
            avg_rewards_rUCB_list.append(avg_rewards_rUCB)
            epsilon_rUCB = Parameters.TRAINING_PARAMETERS["epsilon_rUCB"]
            confidence_rUCB = np.sqrt(epsilon_rUCB * np.log(num_exp_rUCB) / purchase_count_rUCB)
            confidence_rUCB_list.append(confidence_rUCB)

            rUCB_values = avg_rewards_rUCB + confidence_rUCB
            unit_rUCB_values = rUCB_values / bids
            sorted_rUCB_indices = np.lexsort((bids, -unit_rUCB_values))
            order_history_rUCB.append(sorted_rUCB_indices)
            exit_flag_rUCB = False
            
            while (not exit_flag_rUCB) and len(sorted_rUCB_indices) > 0:
                idx = sorted_rUCB_indices[0]
                if purchase_history_rUCB[idx] >= len(train_data[idx]):
                    sorted_rUCB_indices = np.delete(sorted_rUCB_indices, 0)
                    continue
                if len(sorted_rUCB_indices) == 1:
                    current_compensation = bids[idx]
                else:
                    if unit_rUCB_values[sorted_rUCB_indices[1]] <= 0:
                        current_compensation = bids[idx]
                    else:
                        current_compensation = max(bids[idx], rUCB_values[idx] / unit_rUCB_values[sorted_rUCB_indices[1]])

                if current_compensation + total_payment_rUCB > Parameters.TRAINING_PARAMETERS["budget"]:
                    exit_flag_rUCB = True
                    break
                else:
                    selection_history_rUCB.append(idx)
                    total_payment_rUCB += current_compensation
                    payments_rUCB.append(current_compensation)
                    num_exp_rUCB += 1
                    purchase_count_rUCB[idx] += 1

                    model_rUCB_current = nn.Linear(1024, 2)
                    model_rUCB_current.load_state_dict(model_rUCB.state_dict())
                    model_rUCB_current.to(device)
                    optimizer_rUCB_current = optim.Adam(model_rUCB_current.parameters(), lr=0.001)
                    loss_function_rUCB_current = nn.MSELoss()

                    if num_exp_rUCB == Parameters.DATA_PARAMETERS["num_bids"] + 1:
                        val_loss = validate_epoch(moment, model_rUCB_current, val_dataset, loss_function_rUCB_current, device)
                        for _ in range(Parameters.TRAINING_PARAMETERS["epoch_per_purchase"]):
                            val_loss_history_rUCB.append(val_loss)
                            train_loss_history_rUCB.append(None)

                    
                    if purchase_history_rUCB[idx] + Parameters.DATA_PARAMETERS["length_per_epoch"] >= len(train_data[idx]):
                        current_train_data_rUCB.update_data(train_data[idx][purchase_history_rUCB[idx]:], idx)
                        purchase_history_rUCB[idx] = len(train_data[idx])
                    else:
                        current_train_data_rUCB.update_data(train_data[idx][purchase_history_rUCB[idx]:purchase_history_rUCB[idx]+Parameters.DATA_PARAMETERS["length_per_epoch"]], idx)
                        purchase_history_rUCB[idx] += Parameters.DATA_PARAMETERS["length_per_epoch"]
                    
                    for _ in range(Parameters.TRAINING_PARAMETERS["epoch_per_purchase"]):
                        train_loss_history_rUCB.append(train_epoch(moment, model_rUCB_current, current_train_data_rUCB, loss_function_rUCB_current, optimizer_rUCB_current, device))
                        val_loss = validate_epoch(moment, model_rUCB_current, val_dataset, loss_function_rUCB_current, device)
                        val_loss_history_rUCB.append(val_loss)
                    
                    current_loss_diff = val_loss_history_rUCB[-ep_per_purchase-1] - val_loss
                    current_reward = current_loss_diff / (val_loss + 1e-8)
                    reward_history_rUCB[idx].append(current_reward)
                    rewards_rUCB.append(current_reward)

                    torch.cuda.empty_cache()
                    break            

            if exit_flag_rUCB:
                break
        
        print("rUCB allocation done.")

        train_loss_final_rUCB = []
        val_loss_final_rUCB = []
        model_rUCB_final = nn.Linear(1024, 2)
        model_rUCB_final.load_state_dict(model_rUCB.state_dict())
        model_rUCB_final.to(device)
        optimizer_rUCB_final = optim.Adam(model_rUCB_final.parameters(), lr=0.001)
        loss_function_rUCB_final = nn.MSELoss()

        for _ in range(Parameters.TRAINING_PARAMETERS["train_epochs"]):
            train_loss_final_rUCB.append(train_epoch(moment, model_rUCB_final, current_train_data_rUCB, loss_function_rUCB_final, optimizer_rUCB_final, device))
            val_loss_final_rUCB.append(validate_epoch(moment, model_rUCB_final, val_dataset, loss_function_rUCB_final, device))
        
        torch.cuda.empty_cache()

        plt.figure()
        plt.plot(train_loss_final_rUCB, label='train_loss_rUCB')
        plt.plot(val_loss_final_rUCB, label='val_loss_rUCB')
        plt.legend()
        plt.savefig(output_path + '/loss_rUCB.png')
        plt.close()

        test_loss_rUCB = validate_epoch(moment, model_rUCB_final, test_dataset, loss_function_rUCB_final, device)
        torch.save(model_rUCB_final.state_dict(), output_path + '/model_rUCB.pth')
        torch.save(current_train_data_rUCB, output_path + '/current_train_data_rUCB.pth')
        tmp = len(selection_history_rUCB)-Parameters.DATA_PARAMETERS["num_bids"]
        f.write(f"number of selections: {tmp}\n")
        f.write(f"Test loss of rUCB: {test_loss_rUCB}\n")
        print("rUCB model test loss: ", test_loss_rUCB)
        model_rUCB.to('cpu')

        output_selection_history_rUCB = []
        output_payments_rUCB = []
        output_rewards_rUCB = []
        output_order_history_rUCB = []
        for selection, payment, reward, order in zip(selection_history_rUCB, payments_rUCB, rewards_rUCB, order_history_rUCB):
            output_selection_history_rUCB.extend([selection] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
            output_payments_rUCB.extend([payment] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
            output_rewards_rUCB.extend([reward] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
            output_order_history_rUCB.extend([order] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
        
        selection_df_rUCB = pd.DataFrame()
        selection_df_rUCB["selection_history"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_selection_history_rUCB
        selection_df_rUCB["payments"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_payments_rUCB
        selection_df_rUCB["rewards"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_rewards_rUCB
        selection_df_rUCB["order_history"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_order_history_rUCB
        selection_df_rUCB["train_loss"] = train_loss_history_rUCB
        selection_df_rUCB["val_loss"] = val_loss_history_rUCB
        selection_df_rUCB.to_csv(output_path + '/selection_rUCB.csv')

        df_rUCB_loss = pd.DataFrame()
        df_rUCB_loss["train_loss"] = train_loss_final_rUCB
        df_rUCB_loss["val_loss"] = val_loss_final_rUCB
        df_rUCB_loss.to_csv(output_path + '/loss_rUCB.csv')

        print("rUCB done.")

    # %%
    if LinUCB:
        print("LinUCB allocation...")
        model_LinUCB = nn.Linear(1024, 2)
        model_LinUCB.load_state_dict(model.state_dict())
        model_LinUCB.to(device)
        optimizer_LinUCB = optim.Adam(model_LinUCB.parameters(), lr=0.001)
        loss_function_LinUCB = nn.MSELoss()

        selection_history_LinUCB = []
        total_payment_LinUCB = 0
        payments_LinUCB = []

        purchase_history_LinUCB = np.full(Parameters.DATA_PARAMETERS["num_bids"], 0, dtype=int)
        current_train_data_LinUCB = DataProcessor(Parameters.DATA_PARAMETERS["num_bids"])
        train_loss_history_LinUCB = []
        val_loss_history_LinUCB = []

        rewards_LinUCB = []
        reward_history_LinUCB = [[] for _ in range(Parameters.DATA_PARAMETERS["num_bids"])]
        order_history_LinUCB = []
        whole_current_context = torch.zeros(Parameters.DATA_PARAMETERS["num_bids"], 1024, dtype=torch.float32).to(device)

        num_exp_LinUCB = 0
        LinUCB_d = 1024
        LinUCB_A = torch.eye(LinUCB_d).to(device)
        LinUCB_b = torch.zeros((LinUCB_d, 1)).to(device)

        ep_per_purchase = Parameters.TRAINING_PARAMETERS["epoch_per_purchase"]
        loss_context_length = Parameters.TRAINING_PARAMETERS["context_length_LinUCB"]

        # we play each arm once at the beginning
        random_indices_LinUCB_explore = np.random.permutation(Parameters.DATA_PARAMETERS["num_bids"])
        for idx in random_indices_LinUCB_explore:
            selection_history_LinUCB.append(idx)
            payments_LinUCB.append(1)
            total_payment_LinUCB += 1

            model_LinUCB_current = nn.Linear(1024, 2)
            model_LinUCB_current.load_state_dict(model_LinUCB.state_dict())
            model_LinUCB_current.to(device)
            optimizer_LinUCB_current = optim.Adam(model_LinUCB_current.parameters(), lr=0.001)
            loss_function_LinUCB_current = nn.MSELoss()

            if num_exp_LinUCB == 0:
                val_loss = validate_epoch(moment, model_LinUCB_current, val_dataset, loss_function_LinUCB_current, device)
                for _ in range(Parameters.TRAINING_PARAMETERS["epoch_per_purchase"]):
                    val_loss_history_LinUCB.append(val_loss)
                    train_loss_history_LinUCB.append(None)

            num_exp_LinUCB += 1
            
            current_train_data_LinUCB.update_data(train_data[idx][:Parameters.DATA_PARAMETERS["length_per_epoch"]], idx)
            purchase_history_LinUCB[idx] += Parameters.DATA_PARAMETERS["length_per_epoch"]

            data = torch.tensor(train_data[idx][:purchase_history_LinUCB[idx]]).to(device)
            data = data.reshape(1, 2, len(data))
            embeddings = moment(data).embeddings
            current_context = embeddings / torch.norm(embeddings)
            whole_current_context[idx] = current_context
            current_context = current_context.reshape(1024, 1)

            for _ in range(ep_per_purchase):
                train_loss_history_LinUCB.append(train_epoch(moment, model_LinUCB_current, current_train_data_LinUCB, loss_function_LinUCB_current, optimizer_LinUCB_current, device))
                val_loss = validate_epoch(moment, model_LinUCB_current, val_dataset, loss_function_LinUCB_current, device)
                val_loss_history_LinUCB.append(val_loss)

            current_loss_diff = val_loss_history_LinUCB[-ep_per_purchase-1] - val_loss
            current_reward = current_loss_diff / (val_loss + 1e-8)

            reward_history_LinUCB[idx].append(current_reward)
            rewards_LinUCB.append(current_reward)
            order_history_LinUCB.append([])

            LinUCB_A += torch.matmul(current_context, current_context.T)
            LinUCB_b += current_reward * current_context

            torch.cuda.empty_cache()
        
        print("Initial exploration done.")

        while total_payment_LinUCB < Parameters.TRAINING_PARAMETERS["budget"]:  # budget is not used up
            # compute LinUCB values
            LinUCB_A = LinUCB_A.float()
            LinUCB_b = LinUCB_b.float()
            inverse_LinUCB_A = torch.inverse(LinUCB_A)            
            LinUCB_theta = torch.matmul(inverse_LinUCB_A, LinUCB_b)
            LinUCB_p_left = torch.matmul(whole_current_context, LinUCB_theta)
            LinUCB_p_right = torch.diag(torch.matmul(torch.matmul(whole_current_context, inverse_LinUCB_A), whole_current_context.T))
            
            epsilon_LinUCB = Parameters.TRAINING_PARAMETERS["epsilon_LinUCB"]
            LinUCB_p = LinUCB_p_left + epsilon_LinUCB * torch.sqrt(LinUCB_p_right).unsqueeze(1)
            # print("LinUCB_p: ", LinUCB_p)
            unit_LinUCB_p = LinUCB_p.flatten().to('cpu') / bids
            # print("unit_LinUCB_p: ", unit_LinUCB_p)

            sorted_LinUCB_indices = np.lexsort((bids, -unit_LinUCB_p.cpu().numpy())).flatten()
            order_history_LinUCB.append(sorted_LinUCB_indices)

            exit_flag_LinUCB = False

            while (not exit_flag_LinUCB) and len(sorted_LinUCB_indices) > 0:
                idx = sorted_LinUCB_indices[0]
                if purchase_history_LinUCB[idx] >= len(train_data[idx]):
                    sorted_LinUCB_indices = np.delete(sorted_LinUCB_indices, 0)
                    continue
                if len(sorted_LinUCB_indices) == 1:
                    current_compensation = bids[idx]
                else:
                    if unit_LinUCB_p[sorted_LinUCB_indices[1]] == 0:
                        current_compensation = bids[idx]
                    else:
                        current_compensation = max(bids[idx], unit_LinUCB_p[idx] * bids[idx] / unit_LinUCB_p[sorted_LinUCB_indices[1]])
                
                if current_compensation + total_payment_LinUCB > Parameters.TRAINING_PARAMETERS["budget"]:
                    exit_flag_LinUCB = True
                    break
                else:
                    selection_history_LinUCB.append(idx)
                    total_payment_LinUCB += current_compensation
                    payments_LinUCB.append(float(current_compensation))
                    num_exp_LinUCB += 1

                    model_LinUCB_current = nn.Linear(1024, 2)
                    model_LinUCB_current.load_state_dict(model_LinUCB.state_dict())
                    model_LinUCB_current.to(device)
                    optimizer_LinUCB_current = optim.Adam(model_LinUCB_current.parameters(), lr=0.001)
                    loss_function_LinUCB_current = nn.MSELoss()
                    
                    if purchase_history_LinUCB[idx] + Parameters.DATA_PARAMETERS["length_per_epoch"] >= len(train_data[idx]):
                        current_train_data_LinUCB.update_data(train_data[idx][purchase_history_LinUCB[idx]:], idx)
                        purchase_history_LinUCB[idx] = len(train_data[idx])
                    else:
                        current_train_data_LinUCB.update_data(train_data[idx][purchase_history_LinUCB[idx]:purchase_history_LinUCB[idx]+Parameters.DATA_PARAMETERS["length_per_epoch"]], idx)
                        purchase_history_LinUCB[idx] += Parameters.DATA_PARAMETERS["length_per_epoch"]

                    data = torch.tensor(train_data[idx][:purchase_history_LinUCB[idx]]).to(device)
                    data = data.reshape(1, 2, purchase_history_LinUCB[idx])
                    embeddings = moment(data).embeddings
                    current_context = embeddings / torch.norm(embeddings)
                    whole_current_context[idx] = current_context
                    current_context = current_context.reshape(1024, 1)

                    for _ in range(Parameters.TRAINING_PARAMETERS["epoch_per_purchase"]):
                        train_loss_history_LinUCB.append(train_epoch(moment, model_LinUCB_current, current_train_data_LinUCB, loss_function_LinUCB_current, optimizer_LinUCB_current, device))
                        val_loss = validate_epoch(moment, model_LinUCB_current, val_dataset, loss_function_LinUCB_current, device)
                        val_loss_history_LinUCB.append(val_loss)
                    
                    current_loss_diff = val_loss_history_LinUCB[-ep_per_purchase-1] - val_loss
                    current_reward = current_loss_diff / (val_loss + 1e-8)

                    reward_history_LinUCB[idx].append(current_reward)
                    rewards_LinUCB.append(current_reward)
                    
                    # current_sum_reward = np.sum(reward_history_LinUCB[idx][-Parameters.TRAINING_PARAMETERS["epoch_per_purchase"]:])
                    LinUCB_A += torch.matmul(current_context, current_context.T)
                    LinUCB_b += current_reward * current_context

                    torch.cuda.empty_cache()
                    break
            
            if exit_flag_LinUCB:
                break
        
        print("LinUCB allocation done.")

        train_loss_final_LinUCB = []
        val_loss_final_LinUCB = []
        model_LinUCB_final = nn.Linear(1024, 2)
        model_LinUCB_final.load_state_dict(model_LinUCB.state_dict())
        model_LinUCB_final.to(device)
        optimizer_LinUCB_final = optim.Adam(model_LinUCB_final.parameters(), lr=0.001)
        loss_function_LinUCB_final = nn.MSELoss()

        for _ in range(Parameters.TRAINING_PARAMETERS["train_epochs"]):
            train_loss_final_LinUCB.append(train_epoch(moment, model_LinUCB_final, current_train_data_LinUCB, loss_function_LinUCB_final, optimizer_LinUCB_final, device))
            val_loss_final_LinUCB.append(validate_epoch(moment, model_LinUCB_final, val_dataset, loss_function_LinUCB_final, device))

        torch.cuda.empty_cache()

        plt.figure()
        plt.plot(train_loss_final_LinUCB, label='train_loss_LinUCB')
        plt.plot(val_loss_final_LinUCB, label='val_loss_LinUCB')
        plt.legend()
        plt.savefig(output_path + '/loss_LinUCB.png')
        plt.close()
        
        test_loss_LinUCB = validate_epoch(moment, model_LinUCB_final, test_dataset, loss_function_LinUCB_final, device)
        torch.save(model_LinUCB_final.state_dict(), output_path + '/model_LinUCB.pth')
        torch.save(current_train_data_LinUCB, output_path + '/current_train_data_LinUCB.pth')
        f.write(f"number of selections: {len(selection_history_LinUCB)}\n")
        f.write(f"Test loss of LinUCB: {test_loss_LinUCB}\n")
        print("LinUCB model test loss: ", test_loss_LinUCB)
        model_LinUCB_final.to('cpu')

        output_selection_history_LinUCB = []
        output_payments_LinUCB = []
        output_rewards_LinUCB = []
        output_order_history_LinUCB = []
        for selection, payment, reward, order in zip(selection_history_LinUCB, payments_LinUCB, rewards_LinUCB, order_history_LinUCB):
            output_selection_history_LinUCB.extend([selection] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
            output_payments_LinUCB.extend([payment] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
            output_rewards_LinUCB.extend([reward] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
            output_order_history_LinUCB.extend([order] + [None] * (Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] - 1))
        
        selection_df_LinUCB = pd.DataFrame()
        selection_df_LinUCB["selection_history"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_selection_history_LinUCB
        selection_df_LinUCB["payments"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_payments_LinUCB
        selection_df_LinUCB["rewards"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_rewards_LinUCB
        selection_df_LinUCB["order_history"] = [None] * Parameters.TRAINING_PARAMETERS["epoch_per_purchase"] + output_order_history_LinUCB
        selection_df_LinUCB["train_loss"] = train_loss_history_LinUCB
        selection_df_LinUCB["val_loss"] = val_loss_history_LinUCB
        selection_df_LinUCB.to_csv(output_path + '/selection_LinUCB.csv')

        df_LinUCB_loss = pd.DataFrame()
        df_LinUCB_loss["train_loss"] = train_loss_final_LinUCB
        df_LinUCB_loss["val_loss"] = val_loss_final_LinUCB
        df_LinUCB_loss.to_csv(output_path + '/loss_LinUCB.csv')

        print("LinUCB done.")

    # %%

    df_final_loss = pd.DataFrame()
    df_final_loss["train_loss_greedy"] = train_loss_history_random
    df_final_loss["val_loss_greedy"] = val_loss_history_random
    df_final_loss["train_loss_UCB"] = train_loss_final_wUCB
    df_final_loss["val_loss_UCB"] = val_loss_final_wUCB
    df_final_loss["train_loss_rUCB"] = train_loss_final_rUCB
    df_final_loss["val_loss_rUCB"] = val_loss_final_rUCB
    df_final_loss["train_loss_LinUCB"] = train_loss_final_LinUCB
    df_final_loss["val_loss_LinUCB"] = val_loss_final_LinUCB
    df_final_loss.to_csv(output_path + '/final_loss.csv')

    f.close()

    plt.figure()
    plt.plot(np.log(train_loss_history_random), label='Greedy')
    plt.plot(np.log(train_loss_final_wUCB), label='UCB')
    plt.plot(np.log(train_loss_final_rUCB), label='ReverseUCB')
    plt.plot(np.log(train_loss_final_LinUCB), label='LinUCB')
    plt.legend()
    plt.savefig(output_path + '/log_train_loss.png')
    plt.close()

    plt.figure()
    plt.plot(np.log(val_loss_history_random), label='Greedy')
    plt.plot(np.log(val_loss_final_wUCB), label='UCB')
    plt.plot(np.log(val_loss_final_rUCB), label='ReverseUCB')
    plt.plot(np.log(val_loss_final_LinUCB), label='LinUCB')
    plt.legend()
    plt.savefig(output_path + '/log_val_loss.png')
    plt.close()


set_seed(777)

if torch.cuda.is_available():
    print("CUDA is available. Here are the GPU details:")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
else:
    print("CUDA is not available.")

for rep in range(5):
    print(f"Experiment {rep} started.")
    experiment(rep)


print("Done.")


