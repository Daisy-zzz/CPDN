from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import random
import pickle
import numpy as np
import pandas as pd

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import MSELoss
from models.model_test import MODEL
from global_configs import DEVICE, SEQ_LEN

parser = argparse.ArgumentParser()
#parser.add_argument("--dataset", type=str, default="smpd_447")
parser.add_argument("--train_batch_size", type=int, default=512)
parser.add_argument("--dev_batch_size", type=int, default=2048)
parser.add_argument("--test_batch_size", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--seq_length", type=int, default=SEQ_LEN)
parser.add_argument("--seed", type=int, default=5576)

args = parser.parse_args()

def max_min(data):
    data = torch.tensor(data, dtype=torch.float).unsqueeze(1)
    data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    return data

def norm(data):
    data = torch.tensor(data, dtype=torch.float).unsqueeze(1)
    std, mean = torch.std_mean(data, dim=0, keepdim=True)
    norm_data = (data - mean) / std
    return norm_data

def get_appropriate_dataset():
    """
    :return:
        VISUAL:     N * 512
        TEXT:       N * 512
        CATEGORY:   N * 3
        USER:       N * 1
        LABEL       N * 1
    """
    # 加载提取好的图像文本特征
    visual = torch.tensor(np.load("./dataset/features/image_clip.npy"), dtype=torch.float)
    text = torch.tensor(np.load("./dataset/features/text_clip.npy"), dtype=torch.float)
    # user_description = torch.tensor(np.load("./dataset/features/user_description.npy"), dtype=torch.float)
    # location = torch.tensor(np.load("./dataset/features/location.npy"), dtype=torch.float)
    # 处理其他特征
    uid_json = open("./dataset/uid.json")
    cat_json = open("./dataset/category.json")
    subcat_json = open("./dataset/subcategory.json")
    concept_json = open("./dataset/concept.json")
    uid_dict = json.load(uid_json)
    cat_dict = json.load(cat_json)
    subcat_dict = json.load(subcat_json)
    concept_dict = json.load(concept_json)
    data = pd.read_csv("dataset/train_data_1.csv")
    id_arr = list(data['id'])
    vuid_arr = list(data['vuid'])
    category_arr = list(data['category'])
    subcategory_arr = list(data['subcategory'])
    concept_arr = list(data['concept'])
    latitude_arr = list(data['latitude'])
    longitude_arr = list(data['longitude'])
    geoacc_arr = list(data['geoacc'])
    time_arr = list(data['time'])
    text_arr = list(data['text'])
    tags_arr = list(data['tags'])
    tags_len = [len(str(l)) for l in tags_arr]
    #additional data
    additional_data = pd.read_csv("dataset/train_data_additional.csv")
    totalview = list(additional_data["totalViews"])
    totaltag = list(additional_data["totalTags"])
    totalgeotag = list(additional_data["totalGeotagged"])
    totalfave = list(additional_data["totalFaves"])
    totalingroup = list(additional_data["totalInGroup"])
    photocount = list(additional_data["photoCount"])
    followercount = list(additional_data["followerCount"])
    followingcount = list(additional_data["followingCount"])
    totalview = norm(totalview)
    totaltag = norm(totaltag)
    totalgeotag = norm(totalgeotag)
    totalfave = norm(totalfave)
    totalingroup = norm(totalingroup)
    photocount = norm(photocount)
    followercount = norm(followercount)
    followingcount = norm(followingcount)
    additional_info = torch.cat([totalview, totaltag, totalgeotag, totalfave, totalingroup, photocount, followercount, followingcount], dim=1)
    # time dim = 43
    import time
    timel = []
    for t in time_arr:
        timestamp = int(t)
        timeArray = time.localtime(timestamp)
        m = int(time.strftime("%m", timeArray))
        h = int(time.strftime("%H", timeArray))
        w = int(time.strftime("%w", timeArray))
        # month hour weekday
        timel.append([m, w, h])
    from sklearn.preprocessing import OneHotEncoder
    month = list(range(1, 13))
    hour = list(range(0, 24))
    weekday = list(range(0, 7))
    one_hot = OneHotEncoder(categories=[month, weekday, hour])
    one_hot.fit(timel)
    time_data = torch.tensor(one_hot.transform(timel).toarray(), dtype=torch.float)

    ispublic_arr = list(data['ispublic'])
    ispro_arr = list(data['ispro'])
    ispro = torch.tensor(ispro_arr, dtype=torch.float).unsqueeze(1)
    ispublic = torch.tensor(ispublic_arr, dtype=torch.float).unsqueeze(1)
    latitude = norm(latitude_arr)
    longitude = norm(longitude_arr)
    geoacc = norm(geoacc_arr)
    tags = norm(tags_len)
    #user_info = torch.cat((ispublic, latitude, longitude, geoacc, time_data, ispro), dim=1)
    user_info = torch.cat((ispublic, latitude, longitude, geoacc, time_data, ispro, additional_info), dim=1)
    #user_info = time_data
    uid, cat, subcat, concept = [], [], [], []
    num_data = len(id_arr)
    for i in range(0, num_data):
        uid.append(uid_dict[vuid_arr[i].split('/')[0]])
        cat.append(cat_dict[category_arr[i]])
        subcat.append(subcat_dict[subcategory_arr[i]])
        concept.append(concept_dict[concept_arr[i]])
    uid = torch.tensor(uid, dtype=torch.long).unsqueeze(1)
    category = torch.tensor([cat, subcat, concept], dtype=torch.long).transpose(0, 1)
    label = torch.tensor(np.expand_dims(np.asarray(list(data['label'])), axis=1), dtype=torch.float)
    user = torch.cat((uid, user_info), dim=1)
    # N * L * DIM
    visual_seq = []
    text_seq = []
    category_seq = []
    user_seq = []
    seq_len = args.seq_length
    for i in range(0, num_data):
        if i < seq_len - 1:
            visual_entry = torch.cat((torch.zeros((seq_len - 1 - i, visual.shape[1])), visual[:i + 1, :]), dim=0)
            text_entry = torch.cat((torch.zeros((seq_len - 1 - i, text.shape[1])), text[:i + 1, :]), dim=0)
            category_entry = torch.cat((torch.zeros((seq_len - 1 - i, category.shape[1]), dtype=torch.long), category[:i + 1, :]), dim=0)
            user_entry = torch.cat((torch.zeros((seq_len - 1 - i, user.shape[1]), dtype=torch.long), user[:i + 1, :]), dim=0)
        else:
            visual_entry = visual[i - seq_len + 1: i + 1, :]
            text_entry = text[i - seq_len + 1: i + 1, :]
            category_entry = category[i - seq_len + 1: i + 1, :]
            user_entry = user[i - seq_len + 1: i + 1, :]
        visual_seq.append(visual_entry)
        text_seq.append(text_entry)
        category_seq.append(category_entry)
        user_seq.append(user_entry)

    visual = torch.stack(visual_seq, dim=0)
    text = torch.stack(text_seq, dim=0)
    category = torch.stack(category_seq, dim=0)
    user = torch.stack(user_seq, dim=0)

    #idx = torch.randperm(visual.shape[0])
    #visual = visual[idx,:,:].view(visual.size())
    #text = text[idx,:,:].view(text.size())
    #category = category[idx,:,:].view(category.size())
    #user = user[idx,:,:].view(user.size())
    #label = label[idx,:,:].view(label.size())
    split = int(num_data / 10)
    train_split = int(split * 8)
    dev_split = int(split * 9)

    train_dataset = TensorDataset(
        visual[0: train_split, :, :],
        text[0: train_split, :, :],
        category[0: train_split, :, :],
        user[0: train_split, :, :],
        label[0: train_split, :],
    )

    dev_dataset = TensorDataset(
        visual[train_split: dev_split, :, :],
        text[train_split: dev_split, :, :],
        category[train_split: dev_split, :, :],
        user[train_split: dev_split, :, :],
        label[train_split: dev_split, :],
    )

    test_dataset = TensorDataset(
        visual[dev_split:, :, :],
        text[dev_split:, :, :],
        category[dev_split:, :, :],
        user[dev_split:, :, :],
        label[dev_split:, :],
    )
    print(dev_split)
    return train_dataset, dev_dataset, test_dataset


def set_up_data_loader():
    train_dataset, dev_dataset, test_dataset = get_appropriate_dataset()

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """

    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training():
    model = MODEL()
    #model.load_state_dict(torch.load('checkpoints/model-1.249-0.736.pth'))
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    total_para = 0
    for param in model.parameters():
        total_para += np.prod(param.size())
    print('total parameter for the model: ', total_para)

    model.to(DEVICE)

    return model


def train_epoch(model: nn.Module, optimizer, train_dataloader: DataLoader):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        visual, text, category, user, label = batch
        loss, outputs = model(
            visual,
            text,
            category,
            user,
            label
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1

    return tr_loss / nb_tr_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    attn_w = 0
    idx = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            visual, text, category, user, label = batch
            outputs, attn = model.test(
                visual,
                text,
                category,
                user
            )
            label = label.squeeze(1)
            logits = outputs
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label.view(-1))


            logits = logits.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label = np.squeeze(label).tolist()

            preds.extend(logits)
            labels.extend(label)

            dev_loss += loss.item()
            nb_dev_steps += 1
            if idx == 0:
                attn_w = attn.cpu().numpy()
            idx += 1
        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels, dev_loss / nb_dev_steps, attn_w


def test_score_model(model: nn.Module, test_dataloader: DataLoader):
    preds, y_test, loss, attn_w = test_epoch(model, test_dataloader)
    preds = preds.reshape(-1, )
    y_test = y_test.reshape(-1, )
    # non_zeros = np.array(
    #     [i for i, e in enumerate(y_test) if e != 0 or use_zero])
    #
    # preds = preds[non_zeros]
    # y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    return mae, corr, loss, preds, y_test, attn_w

def param_groups_weight_decay(model: nn.Module, weight_decay=1e-4):
    decay = []
    no_decay = []
    no_decay_name = ["bias", "norm.weight"]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not any(nd in name for nd in no_decay_name):
            #print(name)
            decay.append(param)
        else:
            print(name)
            no_decay.append(param)

    return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]


def train(
        model,
        train_dataloader,
        dev_dataloader,
        test_data_loader
):
    test_losses = []
    best_loss = 10
    best_mae = 0
    best_corr = 0
    best_model = 0
    best_preds = 0
    best_label = 0
    best_attn = 0
    param_groups = param_groups_weight_decay(model, 1e-4)
    optimizer = torch.optim.Adam(param_groups, lr=args.learning_rate)
    #optimizer.add_param_group(param_groups)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7], gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer=optimizer, lr_lambda=lambda step: rate(step, *[256, 1, 80])
    # )
    for epoch_i in range(1, int(args.n_epochs) + 1):
        train_loss = train_epoch(model, optimizer, train_dataloader)
        _, _, dev_loss, _ = test_epoch(model, dev_dataloader)
        #lr_scheduler.step()
        #scheduler.step()
        test_mae, test_corr, test_loss, preds, label, attn_w = test_score_model(
            model, test_data_loader
        )

        print(
            "epoch:{}, train_loss:{:.4f}, valid_loss:{:.4f}, test_loss:{:.4f}".format(
                epoch_i, train_loss, dev_loss, test_loss
            )
        )

        print(
            "current mae:{:.4f}, corr:{:.4f}".format(
                test_mae, test_corr
            )
        )

        test_losses.append(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_mae = test_mae
            best_corr = test_corr
            best_model = model.state_dict()
            best_preds = preds
            best_label = label
            best_attn = attn_w
            #if best_loss < 3.5:
                #torch.save(model.state_dict(), "checkpoints/model-{mae}-{corr}.pth".format(mae=test_mae, corr=test_corr))
        print(
            "best mae:{:.4f}, corr:{:.4f}".format(
                best_mae, best_corr
            )
        )

        # wandb.log(
        #     (
        #         {
        #             "train_loss": train_loss,
        #             "valid_loss": valid_loss,
        #             "test_mae": test_mae,
        #             "test_corr": test_corr,
        #             "best_valid_loss": min(valid_losses),
        #         }
        #     )
        # )
    torch.save(best_model, "checkpoints/model-{mae}-{corr}.pth".format(mae=round(best_mae, 3), corr=round(best_corr, 3)))

    return best_mae, best_corr, best_preds, best_label, best_attn

def main():
    # wandb.init(project="SMP")
    # wandb.config.update(args)
    #for seed in range(0, 100):
    set_random_seed(args.seed)
    # for beta in [0.2, 0.4, 0.6, 0.8]:
    #     for alpha in [0.0, 0.2, 0.4, 0.6, 0.8]:
    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
    ) = set_up_data_loader()

    model = prep_for_training()   
    mae, corr, preds, label, attn = train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader
    )
    np.save('attn.npy', attn)
    preds_str = [str(p) for p in list(preds)]
    label_str = [str(l) for l in list(label)]
    with open("dataset/pred.txt", "a+") as f:
        f.write(','.join(preds_str))
    with open("dataset/label.txt", "a+") as f:
        f.write(','.join(label_str))
    # with open("dataset/result.txt", "a+") as f:
    #     f.writelines("alpha:{alpha}, beta:{beta}, mae: {mae}, corr: {corr}\n".format(alpha=alpha, beta=beta, mae=round(mae, 3), corr=round(corr, 3)))

if __name__ == "__main__":
    main()
