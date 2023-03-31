import pandas as pd
import numpy as np

data = pd.read_csv("dataset/train_data.csv")
id_arr = list(data['id'])
vuid_arr = list(data['vuid'])
time_arr = list(data['time'])
category_arr = list(data['category'])
subcategory_arr = list(data['subcategory'])
concept_arr = list(data['concept'])
text_arr = list(data['text'])
tags_arr = list(data['tags'])
ispublic_arr = list(data['ispublic'])
label_arr = list(data['label'])



img_feature = np.load("./dataset/image.npy")
text_feature = np.load("./dataset/text.npy")



# user data (305613, 846)
# fill_loc = "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"
# with open("dataset/train_allmetadata_json/train_user_data.json") as f:
#     train_user = json.load(f)
#     idx = 0
#     user_data = []
#     timel = []
#     countl = []
#     tagsl = []
#     for vuid, v in cat_dic.items():
#         # uid str to id int
#         uid = vuid.split('/')[0]
#         user_dict = train_user[idx]
#         img_path = "./dataset/train_images/" + vuid + '.jpg'
#         # location dim = 400
#         loc_str = ''
#         if user_dict['location_description'] == "\n":
#             loc_str = fill_loc
#         else:
#             loc_str = user_dict['location_description'].strip()
#         loc_str_list = loc_str.split(',')
#         loc_list = []
#         for i in loc_str_list:
#             loc_list.append(float(i))
#         # time
#         timestamp = int(ts_dict[vuid])
#         timeArray = time.localtime(timestamp)
#         m = int(time.strftime("%m", timeArray))
#         h = int(time.strftime("%H", timeArray))
#         w = int(time.strftime("%w", timeArray))
#         timel.append([m, h, w])
#         # user dim = 399
#         user_str_list = user_dict['user_description'].split(',')
#         user_list = []
#         for i in user_str_list:
#             user_list.append(float(i))
#         # assert len(user_list) == 399 and len(loc_list) == 400
#         # other
#         ispro = int(user_dict['ispro'])
#         photo_count = int(user_dict['photo_count'])
#         num_tags = len(tags_dic.get(vuid).split(' '))
#         ispublic = int(add_dic.get(vuid))
#         user_line = []
#         user_line.append(uid_dict[uid])
#         # user_line.extend(loc_list)
#         user_line.extend(user_list)
#         user_line.append(ispro)
#         user_line.append(ispublic)
#         countl.append(photo_count)
#         tagsl.append(num_tags)
#         user_data.append(user_line)
#         idx += 1
#     from sklearn.preprocessing import OneHotEncoder
#     month = list(range(1, 13))
#     hour = list(range(0, 24))
#     weekday = list(range(0, 7))
#     one_hot = OneHotEncoder(categories=[month, hour, weekday])
#     one_hot.fit(timel)
#     time_data = one_hot.transform(timel).toarray()
#     max_count, min_count = max(countl), min(countl) # 2978819 0
#     max_tags, min_tags = max(tagsl), min(tagsl) # 75 1
#     for idx, x in enumerate(countl):
#         countl[idx] = (x - min_count) / (max_count - min_count)
#     for idx, x in enumerate(tagsl):
#         tagsl[idx] = (x - min_tags) / (max_count - min_tags)
#     user_data = np.asarray(user_data)
#     tagsl = np.expand_dims(np.asarray(tagsl), axis=1)
#     countl = np.expand_dims(np.asarray(countl), axis=1)
#     # final dim = 846
#     final_user = np.hstack((user_data, time_data, tagsl, countl))
#     np.save("./dataset/user_.npy", final_user)
#     # user_pd = pd.DataFrame(user_data, columns=['location', 'user', 'ispro', 'photo_count', 'num_tags', 'ispublic'])
#     # user_pd.to_csv('./dataset/user_train.csv')
# f.close()

# words
# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")
# sentence_feature = []
# i = 0
# for vuid, v in cat_dic.items():
#     word_list = cat_dic.get(vuid)
#     word_list.append(text_dic.get(vuid))
#     sentence = "Category is " + word_list[0] + ", subcategory is " + word_list[1]\
#         + ", Concept is " + word_list[2] + ", title is " + word_list[3]
#     inputs = tokenizer(sentence, return_tensors='pt')
#     outputs = model(**inputs)
#     sentence_feature.append(outputs[1].data.cpu().numpy())
#     print(i)
#     i = i + 1
# n = np.vstack(sentence_feature)
# np.save("text.npy", n)

# image
# from transformers import AutoFeatureExtractor, SwinModel
# import torch
# from PIL import Image
# feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# with open("dataset/train_img_path.txt") as f:
#     img_list = []
#     i = 0
#     for ls in f.readlines():
#         print(i)
#         path = ls.strip().split(' ')[0]
#         image = Image.open(path)
#         inputs = feature_extractor(image, return_tensors="pt")
#         with torch.no_grad():
#             outputs = model(**inputs)
#             img_list.append(outputs[1].data.cpu().numpy())
#         i = i + 1
#     img_feature = np.vstack(img_list)
#     np.save('./dataset/image.npy', img_feature)
# f.close()

# split data
# img_dim = 768
# text_dim = 768
# user_dim = 447
# rng = np.random.default_rng(seed=5576)
# img = np.load("./dataset/image.npy")
# text = np.load("./dataset/text.npy")
# user = np.load("./dataset/user.npy")
# all_shuffle = np.hstack((img, text, user, label))
# rng.shuffle(all_shuffle)
# img = all_shuffle[:, :img_dim]
# text = all_shuffle[:, img_dim:img_dim + text_dim]
# user = all_shuffle[:, img_dim + text_dim:img_dim + text_dim + user_dim]
# label = all_shuffle[:, img_dim + text_dim + user_dim:]
# train_split = int(img.shape[0] * 0.6)
# val_split = int(img.shape[0] * 0.8)
# train_data, val_data, test_data = [], [], []
# for i in range(train_split):
#     train_img = img[[i], :]
#     train_text = text[[i], :]
#     train_user = user[[i], :]
#     train_label = label[[i], :]
#     train_data.append(((train_img, train_text, train_user), train_label))
# for i in range(train_split, val_split):
#     val_img = img[[i], :]
#     val_text = text[[i], :]
#     val_user = user[[i], :]
#     val_label = label[[i], :]
#     val_data.append(((val_img, val_text, val_user), val_label))
# for i in range(val_split, int(img.shape[0])):
#     test_img = img[[i], :]
#     test_text = text[[i], :]
#     test_user = user[[i], :]
#     test_label = label[[i], :]
#     test_data.append(((test_img, test_text, test_user), test_label))
# print(len(train_data), len(val_data), len(test_data))
# pkl_data = {"train": train_data, "val": val_data, "test_images": test_data}
# with open("./dataset/smpd.pkl", "wb") as f:
#     pickle.dump(pkl_data, f)