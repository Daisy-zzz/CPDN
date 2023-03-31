import json
import re

import pandas as pd
from sklearn import preprocessing
import torch
device = torch.device("cuda:0")
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torchvision.models as models

data_type = "train"
# temporal
with open("dataset/%s_allmetadata_json/%s_temporalspatial_information.json" % (data_type, data_type)) as f:
    train_ts = json.load(f)
    vuid_arr = []
    time_arr = []
    lon_arr, lat_arr, geoacc_arr = [], [], []
    ts_dict = {}
    # 对时间排序
    for k, v in enumerate(train_ts):
        vuid = v['Uid'] + '/' + v['Pid']
        vuid_arr.append(vuid)
        time_arr.append(v['Postdate'])
        if v['Geoaccuracy'] == '0':
            lon_arr.append(0)
            lat_arr.append(0)
            geoacc_arr.append(0)
        else:
            lon_arr.append(v['Longitude'])
            lat_arr.append(v['Latitude'])
            geoacc_arr.append(v['Geoaccuracy'])
    id_arr = list(range(0, len(time_arr)))
    time_sort, id_sort = (list(t) for t in zip(*sorted(zip(time_arr, id_arr))))
    vuid_sort = []
    lon_sort, lat_sort, geoacc_sort = [], [], []
    for id in id_sort:
        vuid_sort.append(vuid_arr[id])
        lon_sort.append(lon_arr[id])
        lat_sort.append(lat_arr[id])
        geoacc_sort.append(geoacc_arr[id])
f.close()
# category
with open("dataset/%s_allmetadata_json/%s_category.json" % (data_type, data_type)) as f:
    train_cat = json.load(f)
    cat_arr, subcat_arr, concept_arr = [], [], []
    cat_sort, subcat_sort, concept_sort = [], [], []
    for v in train_cat:
        cat_arr.append(v['Category'])
        subcat_arr.append(v['Subcategory'])
        concept_arr.append(v['Concept'])
    for id in id_sort:
        cat_sort.append(cat_arr[id])
        subcat_sort.append(subcat_arr[id])
        concept_sort.append(concept_arr[id])
f.close()
# text
with open("dataset/%s_allmetadata_json/%s_text.json" % (data_type, data_type)) as f:
    train_text = json.load(f)
    text_arr, text_sort = [], []
    tags_arr, tags_sort = [], []
    for v in train_text:
        text_arr.append(v['Title'])
        tags_arr.append(v['Alltags'])
    for id in id_sort:
        text_sort.append(text_arr[id])
        tags_sort.append(tags_arr[id])
f.close()
# additional
with open("dataset/%s_allmetadata_json/%s_additional_information.json" % (data_type, data_type)) as f:
    train_add = json.load(f)
    public_arr, public_sort, path_arr, path_sort = [], [], [], []
    for v in train_add:
        vuid = v['Uid'] + '/' + v['Pid']
        public_arr.append(v['Ispublic'])
        path_arr.append(v['Pathalias'])
    for id in id_sort:
        public_sort.append(public_arr[id])
        path_sort.append(path_arr[id])
f.close()

# import requests
# import time
# from requests.adapters import HTTPAdapter
# s = requests.session()
# s.keep_alive = False
# s.mount('https://', HTTPAdapter(max_retries=3))
# 爬取数据
# user_path_dict = {}
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.55"
# }
# with open('./dataset/train_data_add.csv', 'a+') as f:
#     #f.writelines("vuid, followers, following, totalviews, totaltags, totalgeotags, totalfaves, totalingroup, true_id")
# #df = pd.DataFrame(columns=['id', 'followers', 'following', 'totalviews', 'totaltags', 'totalgeotags', 'totalfaves', 'totalingroup', 'true_id'])
#     for i, path in enumerate(path_sort):
#         flag = False
#         if i < 402:
#             continue
#         uid = vuid_sort[i].split('/')[0]
#         print(i, vuid_sort[i])
#         if user_path_dict.get(uid):
#             if user_path_dict.get(uid) != 'none':
#                 data = user_path_dict.get(uid)
#             else:
#                 data = [vuid_sort[i], 0, 0, 0, 0, 0, 0, 0, 0]
#         else:
#             if path != 'None':
#                 url = 'https://www.flickr.com/people/' + path
#             else:
#                 url = None
#             if url is not None:
#                 while flag is not True:
#                     try:
#                         resp = requests.get(url, headers=headers, timeout=8)
#                         if resp.status_code == 200:
#                             result = resp.text
#                             p1 = '>.*Follower'
#                             p2 = '>[0-9A-Z\s\.]+Following'
#                             p3 = 'totalViews.*?}'
#                             follower = re.findall(p1, result)[0].split(' ')[0][1:]
#                             following = re.findall(p2, result)[0].split(' ')[0][1:]
#                             others = re.findall(p3, result)[0].split(',')
#                             totalviews = others[0].split(':')[1]
#                             totaltags = others[1].split(':')[1]
#                             totalgeotags = others[2].split(':')[1]
#                             totalfaves = others[3].split(':')[1]
#                             totalingroup = others[4].split(':')[1]
#                             true_id = others[5].split(':')[1][1:-2]
#                             data = [vuid_sort[i], follower, following, totalviews, totaltags, totalgeotags, totalfaves, totalingroup, true_id]
#
#                             # followers = html.xpath()
#                             # print(html)  # 拿到页面源代码
#                             resp.close()  # 关掉resp
#                             user_path_dict[uid] = data
#                         else:
#                             user_path_dict[uid] = 'none'
#                             data = [vuid_sort[i], 0, 0, 0, 0, 0, 0, 0, 0]
#                         flag = True
#                     except requests.exceptions.RequestException as e:
#                         print(e)
#                         flag = False
#                         time.sleep(5)
#             else:
#                 user_path_dict[uid] = 'none'
#                 data = [vuid_sort[i], 0, 0, 0, 0, 0, 0, 0, 0]
#         f.writelines('\n' + ','.join(str(s) for s in data))
#         f.flush()
# label
if data_type == " train":
    with open("dataset/train_allmetadata_json/train_label.txt") as f:
        label_arr, label_sort = [], []
        for ls in f.readlines():
            l = ls.strip().split(' ')[0]
            label_arr.append(float(l))
        for id in id_sort:
            label_sort.append(label_arr[id])
    f.close()
# user
# fill_loc = "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"
with open("dataset/%s_allmetadata_json/%s_user_data.json" % (data_type, data_type)) as f:
    train_user = json.load(f)
    ispro_arr, ispro_sort, pcount_arr, pcount_sort, canpro_arr, canpro_sort = [], [], [], [], [], []
    loc_arr, loc_sort = [], []
    tzid_arr, tzid_sort, tzoffset_arr, tzoffset_sort = [], [], [], []
    loc_str = ''
    for v in train_user:
        ispro_arr.append(v['ispro'])
        pcount_arr.append(v['photo_count'])
        canpro_arr.append(v['canbuypro'])
        tzid_arr.append(v['timezone_timezone_id'])
        tzoffset_arr.append(v['timezone_offset'])
        # if v['location_description'] == "\n":
        #     loc_str = fill_loc
        # else:
        #     loc_str = v['location_description']
        # loc_str_list = loc_str.split(',')
        # loc_list = []
        # for i in loc_str_list:
        #     loc_list.append(float(i))
        # loc_arr.append(loc_list)
    # print(vuid_arr[165118], vuid_arr[123131])
    # print(pcount_arr[165118], pcount_arr[123131])
    # print(ispro_arr[165118], ispro_arr[123131])
    for id in id_sort:
        ispro_sort.append(ispro_arr[id])
        pcount_sort.append(pcount_arr[id])
        canpro_sort.append(canpro_arr[id])
        tzid_sort.append(tzid_arr[id])
        tzoffset_sort.append(tzoffset_arr[id])
        # loc_sort.append((loc_arr[id]))
pcount_sort = list(preprocessing.minmax_scale(pcount_sort))
f.close()
# np.save('./dataset/features/location.npy', np.asarray(loc_sort))

# df = pd.read_csv("./dataset/train_data_1.csv")
# df.insert(12, 'canbuypro', canpro_sort, allow_duplicates=False)
# df.insert(13, 'photo_count', pcount_sort, allow_duplicates=False)
# df.insert(14, 'timezone_id', tzid_sort, allow_duplicates=False)
# df.insert(15, 'timezone_offset', tzoffset_sort, allow_duplicates=False)
#
# df.to_csv('./dataset/train_data_2.csv')

# generate train img path
# with open("dataset/train_allmetadata_json/train_user_data.json") as f:
#     with open("./dataset/train_img_path.txt", "a+") as t:
#         train_user = json.load(f)
#         idx = 0
#         for vuid in vuid_sort:
#             user_dict = train_user[idx]
#             img_path = "./dataset/train_images/" + vuid + '.jpg'
#             t.write(img_path + '\n')
#     t.close()
# f.close()


# resnet提特征

# resnet_50 = models.resnet50(pretrained=True)
# resnet_50.to(device)
# resnet_50.eval()

# clip
image_features = []
text_features = []
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
for name, param in model.named_parameters():
    print(name, param.size())
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
batch = 512

a = int(len(vuid_sort) / batch)
b = len(vuid_sort) - a * batch
for j in range(0, a):
    print(j)
    images = []
    texts = []
    for i in range(j * batch, (j + 1) * batch):
        text = ' '.join(tags_sort[i])
        image_path = "./dataset/train_images/" + vuid_sort[i] + '.jpg'
        image = Image.open(image_path)
        images.append(image)
        texts.append(text)
    inputs = processor(text=texts, images=images, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    image_feature = outputs.vision_model_output.pooler_output
    text_feature = outputs.text_model_output.pooler_output
    image_features.append(image_feature.cpu().numpy())
    text_features.append(text_feature.cpu().numpy())
#
images = []
texts = []
for i in range(a * batch, a * batch + b):
    print(i)
    text = ' '.join(tags_sort[i])
    image_path = "./dataset/train_images/" + vuid_sort[i] + '.jpg'
    image = Image.open(image_path)
    images.append(image)
    texts.append(text)
inputs = processor(text=texts, images=images, return_tensors="pt", truncation=True, padding=True).to(device)
with torch.no_grad():
    outputs = model(**inputs)
image_feature = outputs.vision_model_output.pooler_output
text_feature = outputs.text_model_output.pooler_output
image_features.append(image_feature.cpu().numpy())
text_features.append(text_feature.cpu().numpy())

# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
n = np.vstack(text_features)
np.save("./dataset/features/text_clip_pooler.npy", n)
m = np.vstack(image_features)
np.save("./dataset/features/image_clip_pooler.npy", m)


# if data_type == "train":
#     data = zip(*[id_sort, vuid_sort, time_sort, cat_sort, subcat_sort, concept_sort, text_sort, tags_sort, public_sort, label_sort])
#     dataframe = pd.DataFrame(data, columns=['id', 'vuid', 'time', 'category', 'subcategory', 'concept', 'text', 'tags', 'ispublic', 'label'])
#     dataframe.to_csv('./dataset/train_data.csv')
# else:
#     data = zip(*[id_sort, vuid_sort, time_sort, cat_sort, subcat_sort, concept_sort, text_sort, tags_sort, public_sort])
#     dataframe = pd.DataFrame(data, columns=['id', 'vuid', 'time', 'category', 'subcategory', 'concept', 'text', 'tags',
#                                             'ispublic'])
#     dataframe.to_csv('./dataset/test_data.csv')

