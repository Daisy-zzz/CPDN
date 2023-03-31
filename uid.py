import json

import numpy as np
from sklearn.preprocessing import OneHotEncoder

attn = np.load('attn.npy')
print(attn[33, -1, :])

train_uid_list = []
test_uid_list = []
cat_list = []
subcat_list = []
concept_list = []
with open("dataset/train_allmetadata_json/train_category.json") as f:
    train_cat = json.load(f)
    for k, v in enumerate(train_cat):
        train_uid_list.append(v['Uid'])
        cat_list.append(v['Category'])
        subcat_list.append(v['Subcategory'])
        concept_list.append(v['Concept'])
f.close()
with open("dataset/test_allmetadata_json/test_category.json") as f:
    test_cat = json.load(f)
    for k, v in enumerate(test_cat):
        test_uid_list.append(v['Uid'])
        cat_list.append(v['Category'])
        subcat_list.append(v['Subcategory'])
        concept_list.append(v['Concept'])
f.close()

cnt = 0
for uid in set(test_uid_list):
    if uid not in train_uid_list:
        cnt += 1
        print(uid)
print(cnt / len(set(test_uid_list)))


# uid_set = set(uid_list)
# id_list = list(range(len(uid_set)))
# uid_zip = zip(uid_set, id_list)
# uid_dict = dict(uid_zip)

# cat_set = set(cat_list)
# cat_list = list(range(len(cat_set)))
# cat_zip = zip(cat_set, cat_list)
# cat_dict = dict(cat_zip)

# subcat_set = set(subcat_list)
# subcat_list = list(range(len(subcat_set)))
# subcat_zip = zip(subcat_set, subcat_list)
# subcat_dict = dict(subcat_zip)

# concept_set = set(concept_list)
# concept_list = list(range(len(concept_set)))
# concept_zip = zip(concept_set, concept_list)
# concept_dict = dict(concept_zip)
# with open("./dataset/category.json", "a+") as f:
#     json.dump(cat_dict, f)
# f.close()
# with open("./dataset/subcategory.json", "a+") as f:
#     json.dump(subcat_dict, f)
# f.close()
# with open("./dataset/concept.json", "a+") as f:
#     json.dump(concept_dict, f)
# f.close()
# uid_set = [[i] for i in set(uid_list)]
# one_hot = OneHotEncoder()
# one_hot.fit(uid_set)
# user
# user = []
# with open("dataset/train_allmetadata_json/train_category.json") as f:
#     train_cat = json.load(f)
#     cat_dic = {}
#     for k, v in enumerate(train_cat):
#         uid = [[v['Uid']]]
#         user_data = one_hot.transform(uid).toarray()
#         user.append(user_data)
# f.close()

# user = np.asarray(user)
# np.save("./dataset/user_id.npy", user)