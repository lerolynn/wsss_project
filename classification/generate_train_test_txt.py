import random
import math

"""
Split the training data into training and validation data
"""

split_puportion = 0.9

if __name__ == '__main__':
    name = './labels/train_label.txt'
    with open(name, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 获取所有行
        # sum = 0
        file_names = [file_name for file_name in lines]
        # for line in lines:  # 第i行
        #     # 找到第一个空格
        #     file_list.append(line)
        #     # for j in range(len(line)):
        #     #     if line[j].isspace() == True:
        #     #         a = line[:j]
        #     #         # if a not in list:
        #     #         list.append(a)
        #     #         sum += 1
        f.close()

    file_name_len = len(file_names)
    random.Random(4).shuffle(file_names)

    split_loc = math.floor(split_puportion * file_name_len)

    train_split = file_names[0:split_loc]
    val_split = file_names[split_loc:]

    with open('./labels/split_labels_91/train_split.txt', 'a', encoding='utf-8') as train_split_txt:
        for train_img in train_split:
            train_split_txt.write(train_img)
        train_split_txt.close()

    with open('./labels/split_labels_91/val_split.txt', 'a', encoding='utf-8') as val_split_txt:
        for val_img in val_split:
            val_split_txt.write(val_img)
        val_split_txt.close()