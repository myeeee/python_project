import os.path as osp
import glob


def make_list(phase='train'):

    folder_name = 'hymenoptera_data/'
    file_name = osp.join(folder_name + phase + '/**/*.jpg')

    file_list = []

    for f in glob.glob(file_name):
        file_list.append(f)

    return file_list

train_list = make_list('train')
val_list = make_list('val')

#print(train_list)