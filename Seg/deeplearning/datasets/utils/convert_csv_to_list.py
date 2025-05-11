def convert_labeled_list(csv_list, r=1):
    img_pair_list = list()
    for csv_file in csv_list:
        with open(csv_file, 'r') as f:
            img_in_csv = f.read().split('\n')[1:-1]
        img_pair_list += img_in_csv
    img_list = [i.split(',')[0] for i in img_pair_list]
    if len(img_pair_list[0].split(',')) == 1:
        label_list = None
    else:
        label_list = [i.split(',')[-1].replace('.tif', '-{}.tif'.format(r)) for i in img_pair_list]
    return img_list, label_list

def convert_labeled_list(csv_list, r=1):
    img_pair_list = list()
    for csv_file in csv_list:
        with open(csv_file, 'r') as f:
            img_in_csv = f.read().split('\n')[1:-1]
        img_pair_list += img_in_csv
    img_list = [i.split(',')[0] for i in img_pair_list]
    if len(img_pair_list[0].split(',')) == 1:
        label_list = None
    else:
        label_list = [i.split(',')[-1].replace('.tif', '-{}.tif'.format(r)) for i in img_pair_list]
    return img_list, label_list

def STY_convert_labeled_list(csv_list, r=1):
    img_pair_list = list()
    for csv_file in csv_list:
        with open(csv_file, 'r') as f:
            img_in_csv = f.read().split('\n')[1:-1]
        img_pair_list += img_in_csv
    img_list = [i.split(',')[0] for i in img_pair_list]
    if len(img_pair_list[0].split(',')) == 1:
        label_list = None
        sty_list = None
    else:
        label_list = [
            i.split(',')[1].replace('.tif', '-{}.tif'.format(r))
            for i in img_pair_list
        ]
        sty_list = [i.split(',')[2] for i in img_pair_list]

    return img_list, label_list, sty_list  # 返回三个列表


def convert_labeled_list_prostate(csv_list):
    img_pair_list = list()
    for csv_file in csv_list:
        with open(csv_file, 'r') as f:
            img_in_csv = f.read().split('\n')[1:-1]
        img_pair_list += img_in_csv
    img_list = []
    label_list = []
    for item in img_pair_list:
        img_list.append(item.split(',')[0])
        label_list.append(item.split(',')[1])
    return img_list, label_list