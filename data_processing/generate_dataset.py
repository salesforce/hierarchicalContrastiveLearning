import os
import shutil
from torchvision import transforms
from data_processing.filename_parse import deepfashion_name_parse
from torch.utils.data import Dataset
import json
from PIL import Image
from data_processing.fileread_util import txt_parse

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer_ImageNet = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transformer_ImageNet = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def get_deepfashion_img_class_name(filename, mode):
    img_class_name = deepfashion_name_parse(filename, mode)
    return img_class_name

class DatasetCategory(Dataset):
    def __init__(self, root_dir, mode, train_listfile='', val_listfile='',
                 test_listfile='', class_map_file='',
                 class_seen_file='', class_unseen_file='', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if mode == 'train':
            self.list_file = train_listfile  # list of image names and classes (product_id)
            self.mode = 'train'
        elif mode == 'val':
            self.list_file = val_listfile
            self.mode = 'val'
        elif mode == 'test':
            self.list_file = test_listfile
            self.mode = 'test'
        else:
            raise Exception('Mode can only be train, val or test.')
        with open(self.list_file, 'r') as f:
            data_dict = json.load(f)
        assert len(data_dict['images']) == len(data_dict['categories'])
        num_data = len(data_dict['images'])
        self.seen_classes = []
        self.unseen_classes = []
        if len(class_unseen_file) > 0 and len(class_seen_file) > 0:
            class_names = txt_parse(class_seen_file)
            for class_name in class_names:
                if len(class_name) == 0:
                    continue
                if class_name[-1] == '\n':
                    class_name = class_name[:-1]
                self.seen_classes.append(class_name)
            class_names = txt_parse(class_unseen_file)
            for class_name in class_names:
                if len(class_name) == 0:
                    continue
                if class_name[-1] == '\n':
                    class_name = class_name[:-1]
                self.unseen_classes.append(class_name)
        if len(self.seen_classes) == 0:
            self.seen_class_map = self.convert_label_to_int(self.seen_classes, 0)
            self.unseen_class_map = self.convert_label_to_int(self.unseen_classes, len(self.seen_classes))
        if len(class_map_file) > 0:
            with open(class_map_file, 'r') as f:
                self.class_map = json.load(f)
        else:
            self.class_map = {**self.seen_class_map, **self.unseen_class_map}
            print("label map between str and int is {}".format(self.class_map))
            with open('class_map.json', 'w') as fp:
                json.dump(self.class_map, fp)
        self.filenames = []
        self.labels = []
        for i in range(num_data):
            self.filenames.append(data_dict['images'][i])
            self.labels.append(self.class_map[data_dict['categories'][i]])

    def __getitem__(self, index):
        img_name = self.filenames[index]
        label = self.labels[index]
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.filenames)

    def convert_label_to_int(self, label_strs, start_idx):
        idx = start_idx
        label_mapping = {}
        for label in label_strs:
            label_mapping[label] = idx
            idx += 1
        return label_mapping

    def gen_category_classification_folder(self, filename, output_dir, mode='train'):
        # filename is the evaluation partition list file
        img_class_name = get_deepfashion_img_class_name(filename, mode)
        if output_dir[-1] != '/':
            output_dir += '/'
        class_cnt = 0
        class_list = set()
        for f in img_class_name:
            img_class, img_name = f.split()
            full_name = os.path.join(self.root_dir, img_name)
            dest = os.path.join(output_dir, img_class)
            if img_class not in class_list:
                class_list.add(img_class)
                class_cnt += 1
                os.mkdir(dest)
            new_name = img_name.split('/')[-2] + '-' + img_name.split('/')[-1]
            dest_file = os.path.join(dest, new_name)
            shutil.copy(full_name, dest_file)
        return


class Productid(Dataset):
    def __init__(self, root_dir, mode, gallery_listfile, query_listfile, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(gallery_listfile, 'r') as f:
            gallery_dict = json.load(f)
        assert len(gallery_dict['images']) == len(gallery_dict['categories'])
        num_gallery_data = len(gallery_dict['images'])
        with open(query_listfile, 'r') as f:
            query_dict = json.load(f)
        assert len(query_dict['images']) == len(query_dict['categories'])
        num_query_data = len(query_dict['images'])
        class_map, invert_class_map = self.convert_label_to_int(gallery_dict['categories'], 0)
        with open('class_map.json', 'w') as fp:
            json.dump(class_map, fp)
        with open('invert_class_map.json', 'w') as fp:
            json.dump(invert_class_map, fp)
        self.check_invalid_label(query_dict['categories'], class_map)
        self.filenames = []
        self.labels = []
        if mode == 'query':
            for i in range(num_query_data):
                filename = query_dict['images'][i]
                self.filenames.append(filename)
                self.labels.append(class_map[query_dict['categories'][i]])
        elif mode == 'gallery':
            for i in range(num_gallery_data):
                filename = gallery_dict['images'][i]
                self.filenames.append(filename)
                self.labels.append(class_map[gallery_dict['categories'][i]])

    def __getitem__(self, index):
        img_name = self.filenames[index]
        label = self.labels[index]
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.filenames)

    def convert_label_to_int(self, label_strs, start_idx):
        idx = start_idx
        label_mapping = {}
        invert_label_mapping = {}
        label_exist = set()
        for label in label_strs:
            if label not in label_exist:
                label_mapping[label] = idx
                invert_label_mapping[idx] = label
                idx += 1
                label_exist.add(label)
        return label_mapping, invert_label_mapping

    def check_invalid_label(self, label_strs, label_map):
        for label in label_strs:
            if label not in label_map:
                print("Unknow label found in query {}".format(label))
                raise ValueError("Unknow label found")
        return


class deepfashion_eval_dataset(Dataset):
    def __init__(self, prod_id_file, transform):
        self.ann = []
        with open(prod_id_file, 'r') as f:
            data_dict = json.load(f)
        self.ann = data_dict['images']
        self.transform = transform
        self.image = []
        self.prod_to_img = {}
        self.img_to_prod = {}
        self.prod_str_to_int = {}
        prod_id_int = 0
        for img_id, ann in enumerate(self.ann):
            img_fullname = ann
            self.image.append(img_fullname)
            split = img_fullname.split('/')
            prod_id = split[-2][3:]
            img_name = split[-2][3:] + '_' + split[-1].split('.')[-2]
            if prod_id not in self.prod_str_to_int:
                self.prod_str_to_int[prod_id] = prod_id_int
                self.img_to_prod[img_id] = prod_id_int
                self.prod_to_img[prod_id_int] = [img_id]
                prod_id_int += 1
            else:
                prod_id_int_tmp = self.prod_str_to_int[prod_id]
                self.img_to_prod[img_id] = prod_id_int_tmp
                self.prod_to_img[prod_id_int_tmp] += [img_id]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = ann
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index
