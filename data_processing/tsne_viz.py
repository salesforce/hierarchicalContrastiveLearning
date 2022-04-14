import torch
import json
import os
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from network import resnet_modified
from data_processing.hierarchical_dataset import DeepFashionHierarchihcalDatasetEval
import argparse
from sklearn import manifold
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import numpy as np


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset, the superset of train/val')
    parser.add_argument('--listfile', default='', type=str,
                        help='test file with annotation')
    parser.add_argument('--class-map-file', default='', type=str,
                        help='class mapping between str and int')
    parser.add_argument('--repeating-product-file', default='', type=str,
                        help='repeating product ids file')
    parser.add_argument('--ckpt', type=str,
                        help='the pth file to load')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size')
    parser.add_argument('--scale-size', default=256, type=int,
                        help='scale size in validation')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='crop size')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')

    args = parser.parse_args()
    return args

def get_model(args):
    model = resnet_modified.MyResNet(name='resnet50')

    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['state_dict']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:

            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.encoder", "encoder.module")
                if k.startswith("module.head"):
                    k = k.replace("module.head", "head")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        model.encoder = torch.nn.DataParallel(model.encoder)
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    return model


def load_hierarchical(root_dir, list_file,
                      class_map_file, repeating_product_file,
                      input_size, batch_size, crop_size):
    transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = DeepFashionHierarchihcalDatasetEval(os.path.join(root_dir, list_file),
                                                  os.path.join(
                                                      root_dir, class_map_file),
                                                  os.path.join(
                                                  root_dir, repeating_product_file),
                                                  transform=transform)


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=16)

    return dataloader

def get_embeddings(dataloader, model):
    gt_labels, embeddings = [], []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images = images.float().cuda()
            batch_labels = torch.stack(labels, dim=1)
            gt_labels.extend(batch_labels)
            batch_embeddings = model(images)
            embeddings.extend(batch_embeddings)
            
    return gt_labels, embeddings


def calc_nmi(embeddings, labels):
    num_clusters = np.unique(labels).shape[0]
    print(num_clusters)
    kmeans = KMeans(n_clusters=num_clusters,n_jobs=16)
    kmeans.fit(embeddings)
    y_kmeans_pred = kmeans.predict(embeddings)
    nmi = normalized_mutual_info_score(labels, y_kmeans_pred)
    print("NMI: " + str(nmi))

def main():
    global args
    args = parse_option()
    
    model = get_model(args)
    cudnn.benchmark = True
    dataloader = load_hierarchical(args.data, args.listfile,
                                   args.class_map_file, args.repeating_product_file, 
                                   args.input_size, args.batch_size, args.crop_size)
    print("num of batches:" + str(len(dataloader)))
    labels, embeddings = get_embeddings(dataloader, model)
    labels = torch.stack(labels)
    embeddings = torch.stack(embeddings).to('cpu')
    print(len(labels), len(embeddings))
    tsne = manifold.TSNE(n_components=2, init='pca',
                         random_state=0, perplexity=50)
    tsne_rep = tsne.fit_transform(embeddings)
    with open('tsne_curr.npy', 'wb') as f:
        np.save(f, [labels, tsne_rep, embeddings])
    print(tsne_rep.shape, labels.shape)
    

    # with open('tsne_curr.npy', 'rb') as f:
    #     labels, tsne_rep, embeddings = np.load(f, allow_pickle=True)

    calc_nmi(embeddings, labels[:,0])
    with open(args.class_map_file, 'r') as f:
            class_map = json.load(f)
    inv_map = {v: k for k, v in class_map.items()}
    cat_labels = [inv_map[int(x[0])] for x in labels]
    ax = sns.scatterplot(x=tsne_rep[:, 0], y=tsne_rep[:, 1],
                    hue=cat_labels)
    ax.legend(bbox_to_anchor=(1.1, 1.05), loc='upper right')
    plt.savefig('category.png')
    plt.close()

    blouses_shirts_class = class_map['Cardigans']
    category_idx = labels[:, 0] == blouses_shirts_class
    category_1_products = labels[category_idx, 1]
    ax = sns.scatterplot(x=tsne_rep[category_idx, 0],
                         y=tsne_rep[category_idx, 1],
                         hue=category_1_products, style = category_1_products,
                         palette=sns.color_palette('flare', 
                                                    len(category_1_products.unique())))
    ax.legend(bbox_to_anchor=(1.1, 1.05), loc='upper right')
    plt.savefig('product_id.png')

    calc_nmi(embeddings[category_idx], category_1_products)


if __name__ == '__main__':
    main()
