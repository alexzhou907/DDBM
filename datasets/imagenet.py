import torch
import torchvision

import math
import numpy as np
import os
from collections import defaultdict
import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from timm.data.transforms_factory import transforms_imagenet_eval
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS

import pickle
try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC

from typing import Any, Callable, Optional, Tuple

def download_and_unzip(URL, root_dir):
  error_message = f"Download is not yet implemented. Please, go to {URL} urself."
  raise NotImplementedError(error_message)

def _add_channels(img, total_channels=3):
  while len(img.shape) < 3:  # third axis is the channels
    img = np.expand_dims(img, axis=-1)
  while(img.shape[-1]) < 3:
    img = np.concatenate([img, img[:, :, -1:]], axis=-1)
  return img

class TinyImageNetPaths:
  def __init__(self, root_dir, download=False):
    if download:
      download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                         root_dir)
    train_path = os.path.join(root_dir, 'train')
    val_path = os.path.join(root_dir, 'val')
    test_path = os.path.join(root_dir, 'test')

    wnids_path = os.path.join(root_dir, 'wnids.txt')
    words_path = os.path.join(root_dir, 'words.txt')

    self._make_paths(train_path, val_path, test_path,
                     wnids_path, words_path)

  def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
    self.ids = []
    with open(wnids_path, 'r') as idf:
      for nid in idf:
        nid = nid.strip()
        self.ids.append(nid)
    self.nid_to_words = defaultdict(list)
    with open(words_path, 'r') as wf:
      for line in wf:
        nid, labels = line.split('\t')
        labels = list(map(lambda x: x.strip(), labels.split(',')))
        self.nid_to_words[nid].extend(labels)

    self.paths = {
      'train': [],  # [img_path, id, nid, box]
      'val': [],  # [img_path, id, nid, box]
      'test': []  # img_path
    }

    # Get the test paths
    self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      [s for s in os.listdir(test_path) if s.startswith('n')]))
    
    # Get the validation paths and labels
    with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
      for line in valf:
        fname, nid, x0, y0, x1, y1 = line.split()
        fname = os.path.join(val_path, 'images', fname)
        bbox = int(x0), int(y0), int(x1), int(y1)
        label_id = self.ids.index(nid)
        self.paths['val'].append((fname, label_id, nid, bbox))

    # Get the training paths
    train_nids = os.listdir(train_path)
    for nid in train_nids:
      anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
      imgs_path = os.path.join(train_path, nid, 'images')
      if nid not in self.ids:
        continue
      label_id = self.ids.index(nid)
      with open(anno_path, 'r') as annof:
        for line in annof:
          fname, x0, y0, x1, y1 = line.split()
          fname = os.path.join(imgs_path, fname)
          if os.path.isdir(fname):
            continue
          bbox = int(x0), int(y0), int(x1), int(y1)
          self.paths['train'].append((fname, label_id, nid, bbox))


class TinyImageNetDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir, mode='train', preload=False, load_transform=None,
               transform=None, download=False, max_samples=None):
    tinp = TinyImageNetPaths(root_dir, download)
    self.mode = mode
    self.label_idx = 1  # from [image, id, nid, box]
    self.preload = preload
    self.transform = transform
    self.transform_results = dict()

    self.IMAGE_SHAPE = (64, 64, 3)

    self.img_data = []
    self.label_data = []

    self.max_samples = max_samples
    self.samples = tinp.paths[mode]
    self.samples_num = len(self.samples)

    if self.max_samples is not None:
      self.samples_num = min(self.max_samples, self.samples_num)
      self.samples = np.random.permutation(self.samples)[:self.samples_num]

    if self.preload:
      load_desc = "Preloading {} data...".format(mode)
      self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                               dtype=np.float32)
      self.label_data = np.zeros((self.samples_num,), dtype=np.int)
      for idx in tqdm(range(self.samples_num), desc=load_desc):
        s = self.samples[idx]
        img = imageio.imread(s[0])
        img = _add_channels(img)
        self.img_data[idx] = img
        if mode != 'test':
          self.label_data[idx] = s[self.label_idx]

      if load_transform:
        for lt in load_transform:
          result = lt(self.img_data, self.label_data)
          self.img_data, self.label_data = result[:2]
          if len(result) > 2:
            self.transform_results.update(result[2])

  def __len__(self):
    return self.samples_num

  def __getitem__(self, idx):
    if self.preload:
      img = self.img_data[idx]
      lbl = None if self.mode == 'test' else self.label_data[idx]
    else:
      s = self.samples[idx]
      img = imageio.imread(s[0])
      img = _add_channels(img)
      lbl = None if self.mode == 'test' else s[self.label_idx]
      
    if self.transform:
      img = self.transform(img)
    sample = ( img,  lbl)

    return sample

def get_t(img_size, t_config):
    
  t = []
  
  if 'crop_scale' in t_config:
      t.append(Ktransforms.RandomResizedCrop(size=(img_size, img_size), scale=(t_config.crop_scale.min, t_config.crop_scale.max)))
  
  if 'flip_p' in t_config:
      t.append(Ktransforms.RandomHorizontalFlip(p=t_config.flip_p))
  
  if 'jitter' in t_config:
      t.append(Ktransforms.ColorJitter(t_config.jitter.b, t_config.jitter.c, t_config.jitter.s, t_config.jitter.h,  p=t_config.jitter.p))
      
      #t.append(transforms.RandomApply([jitter], p=t_config['jitter_p']))
  
  if 'gray_p' in t_config:
      t.append(Ktransforms.RandomGrayscale(p=t_config.gray_p))
  
  if 'blur_scale' in t_config:
      # blur = GaussianBlur(kernel_size=int(t_config['blur_scale'] * img_size))
      t.append(Ktransforms.RandomGaussianBlur(kernel_size=(int(t_config.blur_scale * img_size), int(t_config.blur_scale * img_size)),
                                        sigma=(0.5, 0.5), p=1.0))
  
  if 'noise_std' in t_config:
      t.append(Ktransforms.RandomGaussianNoise(std=t_config.noise_std))
  
  return Ktransforms.AugmentationSequential(*t)

############ imagenet


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch(data_folder,  img_size=32, filename='train_data_batch_1'):

    d = unpickle(os.path.join(data_folder, filename))
    x = d['data']
    y = d['labels']
    

    x = x/np.float32(255)
    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = np.array(y[0:data_size])

    return X_train, Y_train

class ImageNetPickleDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            train: bool,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            max_cls_id: int = 1000
    ):
        super(ImageNetDataset, self).__init__( )
        split = 'train' if train else 'val'

        files = [f for f in os.listdir(root) if f.startswith(split)]
        self.samples =[]
        self.targets = []
        for i, f in tqdm(enumerate(files), total=len(files)):
          X_train_b, Y_train_b = load_databatch(root,  64, f)
          self.samples.append(X_train_b[Y_train_b < max_cls_id])
          self.targets.append(Y_train_b[Y_train_b < max_cls_id])

        self.samples = np.concatenate(self.samples, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
    
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img

class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            max_cls_id: int = 1000,
            only=-1,
    ):
        for postfix in (os.path.sep, 'train', 'val'):
            if root.endswith(postfix):
                root = root[:-len(postfix)]
        
        root = os.path.join(root, 'train' if train else 'val')
        
        super(ImageNetDataset, self).__init__(
            root,
            # loader=ImageLoader(train),
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform, target_transform=target_transform, is_valid_file=is_valid_file
        )
        
        if only > 0:
            g = torch.Generator()
            g.manual_seed(0)
            idx = torch.randperm(len(self.samples), generator=g).numpy().tolist()

            ws = dist.get_world_size()
            res = (max_cls_id * only) % ws
            more = 0 if res == 0 else (ws - res)
            max_total = max_cls_id * only + more
            if (max_total // ws) % 2 == 1:
                more += ws
                max_total += ws
            
            d = {c: [] for c in range(max_cls_id)}
            max_len = {c: only for c in range(max_cls_id)}
            for c in range(max_cls_id-more, max_cls_id):
                max_len[c] += 1
            
            total = 0
            for i in idx:
                path, target = self.samples[i]
                if len(d[target]) < max_len[target]:
                    d[target].append((path, target))
                    total += 1
                if total == max_total:
                    break
            sp = []
            [sp.extend(l) for l in d.values()]

            print(f'[ds] more={more}, len(sp)={len(sp)}')
            self.samples = tuple(sp)
            self.targets = tuple([s[1] for s in self.samples])
        else:
            self.samples = tuple(filter(lambda item: item[-1] < max_cls_id, self.samples))
            self.targets = tuple([s[1] for s in self.samples])
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target



class ImageNet128Dataset(torch.utils.data.Dataset):
    def __init__(self, root, is_train=True, transform=None) -> None:
        self.root = root
        self.is_train = is_train
        self.transform = transform

        self.all_samples = []
        self.all_labels = []
        if is_train:
          files = ['train_data_batch_%i' % (i + 1) for i in range(100)]
        else:
          files = ['val_data_batch_%i' % (i + 1) for i in range(10)]
        for filename in files:        
            data, labels = self.unpickle(root + '/' + filename)
            labels = np.array(labels)-1
            self.all_samples.append(data)
            self.all_labels.append(labels)

    def __len__(self):
        return len(self.all_labels)
    
    def __getitem__(self, index: int):
        if self.transform is not None:
            return self.transform(self.all_samples[index]), self.all_labels[index]
        return self.all_samples[index], self.all_labels[index]

