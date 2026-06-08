"""
dataset.py

수정사항:
1. shuffle=True → 다양한 클래스 샘플링
2. Tiny-ImageNet synset ID → ImageNet-1k 인덱스 remapping
   - Tiny-ImageNet val의 로컬 레이블(0~199)은 ImageFolder가 알파벳 순으로 부여
   - timm 모델은 ImageNet-1k 기준 인덱스(0~999) 사용
   - synset ID(nXXXXXXXX)를 통해 두 인덱스를 매핑
"""

import os
import json
import zipfile
import urllib.request
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


# ImageNet-1k synset → 클래스 인덱스 매핑
# timm이 사용하는 순서와 동일 (imagenet_class_index.json 기준)
_IMAGENET_CLASS_INDEX_URL = (
    'https://storage.googleapis.com/download.tensorflow.org/'
    'data/imagenet_class_index.json'
)


def _load_imagenet_synset_to_idx(cache_path):
    """
    {synset_id: imagenet_idx} dict 반환.
    로컬 캐시 없으면 다운로드.
    """
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print("  ImageNet class index 다운로드 중...")
        urllib.request.urlretrieve(_IMAGENET_CLASS_INDEX_URL, cache_path)

    with open(cache_path) as f:
        data = json.load(f)  # {"0": ["n01440764", "tench"], ...}

    return {v[0]: int(k) for k, v in data.items()}  # {"n01440764": 0, ...}


def _build_label_map(dataset_root, cache_path):
    """
    Tiny-ImageNet 로컬 인덱스(0~199) → ImageNet-1k 인덱스(0~999) 매핑 반환.
    ImageFolder는 알파벳 순으로 클래스 인덱스를 부여하므로
    classes[i] = synset_id 관계를 이용.
    """
    synset_to_imagenet = _load_imagenet_synset_to_idx(cache_path)

    # ImageFolder의 class_to_idx: {synset_id: local_idx}
    val_dir = os.path.join(dataset_root, 'val')
    tmp_ds  = torchvision.datasets.ImageFolder(root=val_dir)
    # tmp_ds.classes: 알파벳 순 synset ID 리스트
    # tmp_ds.class_to_idx: {synset: local_idx}

    label_map = {}   # local_idx → imagenet_idx
    missing   = []
    for synset, local_idx in tmp_ds.class_to_idx.items():
        if synset in synset_to_imagenet:
            label_map[local_idx] = synset_to_imagenet[synset]
        else:
            missing.append(synset)

    if missing:
        print(f"  경고: ImageNet-1k에 없는 synset {len(missing)}개 "
              f"(해당 샘플은 Fooling Rate 계산에서 제외됨)")

    return label_map


class RemappedSubset(torch.utils.data.Dataset):
    """
    ImageFolder Subset + 레이블 remapping.
    label_map[local_idx] = imagenet_idx
    매핑 없는 클래스는 -1 반환 (Fooling Rate 계산 시 제외).
    """
    def __init__(self, dataset, indices, label_map):
        self.dataset   = dataset
        self.indices   = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, local_label = self.dataset[self.indices[i]]
        imagenet_label   = self.label_map.get(local_label, -1)
        return img, imagenet_label


def get_dataloader(batch_size=32, num_samples=1000,
                   data_root='./data', seed=42):
    """
    Parameters
    ----------
    batch_size  : int
    num_samples : int   총 샘플 수 (val 전체 10000장 이하)
    data_root   : str   'data/' 경로
    seed        : int   재현성용 시드

    Returns
    -------
    loader  : DataLoader  (remapped labels, shuffled)
    dataset : ImageFolder (원본, 참조용)
    """
    tiny_root  = os.path.join(data_root, 'tiny-imagenet-200')
    val_dir    = os.path.join(tiny_root, 'val')
    cache_path = os.path.join(data_root, 'imagenet_class_index.json')

    # 다운로드 및 val 구조 정리
    if not os.path.exists(tiny_root):
        _download_and_extract(data_root, tiny_root, val_dir)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset   = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    label_map = _build_label_map(tiny_root, cache_path)

    # 재현 가능한 셔플 인덱스
    rng     = torch.Generator()
    rng.manual_seed(seed)
    perm    = torch.randperm(len(dataset), generator=rng).tolist()
    indices = perm[:min(num_samples, len(dataset))]

    subset = RemappedSubset(dataset, indices, label_map)
    loader = DataLoader(subset, batch_size=batch_size,
                        shuffle=False,   # 이미 랜덤 인덱스로 뽑음
                        num_workers=4, pin_memory=True)

    print(f"  Dataset: Tiny-ImageNet val | "
          f"samples={len(subset)} | classes mapped={len(label_map)}/200")
    return loader, dataset


def _download_and_extract(data_root, tiny_root, val_dir):
    print("  Tiny-ImageNet 다운로드 중... (약 237MB)")
    os.makedirs(data_root, exist_ok=True)
    url      = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(data_root, 'tiny-imagenet-200.zip')
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_root)
    os.remove(zip_path)
    _reorganize_tiny_val(val_dir)


def _reorganize_tiny_val(val_dir):
    import shutil
    ann_file = os.path.join(val_dir, 'val_annotations.txt')
    img_dir  = os.path.join(val_dir, 'images')
    if not os.path.exists(ann_file):
        return
    with open(ann_file) as f:
        for line in f:
            parts     = line.strip().split('\t')
            fname, cid = parts[0], parts[1]
            class_dir = os.path.join(val_dir, cid)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(img_dir, fname)
            dst = os.path.join(class_dir, fname)
            if os.path.exists(src):
                shutil.move(src, dst)
    shutil.rmtree(img_dir, ignore_errors=True)
    if os.path.exists(ann_file):
        os.remove(ann_file)
    print("  val 폴더 구조 재구성 완료")