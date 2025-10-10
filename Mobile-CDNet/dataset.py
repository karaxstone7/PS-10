
# dataset.py  (STRICT Mobile-CD style)
# Expected layout:
#   <file_root>/
#     A/        B/        label/
#     list/
#       train.txt  val.txt  test.txt      # each line: filename like 0001.png

import os
import cv2
import numpy as np
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, dataset, file_root='data/', transform=None):
        """
        dataset: 'train' | 'val' | 'test'
        file_root: absolute path to Mobile-CD style root
        """
        self.split = dataset
        self.root = os.path.abspath(file_root)
        self.transform = transform

        # STRICT: only accept <root>/list/<split>.txt
        list_path = os.path.join(self.root, "list", f"{dataset}.txt")
        if not os.path.isfile(list_path):
            raise FileNotFoundError(
                f"List file not found: {list_path}. "
                f"Create it under {self.root}/list/{dataset}.txt"
            )

        with open(list_path, "r") as f:
            names = [ln.strip() for ln in f if ln.strip()]

        A_dir = os.path.join(self.root, "A")
        B_dir = os.path.join(self.root, "B")
        L_dir = os.path.join(self.root, "label")

        if not (os.path.isdir(A_dir) and os.path.isdir(B_dir) and os.path.isdir(L_dir)):
            raise FileNotFoundError(f"Missing A/B/label folders under {self.root}")

        pre_images, post_images, gts = [], [], []
        missing = []
        for n in names:
            pa = os.path.join(A_dir, n)
            pb = os.path.join(B_dir, n)
            pl = os.path.join(L_dir, n)
            if os.path.isfile(pa) and os.path.isfile(pb) and os.path.isfile(pl):
                pre_images.append(pa)
                post_images.append(pb)
                gts.append(pl)
            else:
                missing.append(n)

        if missing:
            raise RuntimeError(
                f"{len(missing)} listed names missing in A/B/label under {self.root}. "
                f"First few: {missing[:10]}"
            )

        if len(pre_images) == 0:
            raise RuntimeError(f"No valid samples for split='{self.split}' under {self.root}")

        # Optional debug: print a couple of resolved paths
        if os.environ.get("DEBUG_DATASET", "0") == "1":
            print(f"[DEBUG] Split='{self.split}' root='{self.root}' count={len(pre_images)}")
            for i in range(min(3, len(pre_images))):
                print("[DEBUG]", pre_images[i], "|", post_images[i], "|", gts[i])

        self.pre_images = pre_images
        self.post_images = post_images
        self.gts = gts

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_path = self.pre_images[idx]
        post_path = self.post_images[idx]
        gt_path = self.gts[idx]

        pre = cv2.imread(pre_path, cv2.IMREAD_COLOR)   # BGR
        post = cv2.imread(post_path, cv2.IMREAD_COLOR)
        if pre is None or post is None:
            raise RuntimeError(f"Failed to read: {pre_path} or {post_path}")

        # BGR -> RGB
        pre = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
        post = cv2.cvtColor(post, cv2.COLOR_BGR2RGB)

        lab = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if lab is None:
            raise RuntimeError(f"Failed to read label: {gt_path}")
        # binarize to {0,1} and keep HxWx1
        lab = (lab > 0).astype(np.float32)[..., None]

        # concat into 6 channels and scale to [0,1]
        img6 = np.concatenate([pre, post], axis=2).astype(np.float32) / 255.0

        if self.transform is not None:
            img6, lab = self.transform(img6, lab)

        return img6, lab

    def get_img_info(self, idx):
        import cv2
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
