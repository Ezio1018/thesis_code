import os
import sys
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from PIL import Image

def make_img_gt_dataset(img_root,gt_root):
# Find sub-classes in the root. Since one sub-class in img_root should match one sub-class # in gt_root, we always suppose img_root and gt_root have folders with the same names
 
    if sys.version_info>=(3,5):
        classes=[d.name for d in os.scandir(img_root) if d.is_dir()]
    else:
        classes=[d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root,d))]
 
    classes.sort()
 
    images=[]
    
    for sub_class in classes:
        d1=os.path.join(img_root,sub_class)
        d2=os.path.join(gt_root,sub_class)
 
        img_names=sorted(os.listdir(d1))
        gt_names=sorted(os.listdir(d2))

        for img_name in img_names:
            img_name_without_ext=img_name[0:len(img_name)-4]
            gt_name=img_name_without_ext+"_mask.png"
 
            if gt_name in gt_names:
                img_path=os.path.join(img_root,sub_class,img_name)
                gt_path=os.path.join(gt_root,sub_class,gt_name)
 
                item=(img_path,gt_path)
                images.append(item)
 
    return images
 
class CustomVisionDataset(VisionDataset):
    def __init__(self,img_root,gt_root,loader=default_loader,img_transform=None,gt_transform=None):
        super().__init__(root=img_root,transform=img_transform,target_transform=gt_transform)
    
        self.loader=loader
    
        samples=make_img_gt_dataset(img_root,gt_root)
        self.samples=samples
        self.img_samples=[s[0] for s in samples]
        self.gt_samples=[s[1] for s in samples]
 
    def __getitem__(self,index):
        img_path,gt_path=self.samples[index]
        img_sample = Image.open(img_path).convert('RGB') 
        gt_sample = Image.open(gt_path).convert('1') 
 
        if self.transform is not None:
            img_sample=self.transform(img_sample)
        if self.target_transform is not None:
            gt_sample=self.target_transform(gt_sample)

        return img_sample,gt_sample
 
    def __len__(self):
        return len(self.samples)


 