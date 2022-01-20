import torch
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PATCH_SIZE=(16,16)
IMAGE_SIZE=(256,256)
STEP=8
NUM_ACTION=9
ACTION_X=[0,0,STEP,-STEP,STEP,-STEP,-STEP,STEP,0]
ACTION_Y=[STEP,-STEP,0,0,STEP,-STEP,STEP,-STEP,0]