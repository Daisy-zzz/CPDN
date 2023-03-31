import os
import torch

os.environ["WANDB_PROGRAM"] = "main.py"

DEVICE = torch.device("cuda:1")

VISUAL_DIM = 512
TEXT_DIM = 512
USER_DIM = 48+8
USER_EMB = 60093
CAT_EMB = 11
SUBCAT_EMB = 77
CONCEPT_EMB = 669
SEQ_LEN = 8