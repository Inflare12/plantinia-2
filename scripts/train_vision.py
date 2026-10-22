from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/datasets/PlantVillage-Dataset/raw/color'
OUT=ROOT/'models/plantinia_vision.pt'
p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=3); p.add_argument('--batch',type=int,default=16); p.add_argument('--max-classes',type=int,default=16); args=p.parse_args()
if not DATA.exists(): raise SystemExit(f'Dataset not found: {DATA}. Run bootstrap_models.ps1 first.')
tf=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
all_ds=datasets.ImageFolder(DATA,transform=tf)
keep=sorted(all_ds.classes)[:args.max_classes]; class_to_idx={c:i for i,c in enumerate(keep)}
indices=[i for i,(_,label) in enumerate(all_ds.samples) if all_ds.classes[label] in class_to_idx]
class Filtered(torch.utils.data.Dataset):
    def __len__(self): return len(indices)
    def __getitem__(self,i):
        x,_=all_ds[indices[i]]; old=all_ds.samples[indices[i]][1]; return x,class_to_idx[all_ds.classes[old]]
ds=Filtered(); n=int(len(ds)*.9); train,val=random_split(ds,[n,len(ds)-n],generator=torch.Generator().manual_seed(42))
tr=DataLoader(train,batch_size=args.batch,shuffle=True,num_workers=0); va=DataLoader(val,batch_size=args.batch,shuffle=False,num_workers=0)
model=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT); model.classifier[3]=nn.Linear(model.classifier[3].in_features,len(keep)); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
opt=torch.optim.AdamW(model.parameters(),lr=2e-4); loss_fn=nn.CrossEntropyLoss()
for ep in range(args.epochs):
    model.train(); total=correct=0
    for x,y in tr:
        x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step(); total+=y.numel(); correct+=(logits.argmax(1)==y).sum().item()
    model.eval(); vtotal=vcorrect=0
    with torch.no_grad():
        for x,y in va:
            x,y=x.to(device),y.to(device); logits=model(x); vcorrect+=(logits.argmax(1)==y).sum().item(); vtotal+=y.numel()
    print(f'epoch {ep+1}: train={correct/max(1,total):.3f} val={vcorrect/max(1,vtotal):.3f}')
OUT.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'classes':keep},OUT); print(f'Saved {OUT}')
