import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# our scripts and functions
import settings as st
from src.dataset import DataSet
from src.netcon import SiameseNetwork

def evaluate_pair(output1,output2,target,threshold):
    euclidean_distance = F.pairwise_distance(output1, output2)
    # if target == 1:
    #     return euclidean_distance > threshold
    # else:
    #     return euclidean_distance <= threshold
    cond = euclidean_distance<threshold
    # print(cond)
    pos_sum = 0
    neg_sum = 0
    pos_acc = 0
    neg_acc = 0

    for i in range(len(cond)):
        if target[i]:
            neg_sum+=1
            if not cond[i]:
                neg_acc+=1
        if not target[i]:
            pos_sum+=1
            if cond[i]:
                pos_acc+=1

    return pos_acc,pos_sum,neg_acc,neg_sum


epochs = 5
margin = 2.0

out_path = './output/'

os.makedirs(out_path, exist_ok=True)

# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = DataSet(st.train_path, shuffle=True, augment=True)
val_dataset = DataSet(st.val_path, shuffle=False, augment=False)

train_dataloader = DataLoader(train_dataset, batch_size=8, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

ntrain = len(train_dataloader)
nvalid = len(val_dataloader)

model = SiameseNetwork(backbone="resnet18")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1E-4, weight_decay=1E-5)
criterion = torch.nn.BCELoss()

writer = SummaryWriter(os.path.join(out_path, "summary"))

train_loss = []
valid_loss = []

for epoch in range(epochs):
    print("[{} / {}]".format(epoch + 1, epochs))
    model.train()
    train_epoch_loss = 0.0

    # Training Loop Start
    for (img1, img2), y, (class1, class2) in train_dataloader:
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, y)
        train_epoch_loss += loss.item()
        loss.backward()
        optim.step()
    
    train_epoch_loss /= ntrain
    train_loss.append(train_epoch_loss)
    
    print("Epoch [{}/{}] ----> Training loss :{} \n".format(epoch+1,epochs,train_epoch_loss))

    valid_epoch_loss = 0
    val_pos_accuracy = 0
    val_neg_accuracy = 0
    num_pos = 0
    num_neg = 0
    model.eval()

    for (img1, img2), y, (class1, class2) in val_dataloader:
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, y)
        valid_epoch_loss += loss.item()

        pos_acc,pos_sum,neg_acc,neg_sum = evaluate_pair(output1,output2,y,threshold)
        val_pos_accuracy+=pos_acc
        val_neg_accuracy+=neg_acc
        num_pos+=pos_sum
        num_neg+=neg_sum

    valid_epoch_loss /= nvalid
    val_pos_accuracy /= num_pos
    val_neg_accuracy /= num_neg

    valid_loss.append(valid_epoch_loss)


    print("Validation loss :{} \t\t\t P Acc : {}, N Acc: {}\n".format(valid_epoch_loss,val_pos_accuracy,val_neg_accuracy))


