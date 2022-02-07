
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# our scripts and functions
import settings as st
from src.dataset import DataSet
from src.networks import SiameseNetwork

out_path = './output/'

os.makedirs(out_path, exist_ok=True)

# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = DataSet(st.train_path, shuffle=True, augment=True, normalise=False)
val_dataset = DataSet(st.val_path, shuffle=False, augment=False, normalise=False)

train_dataloader = DataLoader(train_dataset, batch_size=8, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

model = SiameseNetwork(backbone="resnet18")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1E-4, weight_decay=1E-5)
criterion = torch.nn.BCELoss()

writer = SummaryWriter(os.path.join(out_path, "summary"))

best_val = 10000000000

epochs = 100

for epoch in range(epochs):
    print("[{} / {}]".format(epoch+1, epochs))
    model.train()

    losses = []
    correct = 0
    total = 0

    # Training Loop Start
    for (img1, img2), y, (class1, class2) in train_dataloader:
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        prob = model(img1, img2)
        loss = criterion(prob, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        correct += torch.count_nonzero(y == (prob > 0.5)).item()
        total += len(y)

    writer.add_scalar('train_loss', sum(losses)/len(losses), epoch)
    writer.add_scalar('train_acc', correct / total, epoch)

    print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))
    # Training Loop End

    # Evaluation Loop Start
    model.eval()

    losses = []
    correct = 0
    total = 0

    for (img1, img2), y, (class1, class2) in val_dataloader:
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        prob = model(img1, img2)
        loss = criterion(prob, y)

        losses.append(loss.item())
        correct += torch.count_nonzero(y == (prob > 0.5)).item()
        total += len(y)

    val_loss = sum(losses)/max(1, len(losses))
    writer.add_scalar('val_loss', val_loss, epoch)
    writer.add_scalar('val_acc', correct / total, epoch)

    print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, correct / total))
