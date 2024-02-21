import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.cloth_category_dataset import ClothCategoryDataset


def train_category_classifier(category_classifier, optimizer_category_classifier,
                              batch_size, epochs, device="cuda"):
    category_classifier.train()
    category_classifier.requires_grad_(True)
    category_classifier.to(device)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_category_classifier, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    max_acc = 0.0
    for epoch in range(epochs):
        dataset = ClothCategoryDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_acc, total_loss = 0.0, 0.0
        for data, label in tqdm(dataloader):
            data = data.to(device)
            label = label.to(device)

            pred = category_classifier(data)
            loss = criterion(pred, label)

            optimizer_category_classifier.zero_grad()
            loss.backward()
            optimizer_category_classifier.step()

            total_acc += (pred.argmax(dim=-1) == label).sum() / len(dataset)
            total_loss += loss.detach().item() / len(dataloader)
        
        lr_scheduler.step()

        print(f"\nepoch: {epoch+1}/{epochs}  loss: {total_loss}  acc: {total_acc}")

        if total_acc > max_acc:
            torch.save({"model_sd": category_classifier.state_dict(), "label": dataset.label_mapping}, 
                       f"model_zoo/category_classifier/category_classifier_checkpoint_last.pth")
            max_acc = total_acc