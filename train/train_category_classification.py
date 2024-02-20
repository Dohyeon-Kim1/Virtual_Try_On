import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.cloth_category_dataset import ClothCategoryDataset


def main():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.requires_grad_(False)
    model.fc = nn.Linear(2048,6)
    model.fc.requires_grad_(True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 50
    batch_size = 64

    max_acc = 0.0
    model.to(device)
    for epoch in range(epochs):
        dataset = ClothCategoryDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

        total_acc, total_loss = 0.0, 0.0
        for data, label in tqdm(dataloader):
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_acc += (pred.argmax(dim=-1) == label).sum() / len(dataset)
            total_loss += loss.detach().item() / len(dataloader)
        
        lr_scheduler.step()

        print(f"\nepoch: {epoch+1}/{epochs}  loss: {total_loss}  acc: {total_acc}")

        if total_acc > max_acc:
            torch.save({"model_sd": model.state_dict(), "label": dataset.label_mapping}, 
                       f"model_zoo/category_classifier/category_classifier_checkpoint_last.pth")
            max_acc = total_acc


if __name__ == "__main__":
    main()