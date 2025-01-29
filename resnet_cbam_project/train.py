import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets.Reid_Dataloader import ReidDataLoader
from models.resnet_cbam import ResNet50_CBAM
from torch.nn.functional import normalize
import random


def validate_pid_range(pids, num_classes):
    """Validate PID values are in [0, num_classes-1] range"""
    if pids.min() < 0 or pids.max() >= num_classes:
        raise ValueError(f"Invalid PID range detected: min={pids.min()}, max={pids.max()} "
                         f"(should be in [0, {num_classes - 1}])")


def generate_triplets(features, labels):
    """
    Generate triplets (anchor, positive, negative) for triplet loss.
    :param features: Tensor of shape (batch_size, feature_dim).
    :param labels: Tensor of shape (batch_size,).
    :return: Triplets (anchor, positive, negative).
    """
    triplets = []
    for i in range(len(labels)):
        anchor = features[i]
        label = labels[i]

        # Find positive samples (same label but different instance)
        positive_mask = (labels == label)
        positive_mask[i] = False  # Exclude self
        positive_indices = positive_mask.nonzero(as_tuple=True)[0]

        # Find negative samples
        negative_indices = (labels != label).nonzero(as_tuple=True)[0]

        if len(positive_indices) > 0 and len(negative_indices) > 0:
            positive = features[random.choice(positive_indices)]
            negative = features[random.choice(negative_indices)]
            triplets.append((anchor, positive, negative))

    if triplets:
        return [torch.stack(x) for x in zip(*triplets)]
    return None, None, None


def train_one_epoch(model, dataloader, ce_criterion, triplet_criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_triplet_loss = 0.0

    for imgs, pids, _ in dataloader:
        imgs, pids = imgs.to(device), pids.to(device)

        # Validate PID range before processing
        validate_pid_range(pids, num_classes)

        # Forward pass
        outputs, features = model(imgs)
        ce_loss = ce_criterion(outputs, pids)

        # Normalize features for triplet loss
        features = normalize(features, p=2, dim=1)
        anchor, positive, negative = generate_triplets(features, pids)

        triplet_loss = torch.tensor(0.0, device=device)
        if anchor is not None:
            triplet_loss = triplet_criterion(anchor, positive, negative)

        # Combine losses
        loss = ce_loss + triplet_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_ce_loss += ce_loss.item() * imgs.size(0)
        total_triplet_loss += triplet_loss.item() * imgs.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    avg_ce_loss = total_ce_loss / len(dataloader.dataset)
    avg_triplet_loss = total_triplet_loss / len(dataloader.dataset)

    return avg_loss, avg_ce_loss, avg_triplet_loss


def validate(model, dataloader, ce_criterion, triplet_criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_triplet_loss = 0.0
    correct = 0

    with torch.no_grad():
        for imgs, pids, _ in dataloader:
            imgs, pids = imgs.to(device), pids.to(device)

            # Validate PID range
            validate_pid_range(pids, num_classes)

            # Forward pass
            outputs, features = model(imgs)
            ce_loss = ce_criterion(outputs, pids)

            # Normalize features for triplet loss
            features = normalize(features, p=2, dim=1)
            anchor, positive, negative = generate_triplets(features, pids)

            triplet_loss = torch.tensor(0.0, device=device)
            if anchor is not None:
                triplet_loss = triplet_criterion(anchor, positive, negative)

            # Combine losses
            loss = ce_loss + triplet_loss

            total_loss += loss.item() * imgs.size(0)
            total_ce_loss += ce_loss.item() * imgs.size(0)
            total_triplet_loss += triplet_loss.item() * imgs.size(0)

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == pids).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_ce_loss = total_ce_loss / len(dataloader.dataset)
    avg_triplet_loss = total_triplet_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    return avg_loss, avg_ce_loss, avg_triplet_loss, accuracy


def main():
    # Configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "./data"
    dataset_name = "market1501"
    batch_size = 32
    num_epochs = 40
    learning_rate = 0.001
    weight_decay = 5e-4
    margin = 1.0
    num_classes = 751  # Market1501 has 751 unique person IDs (0-750)

    # Data loaders (需要确保ReidDataLoader正确转换PID到0-750范围)
    dataloader_train = ReidDataLoader(
        dataset_name=dataset_name,
        root_dir=root_dir,
        batch_size=batch_size,
        mode='train',
        num_classes=num_classes  # 确保数据加载器知道类别数
    ).get_dataloader()

    dataloader_val = ReidDataLoader(
        dataset_name=dataset_name,
        root_dir=root_dir,
        batch_size=batch_size,
        mode='test',
        num_classes=num_classes
    ).get_dataloader()

    # Model setup
    model = ResNet50_CBAM(num_classes=num_classes).to(device)

    # Loss and optimizer
    ce_criterion = nn.CrossEntropyLoss()
    triplet_criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        train_loss, train_ce_loss, train_triplet_loss = train_one_epoch(
            model, dataloader_train, ce_criterion, triplet_criterion, optimizer, device, num_classes
        )
        val_loss, val_ce_loss, val_triplet_loss, val_accuracy = validate(
            model, dataloader_val, ce_criterion, triplet_criterion, device, num_classes
        )

        scheduler.step()

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} (CE: {train_ce_loss:.4f}, Triplet: {train_triplet_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (CE: {val_ce_loss:.4f}, Triplet: {val_triplet_loss:.4f})")
        print(f"Val Accuracy: {val_accuracy:.2%}")

        torch.save(model.state_dict(), "latest_model.pth")

    print(f"\nTraining completed.")


if __name__ == "__main__":
    main()
