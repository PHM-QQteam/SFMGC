import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gear_dataset import get_multi_datasets
from models_adabn_multi import Gear1DCNN_MultiDomain


SEED = 3407
BASE_PATH = "./data"

S_DATA = os.path.join(BASE_PATH, "data_load2.csv")
S_LABEL = os.path.join(BASE_PATH, "label_load2.csv")
T1_DATA = os.path.join(BASE_PATH, "data_load0.csv")
T1_LABEL = os.path.join(BASE_PATH, "label_load0.csv")
T2_DATA = os.path.join(BASE_PATH, "data_load1.csv")
T2_LABEL = os.path.join(BASE_PATH, "label_load1.csv")

BATCH_SIZE = 128
PRETRAIN_EPOCHS = 20
ADAPT_EPOCHS = 50
LR_PRE = 1e-3
LR_ADAPT = 1e-4
META_LR = 1e-3
THRESHOLD = 0.90
USE_FFT = False
LABEL_SMOOTHING = 0.01
W_ENT = 0.5
W_DIV = 2.0
W_FISH = 0.5
NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def entropy_loss(probabilities):
    return -torch.mean(
        torch.sum(probabilities * torch.log(probabilities + 1e-6), dim=1)
    )


def diversity_loss(logits):
    mean_probability = F.softmax(logits, dim=1).mean(dim=0)
    return torch.sum(mean_probability * torch.log(mean_probability + 1e-6))


def evaluate(model, loader, domain):
    model.eval()
    model.set_domain(domain)
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            logits = model(data)
            prediction = logits.argmax(dim=1)
            correct += prediction.eq(target).sum().item()
            total += target.size(0)

    return 100.0 * correct / total


def train():
    set_seed(SEED)

    source_dataset, target1_train, target1_test, target2_train, target2_test = (
        get_multi_datasets(
            S_DATA,
            S_LABEL,
            T1_DATA,
            T1_LABEL,
            T2_DATA,
            T2_LABEL,
            use_fft=USE_FFT,
        )
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    target1_loader = DataLoader(
        target1_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    target2_loader = DataLoader(
        target2_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    target1_test_loader = DataLoader(
        target1_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    target2_test_loader = DataLoader(
        target2_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = Gear1DCNN_MultiDomain(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    pretrain_optimizer = optim.Adam(model.parameters(), lr=LR_PRE)

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        model.set_domain("source")
        correct = 0
        total = 0

        for data, target in tqdm(
            source_loader,
            desc=f"Pretrain {epoch + 1}/{PRETRAIN_EPOCHS}",
        ):
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            pretrain_optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            pretrain_optimizer.step()

            prediction = logits.argmax(dim=1)
            correct += prediction.eq(target).sum().item()
            total += target.size(0)

            model.set_domain("target1")
            with torch.no_grad():
                model(data)

            model.set_domain("target2")
            with torch.no_grad():
                model(data)

            model.set_domain("source")

    source_accuracy = 100.0 * correct / total
    target1_initial = evaluate(model, target1_test_loader, "target1")
    target2_initial = evaluate(model, target2_test_loader, "target2")

    print(f"Source accuracy: {source_accuracy:.2f}%")
    print(
        f"Initial target accuracy: "
        f"T1={target1_initial:.2f}%, T2={target2_initial:.2f}%"
    )

    for parameter in model.fc.parameters():
        parameter.requires_grad = False

    adaptation_optimizer = optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=LR_ADAPT,
    )

    final_target1 = target1_initial
    final_target2 = target2_initial

    for epoch in range(ADAPT_EPOCHS):
        model.train()

        target1_iterator = iter(target1_loader)
        target2_iterator = iter(target2_loader)
        num_batches = min(len(target1_loader), len(target2_loader))

        for _ in tqdm(
            range(num_batches),
            desc=f"Adapt {epoch + 1}/{ADAPT_EPOCHS}",
        ):
            data_target1, _ = next(target1_iterator)
            data_target2, _ = next(target2_iterator)

            data_target1 = data_target1.to(DEVICE)
            data_target2 = data_target2.to(DEVICE)

            temporary_model = copy.deepcopy(model)

            for parameter in temporary_model.fc.parameters():
                parameter.requires_grad = False

            temporary_optimizer = optim.Adam(
                filter(
                    lambda parameter: parameter.requires_grad,
                    temporary_model.parameters(),
                ),
                lr=META_LR,
            )

            temporary_model.set_domain("target1")
            logits_target1 = temporary_model(data_target1)
            probability_target1 = F.softmax(logits_target1, dim=1)

            confidence_target1, pseudo_label_target1 = probability_target1.max(dim=1)
            mask_target1 = confidence_target1.ge(THRESHOLD).float()

            pseudo_label_loss_target1 = (
                F.cross_entropy(
                    logits_target1,
                    pseudo_label_target1,
                    reduction="none",
                )
                * mask_target1
            ).mean()

            loss_target1 = (
                W_ENT * entropy_loss(probability_target1)
                + W_DIV * diversity_loss(logits_target1)
                + pseudo_label_loss_target1
            )

            temporary_optimizer.zero_grad()
            loss_target1.backward()
            temporary_optimizer.step()

            temporary_model.set_domain("target2")
            logits_target2 = temporary_model(data_target2)
            probability_target2 = F.softmax(logits_target2, dim=1)

            confidence_target2, pseudo_label_target2 = probability_target2.max(dim=1)
            mask_target2 = confidence_target2.ge(THRESHOLD).float()

            pseudo_label_loss_target2 = (
                F.cross_entropy(
                    logits_target2,
                    pseudo_label_target2,
                    reduction="none",
                )
                * mask_target2
            ).mean()

            loss_target2 = (
                W_ENT * entropy_loss(probability_target2)
                + W_DIV * diversity_loss(logits_target2)
                + pseudo_label_loss_target2
            )

            temporary_optimizer.zero_grad()
            loss_target2.backward()
            temporary_optimizer.step()

            adaptation_optimizer.zero_grad()

            for main_parameter, temporary_parameter in zip(
                model.parameters(),
                temporary_model.parameters(),
            ):
                if main_parameter.requires_grad:
                    if main_parameter.grad is None:
                        main_parameter.grad = torch.zeros_like(main_parameter)
                    parameter_difference = (
                        temporary_parameter.data - main_parameter.data
                    )
                    main_parameter.grad.data.add_(
                        -W_FISH * parameter_difference
                    )

            adaptation_optimizer.step()

        final_target1 = evaluate(model, target1_test_loader, "target1")
        final_target2 = evaluate(model, target2_test_loader, "target2")

        print(
            f"Epoch {epoch + 1:03d}: "
            f"T1={final_target1:.2f}%, T2={final_target2:.2f}%"
        )

    print(
        f"Final target accuracy: "
        f"T1={final_target1:.2f}%, T2={final_target2:.2f}%"
    )


if __name__ == "__main__":
    train()
