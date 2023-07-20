from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import time
t0 = time.time()

# import wecloud_callback
import os
import logging

logging.basicConfig(
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    level=logging.INFO,
)
accumulated_training_time = 0


def train(args, model, device, train_loader, optimizer, epoch):
    global accumulated_training_time
    model.train()
    epoch_start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time = time.time()
        # wecloud_callback.step_begin()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        # wecloud_callback.step_end()

        n_iter = (epoch - 1) * len(train_loader) + batch_idx + 1
        # in the end of one iteration
        logging.info(
            "epoch = {}, iteration = {}, trained_samples = {}, total_samples = {}, loss = {}, lr = {}, current_epoch_wall-clock_time = {}".format(
                epoch,  # epoch
                n_iter,  # iteration
                batch_idx * args.b + len(data),  # trained_samples
                len(train_loader.dataset),  # total_samples
                loss.item(),  # loss
                optimizer.param_groups[0]["lr"],  # lr
                time.time() - epoch_start_time,  # current epoch wall-clock time
            )
        )
        end = time.time()
        accumulated_training_time += end - batch_start_time
        print("[profiling] step time: {}s, accumuated training time: {}s".format(end - batch_start_time, accumulated_training_time))
        if args.profiling:
            logging.info(
                f"PROFILING: dataset total number {len(train_loader.dataset)}, training one batch costs {time.time() - batch_start_time} seconds"
            )
            return


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def get_last_ckpt(ckpt_dir):
    ckpts = os.listdir(ckpt_dir)
    if len(ckpts) == 0:
        return None
    ckpts.sort()
    return ckpts[-1]


def get_last_ckpt_epoch(ckpt_path):
    # return int(ckpt_path.split(".")[0].split("_")[-1])
    return int(ckpt_path)


def main():
    global accumulated_training_time, t0
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Resnet Example")
    parser.add_argument("-b", type=int, default=64, help="batch size for dataloader")
    parser.add_argument("-tb", type=int, default=256, help="batch size for dataloader")
    parser.add_argument("--epoch", type=int, default=10, help="num of epochs to train")
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument("--gpu", action="store_true", default=False, help="CUDA training")
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--profiling", action="store_true", default=False, help="profile one batch"
    )
    parser.add_argument("--ckpt-dir", type=str, default="output/checkpoint", help="checkpoint directory")
    parser.add_argument("--data-dir", type=str, default="./data", help="data directory")
    args = parser.parse_args()
    use_cuda = args.gpu and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
        ]
    )

    # cmd = "mkdir -p " + args.ckpt_dir
    # # python 2.7 & 3
    # ret = subprocess.check_output(cmd, shell=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # CIFAR-10 数据集下载
    train_dataset = datasets.CIFAR10(
        root=args.data_dir, train=True, transform=transform, download=True
    )

    test_dataset = datasets.CIFAR10(
        root=args.data_dir, train=False, transform=transforms.ToTensor()
    )

    # 数据载入
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.b, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.tb, shuffle=False
    )

    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    last_epoch = 0
    last_ckpt = get_last_ckpt(args.ckpt_dir)
    if last_ckpt is not None:
        last_epoch = get_last_ckpt_epoch(last_ckpt)
        model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, last_ckpt, "checkpoint.pth")))
        print("load ckpt from %s" % last_ckpt)

    # wecloud_callback.init(total_steps=args.epoch * iter_per_epoch)
    t1 = time.time()
    print("[profiling] init time: {}s".format(t1-t0))
    for epoch in range(1, args.epoch + 1):
        scheduler.step()
        if epoch <= last_epoch:
            continue
        train(args, model, device, train_loader, optimizer, epoch)
        if args.profiling:
            break
        test(model, device, test_loader)
        os.mkdirs(os.path.join(args.ckpt_dir, str(epoch)))
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, str(epoch),"checkpoint.pth"))


if __name__ == "__main__":
    main()
