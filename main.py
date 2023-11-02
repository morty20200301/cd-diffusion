from tools import NativeScalerWithGradNormCount as NativeScaler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from engine import train_model, evaluation
import time
import datetime
from model.PSP import PSPNet
from data.CDDataset import CDDataset


def main():
    # distribution
    best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iou': 0}
    device = torch.device("cuda:1")
    cudnn.benchmark = True

    model = PSPNet(2)
    model.to(device).train()
    model_without_ddp = model
    loss_scaler = NativeScaler()

    train_set = CDDataset(root, split="train")
    val_set = CDDataset(root, split="test")

    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=sampler_val, num_workers=4, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))


    print(f"Start training for 40 epochs")
    start_time = time.time()

    for epoch in range(num_epoch):

        print("start training: ")
        train_model(epoch, num_epoch, model, train_loader, criterion, optimizer, device, lr, loss_scaler)

        print("start evaluation: ")
        evaluation(best_record, epoch, model, model_without_ddp, val_loader, criterion, device, dataset)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    batch_size = 64
    num_epoch = 20
    lr = 0.0001
    dataset = "LEVIR-CD256"
    root = "/home/wangbowen/DATA/" + dataset + "/"
    main()