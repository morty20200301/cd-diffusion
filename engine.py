import numpy as np

from tools import AverageMeter, ProgressMeter, IouCal
import torch
import math
import sys


def train_model(epoch, num_epoch, model, train_loader, criterion, optimizer, device, lr_, loss_scaler):
    model.train()
    train_main_loss = AverageMeter('Train Main Loss', ':.5')
    train_aux_loss = AverageMeter('Train Aux Loss', ':.5')
    lr = AverageMeter('lr', ':.5')
    L = len(train_loader)
    curr_iter = epoch * L
    record = [lr, train_main_loss, train_aux_loss]

    progress = ProgressMeter(L, record, prefix="Epoch: [{}]".format(epoch))
    accum_iter = 1

    for data_iter_step, data in enumerate(train_loader):
        img_a = data['A']
        img_b = data['B'].to(device, dtype=torch.float32)
        mask = data['L'].to(device, dtype=torch.int64)
        index = data['Index']

        optimizer.param_groups[0]['lr'] = lr_ * (1 - float(curr_iter) / (num_epoch * L)) ** 0.9
        # if data_iter_step % accum_iter == 0:
        #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            outputs, aux = model(img_b)
            main_loss = criterion(outputs, mask)
            aux_loss = criterion(aux, mask)
            loss = main_loss + 0.4 * aux_loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        train_main_loss.update(main_loss.item())
        train_aux_loss.update(aux_loss.item())
        lr.update(optimizer.param_groups[0]['lr'])

        curr_iter += 1

        if data_iter_step % 5 == 0:
            progress.display(data_iter_step)


@torch.no_grad()
def evaluation(best_record, epoch, model, model_without_ddp, val_loader, criterion, device, dataset):
    model.eval()
    val_loss = AverageMeter('Val Main Loss', ':.4')
    progress = ProgressMeter(len(val_loader), [val_loss], prefix="Epoch: [{}]".format(epoch))
    acc = 0
    ll = len(val_loader)
    for i_batch, data in enumerate(val_loader):
        img_a = data['A']
        img_b = data['B'].to(device, dtype=torch.float32)
        mask = data['L'].to(device, dtype=torch.int64)
        index = data['Index']
        b, w, h = mask.shape

        outputs = model(img_b)
        _, preds = torch.max(outputs, 1)

        val_loss.update(criterion(outputs, mask).item())

        if i_batch % 5 == 0:
            progress.display(i_batch)

        preds = preds.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        acc += np.sum(preds == mask)/b/w/h

    acc_ = acc/ll

    if acc_ > best_record['acc']:
        best_record['acc'] = acc_
        print("best acc is:", epoch, acc_)

    torch.save(model_without_ddp.state_dict(), "saved_model/" + dataset + "_" + str(epoch) + ".pt")
