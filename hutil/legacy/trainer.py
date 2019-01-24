import time
import math
import os

from tqdm import tqdm
import arrow
import itchat

import torch
from torch.utils.data import DataLoader

from hutil.summary import summary
from hutil.common import cuda

class Trainer(object):
    def __init__(self, net, loss, optimizer, lr_scheduler=None, save_path=None, name='net'):
        super(Trainer, self).__init__()
        self.name = name
        self.net = net
        self.save_path = save_path
        self.criterion = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.print_callback = []
        self.weixin_logined = False
        self.epochs = 0

    def login_weixin(self, save_path='./'):
        itchat.logout()
        itchat.auto_login(hotReload=True, enableCmdQR=2,
                          statusStorageDir=os.path.join(save_path, 'weixin.pkl'))
        self.weixin_logined = True

    def logout_weixin(self):
        itchat.logout()
        self.weixin_logined = False

    def send_weixin(self, msg):
        itchat.send(msg, toUserName='filehelper')

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def fprint(self, msg):
        print(msg)
        for f in self.print_callback:
            try:
                f(msg)
            except Exception as e:
                pass

    def fit(self, dataset, batch_size=32, epochs=20, save_per_epochs=-1, val_dataset=None, val_batch_size=None, send_weixin=False):
        if send_weixin:
            if self.weixin_logined:
                self.print_callback = [self.send_weixin]
            else:
                print("Weixin not logged in")
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        m = len(dataset)
        num_batches = math.ceil(m / batch_size)
        loss_avgs = []
        accs = []
        for epoch in range(1, epochs + 1):
            self.fprint("Epoch %d/%d" % (epoch, epochs))
            # if verbose:
                # bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
                # pbar = tqdm(total=m, bar_format=bar_format,
                            # ascii=True, unit=' samples')
            # else:
            start = time.time()
            if self.lr_scheduler:
                self.lr_scheduler.step()

            loss_avg = 0
            n_correct = 0

            i = 0
            for batch in data_loader:
                i += 1

                *batch_x, batch_y = batch
                batch_x = [ cuda(x) for x in batch_x ]
                batch_y = cuda(batch_y)

                output = self.net(*batch_x)
                loss = self.criterion(output, batch_y)

                loss_avg = (loss_avg * (i - 1) + loss.item()) / i
                pred = torch.argmax(output, dim=1)
                n_correct += (pred == batch_y).sum().item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # if verbose:
                    # total = min(i*batch_size, m)
                    # accs.append(accs)
                    # pbar.set_postfix(loss=loss_avg, acc="%.2f" % acc)
                    # pbar.update(len(batch_x))
            loss_avgs.append(loss_avg)
            self.epochs += 1
            # if verbose:
                # pbar.close()
            # else:
            acc = n_correct / m
            accs.append(acc)
            if val_dataset is None:
                elapsed = int(time.time() - start)
                msg = 'Elapsed: %2ds   Loss: %f   Accuracy: %.3f' % (
                    elapsed, loss_avgs[-1], acc)
                self.fprint(msg)
            else:
                val_acc, val_loss = self.evaluate(
                    val_dataset, val_batch_size)
                elapsed = int(time.time() - start)
                msg = 'Elapsed: %2ds   Loss: %.4f   Accuracy: %.3f   Val_Acc: %.3f   Val_Loss: %.4f' % (
                    elapsed, loss_avgs[-1], acc, val_acc, val_loss)
                self.fprint(msg)
            if save_per_epochs != -1 and epoch % save_per_epochs == 0:
                now = arrow.utcnow().to('Asia/Shanghai').format('YYMMDDHHmm')
                model_path = os.path.join(
                    self.save_path, "%s-%d-%s.pth" % (self.name, self.epochs, now))
                self.save(model_path)
                self.fprint('Model saved to %s' % model_path)
        return loss_avgs, accs

    def evaluate(self, dataset, batch_size):
        batch_size = batch_size or 128
        self.net.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        n_correct = 0
        loss_avg = 0
        i = 0
        with torch.no_grad():
            for batch in data_loader:
                i += 1
                *batch_x, batch_y = batch
                batch_x = [ cuda(x) for x in batch_x ]
                batch_y = cuda(batch_y)

                output = self.net(*batch_x)
                loss = self.criterion(output, batch_y).item()

                loss_avg = (loss_avg * (i - 1) + loss) / i
                pred = torch.argmax(output, dim=1)
                n_correct += (pred == batch_y).sum().item()

        self.net.train()
        return n_correct / len(dataset), loss_avg

    def summary(self, input_size, dtype=None):
        return summary(self.net, input_size, dtype=dtype)
