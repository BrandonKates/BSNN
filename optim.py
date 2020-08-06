import math

class _TempScheduler():
    def __init__(self, temps):
        self.temps = temps
        self.time_step = 0
       
    def step(self):
        self.time_step += 1

class TempVar():
    def __init__(self, val=-math.inf):
        self.val = val

    def __repr__(self):
        return self.val.__repr__()


class JangScheduler(_TempScheduler):
    def __init__(self, temps, N, r, limit):
        super(JangScheduler, self).__init__(temps)
        self.N, self.r, self.limit = N, r, limit
        self.step()

    def step(self):
        super(JangScheduler, self).step()
        tau = max(self.limit, math.exp(-self.r*math.floor(self.time_step/self.N)))
        for temp in self.temps:
            temp.val = tau


class AdaScheduler(_TempScheduler):
    def __init__(self, temps, base_temp, factor=.5, patience=10,
            threshold=.01, cooldown=0, max_temp=2.):
        super(AdaScheduler, self).__init__(temps)
        self.base_temp = base_temp
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.in_cooldown = False
        self.cooldown_epochs = 0
        self.patience_epochs = 0
        self.best_loss = -1
        self.max_temp = max_temp

        for temp in self.temps:
            temp.val = self.base_temp

    def adjust(self, loss):
        if self.best_loss == -1:
            self.best_loss = loss
            return

        if self.in_cooldown:
            self.cooldown_epochs += 1
            if self.cooldown_epochs > self.cooldown:
                self.cooldown_epochs = 0
                self.in_cooldown = False
        else:
            if loss < self._threshold():
                self.best_loss = loss
                self.patience_epochs = 0
            else:
                self.patience_epochs += 1
                if self.patience_epochs > self.patience:
                    self._adjust_temps()
                    self.patience_epochs = 0
                    self.in_cooldown = True

    def _threshold(self):
        return self.best_loss * (1 - self.threshold)

    def _adjust_temps(self):
        for temp in self.temps:
            temp.val += self.factor
            if temp.val > self.max_temp:
                temp.val = self.max_temp

     
class ConstScheduler(_TempScheduler):
    def __init__(self, temps, const):
        super(ConstScheduler, self).__init__(temps)
        self.const = const
        self.set = False
        self.step()

    def step(self):
        super(ConstScheduler, self).step()
        if self.set:
            return
        else:
            for temp in self.temps:
                temp.val = self.const
            self.set = True
