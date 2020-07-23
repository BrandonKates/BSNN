import math

class _TempScheduler():
    def __init__(self, temps):
        self.temps = temps
        self.time_step = 0
       
    def step(self):
        self.time_step += 1

    def avg_temp(self):
        vals = list(map(lambda _: _.val, self.temps))
        return sum(vals)/len(vals)

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
            sef.set = True
