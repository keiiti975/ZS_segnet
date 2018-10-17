

class make_log():
    def __init__(self, log_name='log'):
        self.name = log_name

    def open(self):
        path = "./log/" + self.name + ".txt"
        self.f = open(path, mode="w")

    def write(self, epoch, loss):
        line = "epoch: " + str(epoch) + ",loss: " + str(loss) + "\n"
        self.f.write(line)

    def close(self):
        self.f.close()
