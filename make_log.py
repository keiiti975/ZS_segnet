from datetime import datetime
import os
import os.path


class make_log():
    def __init__(self, project_dir, log_name='log'):
        self.name = log_name
        if not os.path.isdir("./log"):
            os.makedirs("./log")
        self.open()
        self.f.write("--- start_time: " +
                     datetime.now().strftime("%Y/%m/%d %H:%M:%S") +
                     " project_dir: " + project_dir + " ---\n")
        self.close()

    def open(self):
        path = "./log/" + self.name + ".txt"
        self.f = open(path, mode="a")

    def write(self, epoch, loss):
        line = "epoch: " + str(epoch) + ",loss: " + str(loss) + "\n"
        self.f.write(line)

    def close(self):
        self.f.close()
