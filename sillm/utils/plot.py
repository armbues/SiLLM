import matplotlib.pyplot as plt

class Plot:
    def __init__(self,
                 title="Training and Validation Losses",
                 ):
        """
        Initialize a plot for training and validation losses.
        Args:
            title: Title of the plot
        """
        self.title = title
        
        self.train_x = []
        self.train_y = []
        self.valid_x = []
        self.valid_y = []

    def add_train_loss(self, i, loss):
        """
        Add a training loss to the plot.
        Args:
            i: Step number
            loss: Training loss
        """
        self.train_x.append(i)
        self.train_y.append(loss)

    def add_valid_loss(self, i, val_loss):
        """
        Add a validation loss to the plot.
        Args:
            i: Step number
            val_loss: Validation loss
        """
        self.valid_x.append(i)
        self.valid_y.append(val_loss)

    def plot(self):
        """
        Plot the training and validation losses.
        """
        plt.figure(figsize=(10, 6))

        ticks = list(set(self.train_x + self.valid_x))
        plt.xticks(ticks=ticks)

        plt.plot(self.train_x, self.train_y, label='Training Loss')
        plt.plot(self.valid_x, self.valid_y, label='Validation Loss')

        plt.title(self.title)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()

    def save(self, fpath):
        """
        Save the plot to a file.
        Args:
            fpath: File path
        """
        self.plot()
        plt.savefig(fpath)

    def show(self):
        """
        Show the plot.
        """
        self.plot()
        plt.show()