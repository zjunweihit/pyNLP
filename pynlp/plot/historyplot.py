import matplotlib.pyplot as plt
import numpy as np

class HistoryPlot:
    def __init__(self, epochs, style = "ggplot"):
        self.style = style
        self.epochs = epochs


    def show(self, H):
        plt.style.use(self.style)
        plt.figure()
        plt.plot(np.arange(0, self.epochs), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, self.epochs), H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()

