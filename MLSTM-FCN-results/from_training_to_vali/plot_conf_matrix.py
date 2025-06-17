import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    targets = np.load("naovals50epochs_fromtrain.npy")
    preds = np.load("naopreds50epochs_fromtrain.npy")
    targets = np.argmax(targets, axis = 1)
    preds = np.argmax(preds, axis = 1)
    cf = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cf)
    disp.plot()
    plt.show()

if __name__ == "__main__":
    main()
