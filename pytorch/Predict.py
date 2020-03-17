import numpy as np
import json

class ILSVRCPredictor:
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        argmax = np.argmax(out.detach().numpy())
        print(argmax)
        maxid_name = self.class_index[str(argmax)][1]

        return maxid_name

if __name__ == '__main__':
    class_label = json.load(open('imagenet_class_index.json', 'r'))