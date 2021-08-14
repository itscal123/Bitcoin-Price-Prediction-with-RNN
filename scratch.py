import tensorflow as tf
from tensorflow import keras
from preprocess_data import preprocess
from preprocess_data import get_data


def print_dataset(ds):
    first = True
    for inputs, targets in ds:
        if first:
            print("---Batch---")
            print("Feature:", inputs.numpy())
            print("Label:", targets.numpy())
            print("")
            first = False
        else:
            break
 


if __name__ == "__main__":
    train, test = get_data()
    for batch in train.take(4):
        print([arr.numpy() for arr in batch])

