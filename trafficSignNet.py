from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D,\
     MaxPooling2D, Activation, Flatten, Dropout, Dense


class TrafficSignNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        model.add(Conv2D(16, 3, padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D())
        # first set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(32, 3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D())
        # second set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, 3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D())
        # first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        # second set of FC => RELU layers
        # model.add(Flatten())
        # model.add(Dense(128))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network
        return model

