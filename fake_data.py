
import numpy as np
import cv2
import pickle
import constants as C


def transform_image(image, angle=C.ANGLE, translation=C.TRANSLATION):
    height, width, channels = image.shape
    center = (width // 2, height // 2)
    image = cv2.warpAffine(image, cv2.getRotationMatrix2D(center, np.random.uniform(-angle, angle), 1), (width, height))
    image = cv2.warpAffine(image, np.array([[1, 0, translation * width * np.random.uniform(-1, 1)],
                                            [0, 1, translation * height * np.random.uniform(-1, 1)]]), (width, height))
    return image

def data_aug(source, destination, count=C.COUNT):
    with open(source, mode='rb') as f:
        source_data = pickle.load(f)
    source_X, source_Y = source_data['features'], source_data['labels']
    for i in range(count):
        rand_idx = np.random.randint(source_X.shape[0])
        image = transform_image(source_X[rand_idx])
        if i == 0:
            augmented_X = np.expand_dims(image, axis=0)
            augmented_Y = np.array([source_Y[rand_idx]])
        else:
            augmented_X = np.concatenate((augmented_X, np.expand_dims(image, axis=0)))
            augmented_Y = np.append(augmented_Y, source_Y[rand_idx])
    augmented_X = np.concatenate((source_X, augmented_X))
    augmented_Y = np.concatenate((source_Y, augmented_Y))

    new_data = {'features': np.concatenate((source_X, augmented_X)), 'labels': np.concatenate((source_Y, augmented_Y))}
    print(count, " Images Augmented and Added to the dataset")
    with open(destination, mode='wb') as f:
        pickle.dump(new_data, f)
    return new_data

data_aug('train.p', 'train_aug.p')
