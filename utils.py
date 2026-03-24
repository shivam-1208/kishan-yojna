import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load CNN once
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')


def extract_glcm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

    props = ['contrast','dissimilarity','homogeneity','energy','correlation']
    return [graycoprops(glcm, p)[0,0] for p in props]


def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0,10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist.tolist()


def extract_color(img):
    mean = cv2.mean(img)[:3]
    std  = np.std(img, axis=(0,1))
    return list(mean) + list(std)


def extract_cnn(img):
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feat = base_model.predict(img, verbose=0)
    return feat.flatten()


def extract_features(img):
    return (
        extract_glcm(img)
        + extract_lbp(img)
        + extract_color(img)
        + list(extract_cnn(img))
    )