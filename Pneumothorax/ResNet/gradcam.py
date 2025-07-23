import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size, color_mode='rgb')
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = array / 255.
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title('Grad-CAM')
    plt.imshow(cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def main():
    MODEL_PATH = 'pneumothorax_resnet50_finetuned.h5'
    IMG_PATH = 'data/preprocessed/1.2.276.0.7230010.3.1.4.8323329.300.1517875162.258081_aug0.png'  # Change to a test image
    model = load_model(MODEL_PATH)
    img_array = get_img_array(IMG_PATH, size=(256,256))
    last_conv_layer_name = 'conv5_block3_out'  # Last conv layer in ResNet50
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(IMG_PATH, heatmap, cam_path='gradcam_result.jpg')

if __name__ == '__main__':
    main()
