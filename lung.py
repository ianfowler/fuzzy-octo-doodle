import matplotlib.pyplot as plt
import numpy as np

import unet
from unet import utils
import tensorflow as tf
import data_preparation

imgs, segs = data_preparation.load_data(
    "JSRT_imgs", "JSRT_segs", relpath="./prepared_data/")

HOLDOUT = 30
train_examples, train_labels = imgs[:-HOLDOUT], segs[:-HOLDOUT]
test_examples, test_labels = imgs[-HOLDOUT:], segs[-HOLDOUT:]

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_examples, train_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (test_examples, test_labels))

#  Construct
unet_model = unet.build_model(channels=1,
                              num_classes=4,
                              layer_depth=5,
                              filters_root=64)
unet.finalize_model(unet_model)


# Train
trainer = unet.Trainer(checkpoint_callback=False)
trainer.fit(unet_model,
            train_dataset,
            validation_dataset,
            epochs=25,
            batch_size=16)


# Plot a prediction
prediction = unet_model.predict(validation_dataset.batch(batch_size=3))
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 10))
dataset = validation_dataset.map(
    utils.crop_image_and_label_to_shape(prediction.shape[1:]))

for i, (image, label) in enumerate(dataset.take(3)):
    ax[i][0].matshow(image[..., -1])
    ax[i][0].set_title('Original Image')
    ax[i][0].axis('off')
    ax[i][1].matshow(np.argmax(label, axis=-1), cmap=plt.cm.gray)
    ax[i][1].set_title('Original Mask')
    ax[i][1].axis('off')
    ax[i][2].matshow(np.argmax(prediction[i, ...], axis=-1), cmap=plt.cm.gray)
    ax[i][2].set_title('Predicted Mask')
    ax[i][2].axis('off')
plt.tight_layout()
plt.savefig("pred")

# Save
unet_model.save("unet_deep")

# Import
# from unet import custom_objects
# reconstructed_model = tf.keras.models.load_model("unet-5", custom_objects=custom_objects)
