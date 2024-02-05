import numpy as np
from keras import Model, models
from matplotlib import pyplot as plt

model = models.load_model("gan_mnist")
r, c = 5,5
noise = np.random.normal(0, 1, size=(r*c, 100))
gen_imgs = model.predict(noise)
gen_imgs = gen_imgs * 0.5 + 0.5
fig, axs = plt.subplots(r, c)
idx = 0
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(gen_imgs[idx, :,:,0], cmap="gray")
        axs[i, j].axis(False)
        idx += 1
plt.show()
