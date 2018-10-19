#!/usr/bin/python
"""
Author: Liang Zhao
"""

import matplotlib.pyplot as plt
#from PIL import Image

im=plt.imread('real_001.eps')
im2=plt.imread('fake_001.eps')
df = abs(im-im2)

f, axarr = plt.subplots(1, 3)
axarr[0].imshow(im); axarr[0].set_title('Real'); axarr[0].axis('off')
axarr[1].imshow(im2); axarr[1].set_title('Fake'); axarr[1].axis('off')
axarr[2].imshow(df); axarr[2].set_title('Diff'); axarr[2].axis('off')
plt.show()


