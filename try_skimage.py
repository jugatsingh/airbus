from skimage import io, filters, color, feature
from scipy import ndimage
from matplotlib import pyplot as plt


filename = 'image002.jpg'
im = io.imread(filename)
im_gray = color.rgb2gray(im) #convert to greyscale

#threshold
val = filters.threshold_otsu(im_gray)
binary = im_gray <= val
drops = ndimage.binary_fill_holes(im_gray>val)

edges1 = feature.canny(im_gray, sigma=0.5)
edges2 = feature.canny(im_gray, sigma=3)
# io.imshow(sheet)
# io.imshow(im)
# io.show()
# io.imshow(binary)

# io.imshow(drops, cmap='gray')
# io.show()



fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()

plt.show()
