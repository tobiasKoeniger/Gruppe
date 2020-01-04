 
pic0 = invert_image(img[10])
pic1 = invert_image(img[15])

pic_res = pic0 + pic1
pic_res = invert_image(pic_res)


pic_flip = add_images(pic_flip, pic_res)


# Translation
rows, cols = pic_res.shape

x_translate = 100
y_translate = 50

M = np.float32([[1,0,x_translate], [0,1,y_translate]])
pic_res = cv2.warpAffine(pic_res, M, (cols,rows), borderMode = cv2.BORDER_REPLICATE)


print()
print("Verschiebungsmatrix: ")
print(M)
