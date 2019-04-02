import cv2
import numpy as np
import matplotlib.pyplot as plt

# flash image read
img = cv2.imread("images/flash.jpg")
imgf = img.astype(float)
# ---------------------------------------FLASH------------------------------------------------------------------------------


def getflashintensity(colorspace,img):

    if colorspace == "grayscale":

        # flash image intensity (converting grayscale)
        intensityf = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return intensityf

    if colorspace == "lab":

        # flash image get intensity in lab
        imglab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        labintensity = imglab[:, :, 0]
        return labintensity

    if colorspace == "hsv":

        # flash image getting intensity in hsv
        imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsvintensity = imghsv[:, :, 2]
        return hsvintensity

    if colorspace == "formula":
        img = img.astype(float)
        # getting intensity with formula not with grayscale
        pay = (np.multiply(img[:, :, 0], img[:, :, 0]) + np.multiply(img[:, :, 1],img[:, :, 1]) +
               np.multiply(img[:, :, 2], (img[:, :, 2])))

        payda = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2])
        formulaintensity = np.divide(pay, 0.000000000001 + payda)
        formulaintensity.astype(float)
        formulaintensity = formulaintensity.astype('uint8')
        return formulaintensity


intensitygray = getflashintensity("grayscale", img)
intensitylab = getflashintensity("lab", img)
intensityhsv = getflashintensity("hsv", img)
intensityformula = getflashintensity("formula", img)

plt.figure('intensities', figsize=(18, 18))
plt.subplot(221), plt.imshow(intensitygray, cmap='gray')
plt.title('intensity in grayscale'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(intensitylab, cmap='gray')
plt.title('intensity in lab'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(intensityhsv, cmap='gray')
plt.title('intensity in hsv '), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(intensityformula, cmap='gray')
plt.title('intensity with formula '), plt.xticks([]), plt.yticks([])
plt.savefig('results/intensities.jpg')
plt.show()

# ----------------------------------------COLOR LAYER-------------------------------------------------------------------


def getcolorlayer(intensity,colorspace):
    imgg = img.astype(float)
    intensity = intensity.astype(float)
    # getting color layer
    for i in range(3):
        imgg[:, :, i] = np.divide(imgg[:, :, i], 0.000000001 + intensity)

    fcolor = imgg.astype(float)
    cv2.imshow("color layer "+str(colorspace), fcolor)
    fcolor1 = fcolor*255
    fcolor1 = fcolor1.astype(float)
    cv2.imwrite('results/colorlayer'+str(colorspace) + '.jpg', fcolor1)
    cv2.waitKey(0)

    return cv2.normalize(fcolor, 0, 255, cv2.NORM_MINMAX)


colorgray = getcolorlayer(intensitygray,"bgr")
colorformula = getcolorlayer(intensityformula, "formula")
colorhsv = getcolorlayer(intensityhsv, "hsv")
colorlab = getcolorlayer(intensitylab,"lab")

# -----------------------------------------DETAIL---------------------------------------------------------------------
# getting largescale layer and detail of flash with bilateralfilter


def getdetail(intensity, colorspace, d, sigmaColor, sigmaSpace):

    largescaleflash = cv2.bilateralFilter(intensity, d, sigmaColor, sigmaSpace)
    largescaleflash = largescaleflash.astype(float)

    # getting detail with dividing intensity layer and largescale layer
    detail = np.divide(intensity, 0.00000000001 + largescaleflash)

    cv2.imshow("detail " + str(colorspace), detail)
    detail1 = detail * 255
    detail1 = detail1.astype(float)
    cv2.imwrite('results/detail' + str(colorspace) + '.jpg', detail1)
    cv2.waitKey(0)
    return detail


detailgray = getdetail(intensitygray, "bgr", 7, 50,50)
detaillab = getdetail(intensitylab, "formula", 7,50,50)
detailhsv = getdetail(intensityhsv, "hsv",7,50,50)
detailformula = getdetail(intensityformula, "lab",7,50,50)

# so flash image end with color layer and details layer

# ----------------------------------------NO-FLASH---------------------------------------------------------------------

# no-flash image read with image number
img1 = cv2.imread("images/no-flash.jpg")
def getlargescalenoflash(colorspace,d, sigmaColor, sigmaSpace):

    if colorspace == "grayscale":
        # getting intensity with convert to grayscale
        intensitynof = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # getting largescale layer with bilateral filter
        largescalenof = cv2.bilateralFilter(intensitynof, d, sigmaColor, sigmaSpace)
        # converting to float and normalizing
        flargescalenof = largescalenof.astype(float)
        flargescalenof = cv2.normalize(flargescalenof, 0, 255, cv2.NORM_MINMAX)
        return flargescalenof

    if colorspace == "lab":

        # getting intensity with convert to lab
        intensitynof = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        intensitynof = intensitynof[:,:,0]
        # getting largescale layer with bilateral filter
        largescalenof = cv2.bilateralFilter(intensitynof, d, sigmaColor, sigmaSpace)
        # converting to float and normalizing
        flargescalenof = largescalenof.astype(float)
        flargescalenof = cv2.normalize(flargescalenof, 0, 255, cv2.NORM_MINMAX)
        return flargescalenof

    if colorspace == "hsv":

        # getting intensity with convert to hsv
        intensitynof = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        intensitynof = intensitynof[:, :, 2]
        # getting largescale layer with bilateral filter
        largescalenof = cv2.bilateralFilter(intensitynof, d, sigmaColor, sigmaSpace)
        # converting to float and normalizing
        flargescalenof = largescalenof.astype(float)
        flargescalenof = cv2.normalize(flargescalenof, 0, 255, cv2.NORM_MINMAX)
        return flargescalenof


largescalenofinhsv = getlargescalenoflash("hsv",7,50,50)
largescalenofinlab = getlargescalenoflash("lab",7,50,50)
largescalenof = getlargescalenoflash("grayscale",7,50,50)

plt.figure('no-flash large scale layer ', figsize=(18, 9))
plt.subplot(131), plt.imshow(largescalenofinhsv, cmap='gray')
plt.title('largescale layer in hsv'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(largescalenofinlab, cmap='gray')
plt.title('largescale layer in lab'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(largescalenof, cmap='gray')
plt.title('largescale layer in grayscale'), plt.xticks([]), plt.yticks([])
plt.savefig('results/largescales.jpg')
plt.show()


# ------------------------------------------ENHANCING---------------------------------------------------------------
# getting reconstructed image with multiplying detail, largescale and color layers


def enhancing(colorlayer,largescale,detail,colorspace):
    largexdetail = np.multiply(largescale, detail)
    reconstructed = np.ones(colorlayer.shape, dtype=float)

    for i in range(3):
        reconstructed[:, :, i] = np.multiply(largexdetail, colorlayer[:, :, i])

    # normalization
    result = cv2.normalize(reconstructed, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow("result " + str(colorspace), result)
    result1 = result*255
    result1 = result1.astype(float)
    cv2.imwrite('results/result' + str(colorspace) + '.jpg', result1)
    cv2.waitKey(0)
    return result


resultformula = enhancing(colorformula, largescalenof, detailformula,"formula")
resultgray = enhancing(colorgray, largescalenof, detailgray, "grayscale")
resultlab = enhancing(colorlab, largescalenofinlab, detaillab, "lab")
resulthsv = enhancing(colorhsv, largescalenofinhsv, detailhsv, "hsv")