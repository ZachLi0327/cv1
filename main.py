import numpy as np #添加数学库
import cv2 #添加视觉库
import pytesseract #添加ocr接口
from pytesseract import Output
import shutil#拷贝图像的库
from langdetect import detect_langs#可以检测语言种类的库

#拷贝一个原图片
shutil.copy2('/Users/lizhuochen/Desktop/cv/order.jpg', '/Users/lizhuochen/Desktop/cv/order1.jpg')

# get grayscale image 将图片转化为灰色
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal 图片降噪
def remove_noise(image):
    return cv2.medianBlur(image,5)

#thresholding 设置阈值，从而将图片分为前景和后景
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation #膨胀操作的目的是扩大图像中的前景区域（通常是白色像素）来增加物体的尺寸，或是用来填充物体内部的小孔或连接接近的物体。
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

#erosion #腐蚀是一种侵蚀图像前景边界的操作，用于消除边界点，从而减少对象尺寸。
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation  形态学开运算,先膨胀后腐蚀
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#Canny边缘检测是一种经典且常用的边缘检测算法
def canny(image):
    return cv2.Canny(image, 100, 200)

#对图像进行倾斜矫正，如果检测到图像倾斜，则会对图像进行翻转
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
#在图像文字识别的基础上新增只识别数字功能，用pytesseract的image_to_string函数，但是将参数修改为只识别数字
def onlynumbers(img):
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    print(pytesseract.image_to_string(img, config=custom_config))
#整合cv2的图像处理库将读取的文字输入到txt
def read_text_from_image(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#创建一个18x18的矩形结构元素作为内核
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

  dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)

  contours, hierachy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  image_copy = image.copy()

  for contour in contours:#检测每一个文字
    x, y, w, h = cv2.boundingRect(contour)

    cropped = image_copy[y : y + h, x : x + w]

    file = open("results.txt", "a")#打开文件用于追加

    text = pytesseract.image_to_string(cropped)#

    file.write(text)
    file.write("\n")

  file.close()

image = cv2.imread('order.jpg')#读取order图片,可以在这里修改需要识别的图片
image1=cv2.imread('order1.jpg')
custom_config = r'--oem 3 --psm 6' #--oem 3：默认OCR引擎，通常为最好的选择 --psm 6：自动分段
gray = get_grayscale(image) #将图像转化为灰色，其他的操作都需要灰图
thresh = thresholding(gray) #获取二值化的图像
noise=remove_noise(thresh)#去除图片中的噪声
opening = opening(noise)#经过开运算的图像
canny = canny(opening)#边缘检测的结果,canny就是图片预处理的最终结果
string1=pytesseract.image_to_string(canny,lang='eng+chi_tra+chi_sim')#额外安装了简体繁体中文的训练集，将检测语言设置为英文和简体繁体中文
print(string1)#提取预处理图片的文字并输出
cv2.imshow('pretreated image',canny)
cv2.waitKey(0)

# 获取图像的高度、宽度和通道数
h, w, c = image.shape
boxes = pytesseract.image_to_boxes(image)
for b in boxes.splitlines():
    b = b.split(' ')
    image= cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
cv2.imshow('Image with Text Boxes',image)#检测图像中的文字并用边界框标注
cv2.waitKey(0)

d = pytesseract.image_to_data(image1, output_type=Output.DICT)
print(d.keys())#image_to_data 函数来获取图像中文字的详细信息，并将结果以字典的形式返回

n_boxes = len(d['text'])
#遍历由 pytesseract.image_to_data 返回的字典，并且对于每个识别出的文字框
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        image1 = cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('image with word boxes',image1)#展示单词被边界框标出的图片
cv2.waitKey(0)
cv2.destroyAllWindows()
onlynumbers('order.jpg')#展示图片中检测到的数字
read_text_from_image(image)#读取图片中的文字并将文字写入txt