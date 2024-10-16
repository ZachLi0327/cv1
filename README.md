The main function of this project is text detecting,press run button after you make sure you did the following three steps,you will see the pretreated picture and treated picture with bounding box around the text in new windows,you can press any button to close the windows,and the detected text will showed in terminal below.

First step, add the picture you want to detect to the folder"venv".

Secondly, in line 87,you can change the picture need to be detected,(image = cv2.imread('order.jpg')),you can change the order.jpg to your picture.

Thirdly,according to the types of picture ,you can change the tesseract config in line 89,--oem 3: This stands for “OCR Engine Mode” and the value 3 means to use both the LSTM neural net (Long Short-Term Memory) and the legacy Tesseract engine for OCR. This is often the best option for a balance between accuracy and speed.
--psm 6: This stands for “Page Segmentation Mode” and the value 6 means to assume a single uniform block of text.

The additional functions are written in the bottom of the project,

For the function onlynumbers,you can change the picture in line 121, after the execution, the numbers in the picture will be showed in the terminal below.

For the function read_text_from_image, you can change the picture in line 122,after the execution, you will find a new file called result.txt in venv folder.

There are unused two picture called 百度.png and 古诗.webp in the folder venv, you can change the detected picture in line 87 to test the result of detecting Chinese.
