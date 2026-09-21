from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("PaddleOCR loaded successfully!")
print(cv2.__version__)