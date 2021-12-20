import cv2 as cv
from time import time
from windowcapture import WindowCapture
import win32gui,win32ui,win32con,win32api
import numpy as np

def winEnumHandler( hwnd, ctx ):
    if win32gui.IsWindowVisible( hwnd ):
        print (hex(hwnd), win32gui.GetWindowText( hwnd ))

win32gui.EnumWindows( winEnumHandler, None )
loop_time= time()
capture = WindowCapture('Microsoft Solitaire Collection')


while(True):
    screenshot_data =  capture.get_screenshot()
    screenshot_data= cv.imshow('Computer Vision',np.array(screenshot_data,dtype = np.uint8))

    key =  cv.waitKey(1)
    if key ==ord('f') and screenshot_data != None:
        cv.imwrite('C:/Users/Hari/source/repos/Face_recognition/Face_recognition/Positive/{}.jpg'.format(loop_time),screenshot_data)
    elif key == ord('d') and screenshot_data != None:
        cv.imwrite('C:/Users/Hari/source/repos/Face_recognition/Face_recognition/Negative/{}.jpg'.format(loop_time),screenshot_data)
    elif key == ord('q'):
        cv.destroyAllWindows()
        print('Closed all Windows')
        break

   
    



