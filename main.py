import argparse
import cv2
import numpy as np
import onnxruntime

class LYTNet:
    def __init__(self, modelpath):
        # Initialize model
        # self.net = cv2.dnn.readNet(modelpath) ####opencv-dnn读取失败
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.net = onnxruntime.InferenceSession(modelpath, so)
        self.input_height = self.net.get_inputs()[0].shape[1]   ####(1,h,w,3)
        self.input_width = self.net.get_inputs()[0].shape[2]
        self.input_name = self.net.get_inputs()[0].name

    def detect(self, srcimg): 
        input_image = cv2.resize(srcimg, (self.input_width, self.input_height))
        input_image = input_image.astype(np.float32) / 127.5 - 1.0
        blob = np.expand_dims(input_image, axis=0).astype(np.float32)

        result = self.net.run(None, {self.input_name: blob})

        # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
        output_image = np.squeeze(result[0])
        output_image = (output_image + 1.0 ) * 127.5
        output_image = output_image.astype(np.uint8)
        output_image = cv2.resize(output_image, (srcimg.shape[1], srcimg.shape[0]))
        return output_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str,
                        default='testimgs/4_1.JPG', help="image path")
    parser.add_argument('--modelpath', type=str,
                        default='weights/lyt_net_lolv2_real_320x240.onnx', help="model path")
    args = parser.parse_args()

    mynet = LYTNet(args.modelpath)
    srcimg = cv2.imread(args.imgpath)

    dstimg = mynet.detect(srcimg)

    if srcimg.shape[0] >= srcimg.shape[1]:
        boundimg = np.zeros((srcimg.shape[0], 10, 3), dtype=srcimg.dtype)+255
        combined_img = np.hstack([srcimg, boundimg, dstimg])
    else:
        boundimg = np.zeros((10, srcimg.shape[1], 3), dtype=srcimg.dtype)+255
        combined_img = np.vstack([srcimg, boundimg, dstimg])
    
    winName = 'Deep Learning use OpenCV-dnn'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, combined_img)  ###原图和结果图也可以分开窗口显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()