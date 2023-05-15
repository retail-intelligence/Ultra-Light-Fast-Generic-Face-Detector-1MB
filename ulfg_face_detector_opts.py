import argparse

class ULFGFaceDetectorOpts:
    """Class for parsing arguments.
    """

    def __init__(self):
        """Initializer function.
        """        
        
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--input_size', type=int, default=640, help='Define network input size, default optional value 128/160/320/480/640/1280')
        self.parser.add_argument('--net_type', type=str, default="RFB", help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
        self.parser.add_argument('--threshold', type=float, default=0.6, help='Score threshold')
        self.parser.add_argument('--candidate_size', type=int, default=1500, help='Nms candidate size')
        self.parser.add_argument('--test_device', type=str, default='cuda:0', help='cuda:0 or cpu')
        self.parser.add_argument('--img', type=str, default='test_imgs', help='Image or directory containing images')

        self.face_detector_opts = self.parser.parse_args()
    
    def parse(self):
        """Returns parsed arguments.

        :return: parsed arguments
        :rtype: object
        """        

        return self.face_detector_opts