import sys
import cv2
import os
import wget

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision.ssd.config.fd_config import define_img_size

class ULFGFaceDetector:
    """Class containing method for detecting faces in images.
    """    

    def __init__(self, weights_path='', input_size=640, net_type="RFB", threshold=0.6, candidate_size=1500, test_device="cuda:0"):
        """Initializer function.

        :param input_size: define network input size, default
         optional value 128/160/320/480/640/1280
        :type input_size: 640
        :param net_type: The network architecture ,optional:
         RFB (higher precision) or slim (faster)
        :type net_type: str
        :param threshold: score threshold
        :type threshold: float
        :param candidate_size: nms candidate size
        :type candidate_size: int
        :param test_device: cuda:0 or cpu
        :type test_device: str
        """        
        
        if weights_path != '':
            if not os.path.exists(weights_path):
                raise FileExistsError(f'Weights not found: {weights_path}')
            if not weights_path.endswith('.pth'):
                raise TypeError('Only .pth format is accepted for this model')
        else:
            DEFAULT_WEIGHTS_PATH = os.getcwd() + '/models/face_detection/'
            
            if net_type == 'RFB':
                MODEL_URL = 'https://rtvad.blob.core.windows.net/models/version-RFB-320.pth'
                weights_path = os.path.join(DEFAULT_WEIGHTS_PATH,'version-RFB-320.pth')
            if net_type == 'slim':
                MODEL_URL = 'https://rtvad.blob.core.windows.net/models/version-slim-320.pth'
                weights_path = os.path.join(DEFAULT_WEIGHTS_PATH,'version-slim-320.pth')
            if not os.path.exists(weights_path):
                if not os.path.exists(DEFAULT_WEIGHTS_PATH):    
                    os.makedirs(DEFAULT_WEIGHTS_PATH)         
                print('\nDownloading face detection model weights:')
                wget.download(MODEL_URL, DEFAULT_WEIGHTS_PATH)
        
        define_img_size(input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

        label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voc-model-labels.txt")
        test_device = test_device

        class_names = [name.strip() for name in open(label_path).readlines()]

        self.candidate_size = candidate_size

        if net_type == 'slim':
            from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
            model_path = os.path.join(os.getcwd(), "models/face_detection/version-slim-320.pth")
            # model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/pretrained/version-slim-640.pth")
            net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
            self.predictor = create_mb_tiny_fd_predictor(net, candidate_size=self.candidate_size, device=test_device)
        elif net_type == 'RFB':
            from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
            model_path = os.path.join(os.getcwd(), "models/face_detection/version-RFB-320.pth")
            # model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/pretrained/version-RFB-640.pth")
            net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
            self.predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=self.candidate_size, device=test_device)
        else:
            print("The net type is wrong!")
            sys.exit(1)
        net.load(model_path)
        
        self.threshold = threshold


    def detect_faces(self, img):
        """Detects faces in img.

        :param img: image for detecting faces
        :type img: numpy array or str
        :return: list of lists containing bboxes of
         detected faces
        :rtype: list
        """        

        if type(img) == str:
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # if image not in RGB mode
        boxes, labels, probs = self.predictor.predict(img, self.candidate_size / 2, self.threshold) # image must be in RGB color mode
        
        return boxes.tolist()
