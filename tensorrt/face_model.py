import cv2
import numpy as np
from common.image import ImageData
from common import face_preprocess
from core.face_processors import FaceEmbedding, FaceEmbeddingBatchTRT
from core.face_detectors import RetinafaceBatchTRT, SCRFDBatchTRT

import time

class FaceModelBase:
    def __init__(self):
        self.embedd_model = None
        self.detector_input_size = None
        self.detector = None
        self.embedd_model = None
        self.embedding_batch_size = None

    def visualize_detect(self, base_image, bboxes, points):
        '''
            Visualize detection
        '''
        vis_image = base_image.copy()
        for i in range(len(bboxes)):
            pt1 = tuple(map(int, bboxes[i][0:2]))
            pt2 = tuple(map(int, bboxes[i][2:4]))
            cv2.rectangle(vis_image, pt1, pt2, (0, 255, 0), 1)
            for lm_pt in points[i]:
                cv2.circle(vis_image, tuple(map(int, lm_pt)), 3, (0, 0, 255), 3)
        return vis_image

    def get_inputs(self, bgr_image, threshold = 0.8, polygons = None):
        '''
            Get boxes & 5 landmark points
            Input:
                - bgr_image: BGR image
            Output:
                - bboxes: face bounding boxes
                - points: 5 landmark points for each cor-response face
        '''
        # vis_image = bgr_image.copy()
        # image = ImageData(bgr_image, self.detector_input_size)
        # image.resize_image(mode='pad')
        assert type(bgr_image).__name__ == 'ndarray', "Input must be numpy array: {}".format(type(bgr_image).__name__)  
        assert len(bgr_image.shape) == 3, "Input must be single BGR image"   
        bboxes, points = self.detector.detect([bgr_image], threshold=threshold, polygons = polygons)
        bboxes = bboxes[0]
        points = points[0]
        # if len(bboxes) > 0:
        #     # Post processing
        #     bboxes = bboxes[:, :4]
        #     bboxes /= image.scale_factor
        #     points /= image.scale_factor
        # del image
        return bboxes, points

    @staticmethod
    def get_face_align(bgr_img, bboxes, points, image_size='112,112'):
        '''
            Align face from given bounding boxes and landmark points
        '''
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        aligned = []
        for i in range(len(bboxes)):
            bbox_i = bboxes[i]
            points_i = points[i]
            nimg = face_preprocess.preprocess(rgb_img, bbox_i, points_i, image_size=image_size)
            aligned.append(nimg)
        return aligned

    def get_features(self, aligned, flip = 0):
        '''
            Extract embedding for face
            Input:
                - aligned: batch of RGB aligned images
        '''
        return self.embedd_model.get_features(aligned, batch_size = self.embedding_batch_size, flip = flip)


class FaceModelBatchTRT(FaceModelBase):
    '''
        Face model with batch inference (both Retinaface + Embedding)
    '''
    def __init__(self,
        detector_batch_size = 1,
        detector_weight = 'weights/retinaface_r50_v1-1x3x640x640-fp16.trt',
        detector_input_size = (640, 640), 
        detector_post_process_type = 'SINGLE',
        detector_engine = 'TRT',
        detector_model  = 'RETINA',
        detector_triton_input_names = None,
        detector_triton_output_names = None,
        detector_triton_model_name = None,
        embedding_batch_size = 4,
        embedding_weight = "weights/resnet124-batchsize_8-fp32.trt",
        embedding_input_size = (112, 112),
        embedding_engine = 'TRT',
        embedding_triton_input_names = None,
        embedding_triton_output_names = None,
        embedding_triton_model_name = None):
        '''
            Init Detector & Embedding Extractor
        '''
        super(FaceModelBatchTRT, self).__init__()
        self.detector_input_size = detector_input_size
        self.detector = None
        if detector_weight is not None:
            if detector_model == 'RETINA':
                self.detector = RetinafaceBatchTRT(model_path = detector_weight, \
                                                batch_size = detector_batch_size,\
                                                input_shape = detector_input_size, \
                                                post_process_type = detector_post_process_type,
                                                engine = detector_engine,
                                                triton_input_names = detector_triton_input_names,
                                                triton_output_names = detector_triton_output_names,
                                                triton_model_name = detector_triton_model_name)
                self.detector.prepare(nms=0.4)
            elif detector_model == 'SCRFD':
                self.detector  = SCRFDBatchTRT(model_path=detector_weight,
                                                    input_shape = detector_input_size,
                                                    batch_size = detector_batch_size, # Current support only batchsize = 1
                                                    engine = detector_engine,
                                                    triton_input_names = detector_triton_input_names,
                                                    triton_output_names = detector_triton_output_names,
                                                    triton_model_name = detector_triton_model_name)
        self.embedd_model = None
        if embedding_weight is not None:
            self.embedd_model = FaceEmbeddingBatchTRT(trt_model_path = embedding_weight, \
                                            batch_size = embedding_batch_size, \
                                            input_shape=embedding_input_size,
                                            engine = embedding_engine,
                                            triton_input_names = embedding_triton_input_names,
                                            triton_output_names = embedding_triton_output_names,
                                            triton_model_name = embedding_triton_model_name)
                                            


    def get_inputs_batch(self, list_bgr_image, threshold = 0.6, polygons = None):
        '''
            Batch-processing: get boxes & 5 landmark points
            Input:
                - list_bgr_image: list BGR image [B x W x H x 3]
            Output:
                - bboxes: face bounding boxes [B x N x 4]
                - points: 5 landmark points for each cor-response face [B x N x 5 x 2]
        ''' 
        list_bboxes, list_points = self.detector.detect(list_bgr_image, threshold=threshold, polygons = polygons)
        return list_bboxes, list_points

    @staticmethod
    def get_face_align(bgr_img, bboxes, points, image_size='112,112'):
        '''
            Align face from given bounding boxes and landmark points
        '''
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        aligned = []
        for i in range(len(bboxes)):
            bbox_i = bboxes[i]
            points_i = points[i]
            nimg = face_preprocess.preprocess(rgb_img, bbox_i, points_i, image_size=image_size)
            aligned.append(nimg)
        return aligned

    def get_features(self, aligned, flip = 0):
        '''
            Extract embedding for face
            Input:
                - aligned: batch of RGB aligned images
        '''
        return self.embedd_model.get_features(aligned, flip = flip)