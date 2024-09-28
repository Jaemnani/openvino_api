import numpy as np
import cv2
from .openvino_model import OPENVINO_MODEL
from openvino.runtime.utils.data_helpers.wrappers import OVDict
import os

# adas_0001/face-detection-adas-0001
class face_detection(OPENVINO_MODEL):
    def __init__(self, model_xml, device="CPU"):
        super().__init__(model_xml=model_xml, device=device)
        
        self.confidence_threshold=0.5
        self.roi_scale_factor=1.15
        
        if self.input_count == 1:
            self.input_shape = self.input_shape[0]
            self.input_name = self.input_name[0]
            self.input_type = self.input_type[0]
        if self.output_count == 1:
            self.output_shape = self.output_shape[0]
            self.output_name = self.output_name[0]
            self.output_type = self.output_type[0]
        print("check")
        
    def inference(self, input_image):
        batch_img = self.preprocessing(input_image)
        pred = self.infer_request.infer(inputs={self.input_name: batch_img})
        if pred.__class__ == OVDict:
            pred = pred[0]
        results = self.postprocessing(pred)
        self.last_output = results
        return results
    
    def preprocessing(self, input_image):
        self.input_image = input_image
        self.input_size = input_image.shape
        if self.input_shape[-1] <= 3:
            self.target_size = self.input_shape[::-1][1:3]
            batch_img = np.array([cv2.resize(input_image, self.target_size)])
        else:
            self.target_size = self.input_shape[::-1][:2]
            resized_img = cv2.resize(input_image, self.target_size)
            batch_img = np.array([resized_img.transpose(2,0,1)])
        return batch_img
    
    def postprocessing(self, pred):
        outputs = pred
        
        results = []
        for output in outputs[0][0]:
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.position = np.array((output[3], output[4])) # (x, y)
            self.size = np.array((output[5], output[6])) # (w, h)

            # result = FaceDetector.Result(output)
            if self.confidence < self.confidence_threshold:
                break # results are sorted by confidence decrease

            self.resize_roi(self.input_size[1], self.input_size[0])
            self.rescale_roi(self.roi_scale_factor)
            self.clip(self.input_size[1], self.input_size[0])
            result = np.array([self.label, self.confidence, self.position[0], self.position[1], self.size[0], self.size[1]])
            results.append(result)

        return np.array(results)
    
    def rescale_roi(self, roi_scale_factor=1.0):
            self.position -= self.size * 0.5 * (roi_scale_factor - 1.0)
            self.size *= roi_scale_factor

    def resize_roi(self, frame_width, frame_height):
        self.position[0] *= frame_width
        self.position[1] *= frame_height
        self.size[0] = self.size[0] * frame_width - self.position[0]
        self.size[1] = self.size[1] * frame_height - self.position[1]

    def clip(self, width, height):
        min = [0, 0]
        max = [width, height]
        self.position[:] = np.clip(self.position, min, max)
        self.size[:] = np.clip(self.size, min, max)
        
class landmarks_regression(OPENVINO_MODEL):
    REFERENCE_LANDMARKS = [
        (30.2946 / 96, 51.6963 / 112), # left eye
        (65.5318 / 96, 51.5014 / 112), # right eye
        (48.0252 / 96, 71.7366 / 112), # nose tip
        (33.5493 / 96, 92.3655 / 112), # left lip corner
        (62.7299 / 96, 92.2041 / 112)] # right lip corner

    UNKNOWN_ID = -1
    UNKNOWN_ID_LABEL = "Unknown"
    
    def __init__(self, model_xml, device="CPU"):
        super().__init__(model_xml=model_xml, device=device)  
        
        if self.input_count == 1:
            self.input_shape = self.input_shape[0]
            self.input_name = self.input_name[0]
            self.input_type = self.input_type[0]
        if self.output_count == 1:
            self.output_shape = self.output_shape[0]
            self.output_name = self.output_name[0]
            self.output_type = self.output_type[0]
            
        self.aligned_image = None
            
    def preprocessing(self, input_image):
        self.input_image = input_image
        self.input_size = self.input_image.shape[:2][::-1]
        if self.input_shape[-1] <= 3:
            self.target_size = self.input_shape[::-1][1:3]
            batch_img = np.array([cv2.resize(input_image, self.target_size)])
        else:
            self.target_size = self.input_shape[::-1][:2]
            resized_img = cv2.resize(input_image, self.target_size)
            batch_img = np.array([resized_img.transpose(2,0,1)])
        return batch_img
    
    def inference(self, input_image):
        batch_img = self.preprocessing(input_image)
        pred = self.infer_request.infer(inputs={self.input_name: batch_img})
        if pred.__class__ == OVDict:
            pred = pred[0]
        result = self.postprocessing(pred)
        return result
    
    def postprocessing(self, pred):
        if pred.shape == (1, 98, 16, 16): # 98
            heatmaps = pred[0]
            
            w = float(self.input_size[0])
            h = float(self.input_size[1])
            center = (w/2., h/2.)
            scale = (w, h)
                        
            
            # getMaxPreds
            hm_shape = heatmaps[0].shape
            heatMapDatas = heatmaps.reshape(-1, hm_shape[0] * hm_shape[1])
            idxs = np.argmax(heatMapDatas, axis=1)
            maxVals = np.array([hm[idxs[i]] for i, hm in enumerate(heatMapDatas)])
            # maxVals = np.take_along_axis(heatMapDatas, idxs[:, np.newaxis], axis=1).flatten()
            preds = np.array([[float(idx % hm_shape[1]), float(idx // hm_shape[1])] if maxVals[i] > 0 else [-1, -1] for i, idx in enumerate(idxs) ])
            
            # 최대값 근처 보정
            for idx, heatmap in enumerate(heatmaps):
                px, py = int(preds[idx][0]), int(preds[idx][1])
                
                if 1 < px < heatmap.shape[1]-1 and 1<py<heatmap.shape[0]-1:
                    diffFirst = heatmap[py, px+1]-heatmap[py,px-1]
                    diffSecond = heatmap[py+1,px]-heatmap[py-1,px]
                    preds[idx][0] += np.sign(diffFirst) * 0.25
                    preds[idx][1] += np.sign(diffSecond) * 0.25

            # preds 변환
            trans = self.affine_transform(center, scale, 0, hm_shape[0], hm_shape[1], np.zeros(2), True) 
            
            landmarks = []
            for idx, pred in enumerate(preds):
                coord = np.array([ [pred[0]], [pred[1]], [1] ], dtype=np.float32)
                trans = trans.astype(np.float32)
                point = np.dot(trans, coord)
                x = np.uint32(point[0][0])
                y = np.uint32(point[1][0])
                landmarks.append((x,y))
            return np.array(landmarks)
                
                
            # for landmarkId in range(numberLandmarks):
            # coord = np.array([[preds[landmarkId][0]], [preds[landmarkId][1]], [1]], dtype=np.float32)
            # trans = trans.astype(np.float32)
            # point = np.dot(trans, coord)
            # x = int(point[0, 0])
            # y = int(point[1, 0])
            # landmarks.append((x, y))
            
            # ori_size = np.array(self.input_image.shape[:2])
            # heat_size = np.array(heatmaps.shape[-2:])
            # scale = ori_size / heat_size
            # coords = []
            # for heatmap in heatmaps:
            #     scaled_heatmap = cv2.resize(heatmap, tuple(self.input_size), interpolation=cv2.INTER_CUBIC)
            #     argmax = np.argwhere(scaled_heatmap==scaled_heatmap.max()).astype(float)
            #     if argmax.ndim > 1:
            #         argmax = np.median(argmax, axis=0)
            #     coords.append(argmax.flatten()[::-1])
            # outputs = np.array(coords).astype(int)  
            print('Check Landmark 98 postprocessing')
            
        else:
            outputs = pred.reshape(-1, 2)
            self.last_output = outputs
            img_size = np.array(self.input_size).astype(float)
            scaled_outputs = outputs * img_size
            
            # self._align_rois(self.input_image, outputs)
            
            
            
        return scaled_outputs
    
    def _align_rois(self, image=None, image_landmarks=None):
        if image== None:
            image= self.input_image
        if image_landmarks == None:
            image_landmarks = self.last_output
        scale = np.array((image.shape[1], image.shape[0]))
        desired_landmarks = np.array(self.REFERENCE_LANDMARKS, dtype=float) * scale
        landmarks = image_landmarks * scale

        transform = self.get_transform(desired_landmarks, landmarks)
        dst_img = image.copy()
        src_img = image.copy()
        cv2.warpAffine(src_img, transform, tuple(scale), dst_img, flags=cv2.WARP_INVERSE_MAP)
        return dst_img
    
    def rotate_point(self, pt, rot_rad):
        """회전 변환"""
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        return np.array([pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs])

    def get_3rd_point(self, a, b):
        """세 번째 점 계산"""
        direct = a - b
        return b + np.array([-direct[1], direct[0]])

    def affine_transform(self, center, scale, rot, dst_w, dst_h, shift=np.array([0., 0.]), inv=False):
        """어파인 변환 행렬 계산"""
        pi = np.pi
        rot_rad = pi * rot / 180
        scale_tmp = scale

        # Source points 계산
        src_dir = self.rotate_point(np.array([0., scale_tmp[1] * -0.5]), rot_rad)
        src = np.zeros((3, 2), dtype=np.float32)
        src[0] = np.array([center[0] + scale_tmp[0] * shift[0], center[1] + scale_tmp[1] * shift[1]])
        src[1] = np.array([center[0] + src_dir[0] + scale_tmp[0] * shift[0], center[1] + src_dir[1] + scale_tmp[1] * shift[1]])
        src[2] = self.get_3rd_point(src[0], src[1])

        # Destination points 계산
        dst_dir = np.array([0., dst_w * -0.5])
        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0] = np.array([dst_w * 0.5, dst_h * 0.5])
        dst[1] = dst[0] + dst_dir
        dst[2] = self.get_3rd_point(dst[0], dst[1])

        # Affine 변환 행렬 계산
        if inv:
            trans = cv2.getAffineTransform(dst, src)
        else:
            trans = cv2.getAffineTransform(src, dst)

        return trans

    # 5 points
    def get_aligned_image(self):
        if isinstance(self.last_output, np.ndarray):
            desired_landmarks = np.array(self.REFERENCE_LANDMARKS, dtype=float) * self.input_size
            # desired_landmarks = np.round(desired_landmarks)
            landmarks = self.last_output * self.input_size
            # landmarks = np.round(landmarks)
            
            save_img = self.input_image.copy()
            for landmark in landmarks:
                x, y = landmark
                save_img = cv2.circle(save_img, (int(np.round(x)), int(np.round(y))), radius=0, color=(0,255,0), thickness=2)
            for desired_landmark in desired_landmarks:
                x, y = desired_landmark
                save_img = cv2.circle(save_img, (int(np.round(x)), int(np.round(y))), radius=0, color=(255, 0, 255), thickness=2)
            
            cv2.imwrite(self.image_path.replace(".jpg", "_landmarks.jpg"), save_img)
            
            # warpafine 하는 이유는 얼굴 이미지 정렬해서 embedding을 잘 먹일 수 있도록 하기 위함.
            aligned_img = self.input_image.copy()
            transform = self.get_transform(desired_landmarks, landmarks)
            aligned_result = cv2.warpAffine(self.input_image.copy(), transform, tuple(self.input_size), aligned_img, flags=cv2.WARP_INVERSE_MAP)
            cv2.imwrite(self.image_path.replace(".jpg", "_align.jpg"), aligned_img)
            
            return aligned_img
        else:
            print("Inference first.")
            
    # 5 points
    def get_landmarks_pointed_image(self):
        if self.last_output != None:
            desired_landmarks = np.array(self.REFERENCE_LANDMARKS, dtype=float) * self.input_size
            # desired_landmarks = np.round(desired_landmarks)
            landmarks = self.last_output * self.input_size
            # landmarks = np.round(landmarks)
            
            save_img = self.input_image.copy()
            for landmark in landmarks:
                x, y = landmark
                save_img = cv2.circle(save_img, (int(np.round(x)), int(np.round(y))), radius=0, color=(0,255,0), thickness=2)
            for desired_landmark in desired_landmarks:
                x, y = desired_landmark
                save_img = cv2.circle(save_img, (int(np.round(x)), int(np.round(y))), radius=0, color=(255, 0, 255), thickness=2)
            
            # cv2.imwrite(image_path.replace(".jpg", "_landmarks.jpg"), save_img)
            
            return save_img
        else:
            print("Inference first.")
        
    def normalize(self, array, axis):
        mean = array.mean(axis=axis)
        array -= mean
        std = array.std()
        array /= std
        return mean, std
    
    def get_transform(self, src, dst):
        assert np.array_equal(src.shape, dst.shape) and len(src.shape) == 2, \
            '2d input arrays are expected, got {}'.format(src.shape)
        src_col_mean, src_col_std = self.normalize(src, axis=0)
        dst_col_mean, dst_col_std = self.normalize(dst, axis=0)

        u, _, vt = np.linalg.svd(np.matmul(src.T, dst))
        r = np.matmul(u, vt).T

        transform = np.empty((2, 3))
        transform[:, 0:2] = r * (dst_col_std / src_col_std)
        transform[:, 2] = dst_col_mean.T - np.matmul(transform[:, 0:2], src_col_mean.T)
        return transform    

class face_embedding(OPENVINO_MODEL):
    def __init__(self, model_xml, device="CPU"):
        super().__init__(model_xml=model_xml, device=device)
        
        match_threshold=0.5
        match_algo = "HUNGARIAN"
        
        if self.input_count == 1:
            self.input_shape = self.input_shape[0]
            self.input_name = self.input_name[0]
            self.input_type = self.input_type[0]
        if self.output_count == 1:
            self.output_shape = self.output_shape[0]
            self.output_name = self.output_name[0]
            self.output_type = self.output_type[0]
            
    def inference(self, input_image):
        batch_img = self.preprocessing(input_image)
        pred = self.infer_request.infer(inputs={self.input_name: batch_img})
        if pred.__class__ == OVDict:
            pred = pred[0]
        results = self.postprocessing(pred)
        self.last_output = results
        return results
    
    def preprocessing(self, input_image):
        self.input_image = input_image
        self.input_size = input_image.shape
        if self.input_shape[-1] <= 3:
            self.target_size = self.input_shape[::-1][1:3]
            batch_img = np.array([cv2.resize(input_image, self.target_size)])
        else:
            self.target_size = self.input_shape[::-1][:2]
            resized_img = cv2.resize(input_image, self.target_size)
            batch_img = np.array([resized_img.transpose(2,0,1)])
        return batch_img
    
    def postprocessing(self, pred):
        outputs = pred.flatten()
        return outputs
