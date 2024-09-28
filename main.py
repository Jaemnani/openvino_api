import numpy as np # numpy
import cv2 # opencv2
from glob import glob # file list
import os # file info, copy, mkdir

# custom
from models import face_detection, landmarks_regression, face_embedding

## Default parameters
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 255, 0)
thickness = 2
    
image_paths = glob("./datas/*")

output_path = "./outputs/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# # Face Detection
face_path = "./weights/face/face-detection-adas-0001/face-detection-adas-0001.xml"  # FaceDetection [1, 1, 200(max value), 7]
# face_path = "./weights/face/face-detection-retail-0004/face-detection-retail-0004.xml"  # FaceDetection [1, 1, 200(max value), 7]
# face_path = "./weights/face/face-detection-retail-0005/face-detection-retail-0005.xml"  # FaceDetection [1, 1, 200(max value), 7]

for image_path in image_paths:
    if os.path.exists(face_path):
        print("exists:", face_path)
        
        # # Face Detection
        img = cv2.imread(image_path)
        model = face_detection(face_path)
        results = model(img) #results = model.inference(image_path)
        
        draw_origin_img = img.copy()
        for idx, det in enumerate(results):
            # Det Crop
            sp = det[2:4]
            wh = det[4:]
            
            minp = np.round(sp).astype(int)
            maxp = np.round(sp + wh).astype(int)
            
            crop_img = img[minp[1]:maxp[1], minp[0]:maxp[0]]
            
            # Debug
            sub_path = "/face_crop/"
            if not os.path.exists(output_path+sub_path):
                os.makedirs(output_path+sub_path)
            if ".jpg" in image_path:
                save_crop_name = os.path.basename(image_path).replace(".jpg", "_face_crop_%.2d.jpg"%(idx))
            elif ".png" in image_path:
                save_crop_name = os.path.basename(image_path).replace(".png", "_face_crop_%.2d.png"%(idx))
            crop_path = output_path + sub_path + save_crop_name
            cv2.imwrite(crop_path, crop_img)
            print("Save Crop image :", crop_path)
            draw_origin_img = cv2.rectangle(draw_origin_img, minp, maxp, color=color, thickness=thickness)
            draw_origin_img = cv2.putText(draw_origin_img,"%d(%d)"%(det[0], int(det[1] * 100.)) , (minp[0], minp[1]-10), fontFace=fontFace, color=color, thickness=thickness, fontScale=fontScale)
            # 
            
            # # # Landmark Regression
            landmarks_path = "./weights/face/landmarks-regression-retail-0009/landmarks-regression-retail-0009.xml" # landmarks of five points
            # landmarks_path = "./weights/face/facial-landmarks-35-adas-0002/facial-landmarks-35-adas-0002.xml"
            # landmarks_path = "./weights/face/facial-landmarks-98-detection-0001/facial-landmarks-98-detection-0001.xml"
            
            # # Landmark Regression
            if os.path.exists(landmarks_path):
                print("exists:", landmarks_path)
                model = landmarks_regression(landmarks_path)
                results = model(crop_img)
                
                draw_crop_img = crop_img.copy()
                sub_path = "/face_landmarks/"
                if not os.path.exists(output_path+sub_path):
                    os.makedirs(output_path+sub_path)
                if ".jpg" in save_crop_name:
                    save_lm_name = os.path.basename(save_crop_name).replace(".jpg", "_landmark.jpg")
                elif ".png" in save_crop_name:
                    save_lm_name = os.path.basename(save_crop_name).replace(".png", "_landmark.png")
                for landmark in results:
                    x, y = landmark
                    draw_crop_img = cv2.circle(draw_crop_img, (int(np.round(x)), int(np.round(y))), radius=0, color=(0,255,0), thickness=2)
                save_path = output_path + sub_path + save_lm_name
                print("Save Result :", save_path)
                cv2.imwrite(save_path, draw_crop_img)
                
                sub_path = "/face_aligned/"
                if not os.path.exists(output_path+sub_path):
                    os.makedirs(output_path+sub_path)
                if ".jpg" in save_crop_name:
                    save_align_name = os.path.basename(save_crop_name).replace(".jpg", "_aligned.jpg")
                elif ".png" in save_crop_name:
                    save_align_name = os.path.basename(save_crop_name).replace(".png", "_aligned.png")
                
                crop_aligned_image = model._align_rois() # align method
                
                save_path2 = output_path + sub_path + save_align_name
                print("Save Result :", save_path2)
                cv2.imwrite(save_path2, crop_aligned_image)
            else:
                exit()

            # # # Face Embedding 

            embedding_path = "./weights/face/face-reidentification-retail-0095/face-reidentification-retail-0095.xml" # embedding for REID
            # embedding_path = "./weights/face/face-recognition-resnet100-arcface-onnx/face-recognition-resnet100-arcface-onnx.xml" # embedding
            # embedding_path = "./weights/face/facenet-20180408-102900/facenet-20180408-102900.xml" # embedding

            if os.path.exists(embedding_path):
                print("exists:", embedding_path)
                model = face_embedding(embedding_path)
                results = model(crop_aligned_image)
                
                # Debug
                sub_path = "/face_embedding/"
                if not os.path.exists(output_path+sub_path):
                    os.makedirs(output_path+sub_path)
                if ".jpg" in save_crop_name:
                    save_embedding_name = os.path.basename(save_crop_name).replace(".jpg", "_embedding.txt")
                elif ".png" in save_crop_name:
                    save_embedding_name = os.path.basename(save_crop_name).replace(".png", "_embadding.txt")
                det_path = output_path+ sub_path + save_embedding_name
                print("Save Result :", det_path)
                np.savetxt(det_path, results, fmt="%f")
                #

            else:
                exit()
                
            

        # Debug
        sub_path = "/face_result/"
        if not os.path.exists(output_path+sub_path):
            os.makedirs(output_path+sub_path)
        if ".jpg" in image_path:
                save_face_result_name = os.path.basename(image_path).replace(".jpg", "_face_result.jpg")
        elif ".png" in image_path:
            save_face_result_name = os.path.basename(image_path).replace(".png", "_face_result.png")
        det_path = output_path+ sub_path + save_face_result_name
        print("Save Result :", det_path)
        cv2.imwrite(det_path, draw_origin_img)
        # 
        
    else:
        exit()
    
# # Load Embedding
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine    

embedding_paths = glob("./outputs/face_embedding/*.txt")
embedding_list = [np.loadtxt(filename) for filename in embedding_paths]

# # # matching
# cost_matrix = np.zeros((53, 53))
# for i, emba in enumerate(embedding_list):
#     for j, embb in enumerate(embedding_list):
#         if i != j:
#             cost_matrix[i,j] = cosine(emba, embb) # 1 - (np.dot(a,b) / np.sqrt( a**2 + b**2 ))
#         else:
#             cost_matrix[i,j] = np.inf
            
# row_indices, col_indices = linear_sum_assignment(cost_matrix)

# for i, j in zip(row_indices, col_indices):
#     print(f"Embedding {i} is matched with Embedding {j} with a cost of {cost_matrix[i, j]}")
            
            
# # clustering
from sklearn.cluster import DBSCAN
import shutil

embedding_paths = glob("./outputs/face_embedding/*.txt")
embedding_list = [np.loadtxt(filename) for filename in embedding_paths]

clustering_method = "DBSCAN"  # "KMeans" 로 변경할 수 있음

if clustering_method == "DBSCAN":
    # DBSCAN 클러스터링 (임계값과 최소 샘플 수는 데이터에 맞게 조정 필요)
    db = DBSCAN(eps=0.5, min_samples=1, metric='cosine').fit(embedding_list)
    labels = db.labels_
    

# 클러스터링 결과 출력
for i, label in enumerate(labels):
    print(f"Embedding file: {embedding_paths[i]} belongs to cluster {label}")

# 클러스터링 결과에 따라 파일을 다른 폴더로 정리
output_clusters_path = "./outputs/face_clusters/"
if not os.path.exists(output_clusters_path):
    os.makedirs(output_clusters_path)

for i, label in enumerate(labels):
    cluster_folder = os.path.join(output_clusters_path, f"cluster_{label}")
    if not os.path.exists(cluster_folder):
        os.makedirs(cluster_folder)
    
    # 원본 이미지 이름 추출 (임베딩 파일 이름을 기반으로 추정)
    embedding_file_name = os.path.basename(embedding_paths[i])
    image_file_name = embedding_file_name.replace("_embedding.txt", ".jpg")  # _aligned.jpg로 추정
    
    # 이미지 파일 경로
    image_file_path = os.path.join("./outputs/face_crop/", image_file_name)
    
    if os.path.exists(image_file_path):
        # 클러스터 폴더에 이미지 복사
        save_image_path = os.path.join(cluster_folder, image_file_name)
        # os.rename(image_file_path, save_image_path) # file move
        shutil.copy(image_file_path, save_image_path) # file copy
        print(f"copied {image_file_name} to {cluster_folder}")
    else:
        print(f"Image file {image_file_path} not found")




print('End')


