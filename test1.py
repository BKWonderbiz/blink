import os
import cv2
import numpy as np
import warnings
import time

from anti_spoof_predict import AntiSpoofPredict
from generate_patches import CropImage
from utility import parse_model_name
warnings.filterwarnings('ignore')

def check_image(image):
    height, width, channel = image.shape
    if width / height != 3 / 4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

def test(frame, model_dir ="D:/FaceRecogination/blink2.0/ats-fastapi-v1/resources/anti_spoof_models", device_id=0):
    # frame = cv2.resize(frame , (int(frame.shape[0]*3/4,frame.shape[0])))
    frame = cv2.resize(frame, (int(frame.shape[0] * 3 / 4), frame.shape[0]))

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    # Check if the image aspect ratio is appropriate
    result = check_image(frame)
    if result is False:
        return 0

    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))
    test_speed = 0

    # Sum the prediction from each model in the directory
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    # Determine if the face is real or fake
    label = np.argmax(prediction)
    print(prediction)
    print(label)
    value = prediction[0][label] / 2
    print(value)
    if label == 1:
        print(f"Frame is Real Face. Score: {value:.2f}.")
        return 1  # Real face
    else:
        print(f"Frame is Fake Face. Score: {value:.2f}.")
        return 0  # Fake face


# import os
# import cv2
# import numpy as np
# import warnings
# import time
# from collections import deque
# from anti_spoof_predict import AntiSpoofPredict
# from generate_patches import CropImage
# from utility import parse_model_name
# warnings.filterwarnings('ignore')
#   # Assuming these are defined elsewhere


# def check_image(image):
#     height, width, channel = image.shape
#     if width / height != 3 / 4:
#         print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
#         return False
#     else:
#         return True



# # Initialize a deque with a fixed size of 5
# last_five_results = deque(maxlen=5)

# def test(frame, model_dir="D:/FaceRecogination/fastapi-face-recognition/resources/anti_spoof_models", device_id=0):
#     global last_five_results

#     # Resize the frame
#     frame = cv2.resize(frame, (int(frame.shape[0] * 3 / 4), frame.shape[0]))

#     model_test = AntiSpoofPredict(device_id)
#     image_cropper = CropImage()

#     # Check if the image aspect ratio is appropriate
#     result = check_image(frame)
#     if result is False:
#         return 0

#     image_bbox = model_test.get_bbox(frame)
#     prediction = np.zeros((1, 3))
#     test_speed = 0

#     # Sum the prediction from each model in the directory
#     for model_name in os.listdir(model_dir):
#         h_input, w_input, model_type, scale = parse_model_name(model_name)
#         param = {
#             "org_img": frame,
#             "bbox": image_bbox,
#             "scale": scale,
#             "out_w": w_input,
#             "out_h": h_input,
#             "crop": True,
#         }
#         if scale is None:
#             param["crop"] = False
#         img = image_cropper.crop(**param)
#         start = time.time()
#         prediction += model_test.predict(img, os.path.join(model_dir, model_name))
#         test_speed += time.time() - start

#     # Determine if the face is real or fake
#     label = np.argmax(prediction)
#     value = prediction[0][label] / 2
#     if label == 1:
#         print(f"Frame is Real Face. Score: {value:.2f}.")
#         last_five_results.append(1)  # Real face detected
#     else:
#         print(f"Frame is Fake Face. Score: {value:.2f}.")
#         last_five_results.append(0)  # Fake face detected

#     # Check if the last five results were all fake
#     if last_five_results.count(0) == len(last_five_results):
#         print("Fail-safe triggered: Last five frames were fake.")
#         return 0

#     # If a real face was detected in the latest frame, return true
#     if last_five_results[-1] == 1:
#         return 1
#     else:
#         return 0
