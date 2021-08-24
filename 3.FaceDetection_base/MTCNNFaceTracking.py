import dlib
from facenet_pytorch import MTCNN
import torch
from deepface import DeepFace
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import os
import threading
import time


def check_tracker_quality(face_trackers, base_image, quality_thresh_hold=7):
    # Update all the trackers and remove the ones for which the update
    # indicated the quality was not good enough
    fids_to_delete = []
    for fid in face_trackers.keys():
        trackingQuality = face_trackers[fid].update(base_image)

        # If the tracking quality is good enough, we must delete
        # this tracker
        if trackingQuality < quality_thresh_hold:
            fids_to_delete.append(fid)

    for fid in fids_to_delete:
        print("Removing tracker " + str(fid) + " from list of trackers")
        face_trackers.pop(fid, None)
    return face_trackers


def create_new_tracker(current_face_id, frame, loc):
    print("Creating new tracker " + str(current_face_id))

    x, y, w, h = [int(i) for i in loc]
    # Create and store the tracker
    tracker = dlib.correlation_tracker()
    tracker.start_track(frame,
                        dlib.rectangle(x - int(w / 4),
                                       y - int(h / 4),
                                       x + w + int(w / 4),
                                       y + h + int(h / 4)))
    return tracker


# We are not doing really face recognition
def do_recognize_person(face_names, fid, face_img):
    # time.sleep(2)
    # data = cv2.imread('A:\\PYTHON\\FinalBachelorProject\\FaceDataBase\\Biden\\Biden1.jpg')
    # all_faces = RetinaFace.detect_faces(data)
    # print(all_faces)
    verified_identity = None
    try:
        face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        df = DeepFace.find(face_img_bgr,
                           db_path="A:/PYTHON/FinalBachelorProject/FaceDataBase",
                           enforce_detection=False,
                           detector_backend='retinaface',
                           # detector_backend='mtcnn',
                           model_name='ArcFace',
                           model=model,
                           distance_metric="euclidean_l2")
        print(df)
        print()
        # filtered_faced = df[df['ArcFace_euclidean_l2'] <= 1]
        closest_identity = df.loc[df['ArcFace_euclidean_l2'].idxmin()]
        print(closest_identity['identity'])
        verified_identity = closest_identity['identity'].split('\\')[-1].split('/')[0]
        print()
        print(verified_identity)
    except (ValueError, cv2.error):
        print('No Face Detected')
        # all_faces = RetinaFace.detect_faces(data)
        # print(all_faces)
        # cv2.imshow('face', face_img)
        # cv2.waitKey()

    num_of_zeros = 3 - len(str(fid))
    if verified_identity is None:
        face_names[fid] = {
            'name': '{person_id}'.format(person_id=num_of_zeros * '0' + str(fid)),
            'detected': False
        }
    else:
        face_names[fid] = {
            'name': verified_identity,
            'dir': '{person_id}'.format(person_id=verified_identity + '\\' + num_of_zeros * '0' + str(fid)),
            'detected': True
        }


def check_existence_of_new_faces(result_list, face_trackers,
                                 face_names, frame, current_face_id):
    verified_trackers = []
    for result in result_list:
        x, y, x1, y1 = result
        w, h = abs(x1 - x), abs(y1 - y)

        # Calculate the center point
        x_bar = (x + x1) / 2
        y_bar = (y + y1) / 2

        # Variable holding information which faceid we
        # matched with
        matched_fid = None

        # Now loop over all the trackers and check if the
        # center point of the face is within the box of a
        # tracker
        for fid in face_trackers.keys():
            tracked_position = face_trackers[fid].get_position()
            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())

            # calculate the center point
            t_x_bar = t_x + 0.5 * t_w
            t_y_bar = t_y + 0.5 * t_h

            # check if the center point of the face is within the
            # rectangle of a tracker region. Also, the center point
            # of the tracker region must be within the region
            # detected as a face. If both of these conditions hold
            # we have a match
            if ((t_x <= x_bar <= (t_x + t_w)) and
                    (t_y <= y_bar <= (t_y + t_h)) and
                    (x <= t_x_bar <= (x + w)) and
                    (y <= t_y_bar <= (y + h))):
                matched_fid = fid
                verified_trackers.append(fid)

        if matched_fid is None:
            loc = [x, y, w, h]
            tracker = create_new_tracker(current_face_id, frame, loc)

            face_trackers[current_face_id] = tracker
            verified_trackers.append(current_face_id)
            # Start a new thread that is used to simulate
            # face recognition. This is not yet implemented in this
            # version :)
            # t = threading.Thread(target=do_recognize_person,
            #                      args=(face_names, current_face_id))
            # t.start()
            # Increase the currentFaceID counter
            current_face_id += 1

    # Remove the trackers that are not in the second detection
    for fid in face_trackers.copy().keys():
        if fid not in verified_trackers:
            print("Removing tracker " + str(fid) +
                  " from list of trackers because of not appearing in the second detection")
            face_trackers.pop(fid, None)

    return current_face_id, face_trackers


# draw an image with detected objects and save the faces
def plot_image_with_boxes(frame, face_trackers, face_name, num_of_digits,
                          frame_number, output_file_name, face_shapes=None, save_image=True):
    face_img = frame.copy()
    for fid in face_trackers.keys():
        tracked_position = face_trackers[fid].get_position()
        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())

        # face = frame[t_y:t_y + t_h, t_x:t_x + t_w]
        face = face_img[t_y:t_y + t_h, t_x:t_x + t_w]
        if face_shapes is not None:
            face = cv2.resize(face, face_shapes, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        if fid in face_name.keys():
            box_name = face_name[fid]['name']

        else:
            box_name = 'Detecting...'
            do_recognize_person(face_name, fid, face)

        if save_image:
            if face_name[fid]['detected']:
                directory = '\\'.join(output_file_name) + '\\labeled\\' + face_name[fid]['dir']
                id_person = face_name[fid]['dir'].split('\\')[1]
            else:
                directory = '\\'.join(output_file_name) + '\\unlabeled\\' + face_name[fid]['name']
                id_person = face_name[fid]['name']
            # directory = output_file_name + '\\unlabeled\\' + face_name[fid]
            if not os.path.exists(directory):
                os.makedirs(directory)

            num_of_zeros = num_of_digits - len(str(frame_number))
            img_name = directory + '\\{source}_{n_video}_{frameNumber}_{id_p}_{x}_{y}_{h}_{w}.jpg'.format(
                source=output_file_name[-2], n_video=output_file_name[-1]
                , frameNumber=num_of_zeros * '0' + str(frame_number)
                , id_p=id_person, x=t_x, y=t_y, h=t_h, w=t_w)

            try:
                cv2.imwrite(img_name, face)
            except cv2.error:
                pass

        cv2.rectangle(frame, (t_x, t_y),
                      (t_x + t_w, t_y + t_h),
                      (0, 0, 255), 2)
        cv2.putText(frame, box_name, (int(t_x + t_w / 2), int(t_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (204, 204, 0), 2)


def face_tracking(path, output_file_name, detection_quality_tsh=7, save_image=True, show_result=True,
                  face_detection_tsh=10, resize_faces_shape=None, desired_time=None):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    if show_result:
        cv2.namedWindow("Face-Tracking", cv2.WINDOW_AUTOSIZE)  ###make window for image or video

        # Start the window thread for the two windows we are using
        cv2.startWindowThread()

    # Get frame/second of the video
    frames_per_second = int(vidObj.get(cv2.CAP_PROP_FPS))
    print('frames/second:', frames_per_second)
    if desired_time is None:
        frame_threshold = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        frame_threshold = int(desired_time * 60 * frames_per_second)
    number_of_all_frames_digits = len(str(frame_threshold))

    # Used as counter variable
    frame_counter = 0
    current_face_id = 1

    # Variables holding the correlation trackers and the name per faceid
    face_trackers = {}
    face_names = {}

    while vidObj.isOpened() and frame_counter < frame_threshold:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        if success:
            # resize_frame = cv2.resize(image, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            resize_frame = image
            face_trackers = check_tracker_quality(face_trackers, resize_frame,
                                                  quality_thresh_hold=detection_quality_tsh)

            if (frame_counter % face_detection_tsh) == 0:
                all_faces, _ = mtcnn.detect(resize_frame)
                if all_faces is not None:
                    current_face_id, face_trackers = check_existence_of_new_faces(all_faces, face_trackers,
                                                                                  face_names,
                                                                                  resize_frame, current_face_id)

            frame_counter += 1
            plot_image_with_boxes(resize_frame, face_trackers, face_names, number_of_all_frames_digits,
                                  frame_counter, output_file_name, face_shapes=resize_faces_shape, save_image=save_image)
            if show_result:
                cv2.imshow('Face-Tracking', resize_frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        elif (not success) or (frame_counter == frame_threshold):
            vidObj.release()

    if show_result:
        # Destroy any OpenCV windows and exit the application
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print('dlib version:', dlib.__version__)
    # print(dlib.cuda.get_device())
    # print('dlib using cuda:', dlib.DLIB_USE_CUDA)
    model = DeepFace.build_model('ArcFace')

    addr = []
    all_paths = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(keep_all=True, device=device)
    base_addr = 'A:\\PYTHON\\FinalBachelorProject\\Videos'

    # num_of_sources = len(os.listdir(base_addr))
    # source_digits = len(str(num_of_sources))
    source_digits = 2

    selected_channels = ['1', '2']
    selected_videos = [1]

    for root, dirs, files in os.walk(base_addr):
        num_of_videos = len(files)
        # file_digits = len(str(num_of_videos))
        file_digits = 3
        root_list = root.split('\\')
        if root_list[-1] in selected_channels:
            filtered_files = [f for f in files if int(f.split('.mp4')[0]) in selected_videos]
        # i = 1
            for file in filtered_files:
                all_paths.append(os.path.join(root, file))
                root_list[-2] = 'MTCNNFrames'
                source_number = root_list[-1]
                file_number = file.split('.mp4')[0]
                root_list[-1] = (source_digits - len(source_number)) * '0' + source_number
                new_root = root_list.copy()
                new_root.append((file_digits - len(file_number)) * '0' + file_number)
                addr.append(new_root)
            # i += 1
    for i in range(len(addr)):
        start_time = time.time()
        p = all_paths[i]
        address = addr[i]
        face_tracking(
            path=p,
            output_file_name=address,
            detection_quality_tsh=7,
            face_detection_tsh=20,
            # resize_faces_shape=(300, 400),
            resize_faces_shape=None,
            save_image=True,
            show_result=False,
            # desired_time=1
        )

        elapsed_time = time.time() - start_time
        time_report = f'Finished in {elapsed_time} seconds, or {elapsed_time / 60} minutes'
        print(time_report)
        file = open(r"{folder}\time.txt".format(folder='\\'.join(address)), "w+")
        file.write(time_report)
        file.close()
