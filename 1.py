import cv2
import numpy as np

def main():

    video_capture = cv2.VideoCapture('input_video.mp4')

    ad_image = cv2.imread('Advertisement_Image.png', cv2.IMREAD_UNCHANGED)

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_width = 1920
    output_height = 1080
  
    ad_image_resized = cv2.resize(ad_image, (output_width // 3, output_height // 3))

    ad_x = 0
    ad_y = 50

    frame_rate = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, frame_rate, (output_width, output_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (output_width, output_height))

        ad_bgr = ad_image_resized[..., :3]  
        ad_alpha = ad_image_resized[..., 3] / 255.0  

        roi = resized_frame[ad_y:ad_y + ad_image_resized.shape[0], ad_x:ad_x + ad_image_resized.shape[1]]

        ad_alpha = ad_alpha[:, :, None] 

        blended = (roi * (1 - ad_alpha) + ad_bgr * ad_alpha).astype(np.uint8)

        resized_frame[ad_y:ad_y + ad_image_resized.shape[0], ad_x:ad_x + ad_image_resized.shape[1]] = blended

        out.write(resized_frame)

        cv2.imshow('Resized Frame with Advertisement', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
