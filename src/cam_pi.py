
#camera class with static methods to access camera interface on Raspberry Pi

from picamera import PiCamera
import os
from time import sleep

class Camera:
    def capture_single_image(image_path = "./", image_name_prefix = "image_", image_capture_delay = 3):
        
        image_path = os.path.abspath(image_path)
        os.makedirs(image_path, exist_ok=True)
        


        # count files and folders in the directory
        image_name_suffix = len(os.listdir(image_path)) + 1
        complete_image_paths = []
        complete_image_paths.append(image_path+"/"+image_name_prefix+"{}.png".format(image_name_suffix))
        complete_image_paths = tuple(complete_image_paths)
        
        # take access to PiCamera
        camera = PiCamera()
        # show PiCamera preview
        camera.start_preview()
        # wait for image_capture_delay
        sleep(image_capture_delay)
        # take the photo and save it
        camera.capture(complete_image_paths[0])
        # close PiCamera
        camera.stop_preview()
        camera.close()

        return complete_image_paths


Camera.capture_single_image = staticmethod(Camera.capture_single_image)

if __name__ == "__main__":
    print("Single capture start")
    Camera.capture_single_image()
