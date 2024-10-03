import pandas as pd
import cv2 
import matplotlib.pyplot as plt
from PIL import Image
import os 
from ultralytics import YOLO

def display_image(image_path:str) -> None:
    """Display an Image using the path of the image.

    Args:
        image_path (str): path and name to the image we want to display.
    """
    with Image.open(image_path) as img:
        img.show()

def show_first_image(labels_train:pd.DataFrame) -> None:
    """Show the first image from the label train csv file.

    Args:
        labels_train (pd.DataFrame): DataFrame created by reading the label_train.csv file.
    """
    first_image_name = labels_train.loc[0, "frame"]
    image_path = os.path.join("data/images/images", first_image_name)
    display_image(image_path)
    return None

def process_image(image_path:str, resize_to:tuple[int]=None) -> Image:
    """Process a given image to convert it into a good format for gif creation.
    Resize the image if necessary.

    Args:
        image_path (str): path and name of the image to process
        resize_to (tuple[int], optional): Tuple of two integers if we want to resize the image to this values. Defaults to None.

    Returns:
        Image: Process image in terms of resize and format.
    """
    try:
        with Image.open(image_path) as img:
            if resize_to:
                img = img.resize(resize_to, Image.Resampling.LANCZOS)
            return img.convert('P')
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def create_gif_from_images(image_folder:str, output_gif:str='output.gif', duration:int=100,subset_images:int=1) -> None :
    """Create a Gif using the images in a given folder.

    Args:
        image_folder (str): path to the folder containing all the input data.
        output_gif (str, optional): name of the gif created by the function. Defaults to 'output.gif'.
        duration (int, optional): duration of the gif. Defaults to 100.
        subset_images (int, optional): step of images choose to create the gif. If fix to one, every image of the folder is used. Defaults to 1.
    """
    if not os.path.exists(image_folder):
        print(f"Folder {image_folder} does not exist. This folder is supposed to contain images.")
        return None

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')][::subset_images])

    if not image_files:
        print(f"No images found to create GIF in {image_folder}.")
        return None

    images = []
    for image_file in image_files:
        img_path = os.path.join(image_folder, image_file)
        img = Image.open(img_path)
        images.append(img)

    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=0)
    print(f"GIF created and saved as {output_gif}")
    return None


def run_yolo_inference(image_folder : str, output_folder :str , model_path :str ='yolov8m.pt', limit:int =50) -> str:
    """Run the inference of the YOLO model on the input images to detect important objects on images (car, traffic light, persons,...)

    Args:
        image_folder (str): path to the folder containing the input images.
        output_folder (str): name of the folder that will contain the output images of the model.
        model_path (str, optional): name of the model we want to use. Defaults to 'yolov8m.pt'.
        limit (int, optional): limit the number of images on which the inference is done. Defaults to 50.

    Returns:
        str: folder path to the output images.
    """
    model = YOLO(model_path)
    
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')])[:limit]
    model.predict(source=image_files, save=True, project=output_folder, name='inference_output')

    output_path = os.path.join(output_folder, 'inference_output')
    if not os.path.exists(output_path):
        print(f"Error: The expected output directory '{output_path}' was not created.")
    else:
        print(f"Inference complete. Results saved to {output_path}")
    return output_path 

def main():
    """Run the inference of the YOLO model on the input images.
    """
    image_folder = './data/images/images' 
    output_folder = './results'
    
    create_gif_from_images(image_folder=image_folder,output_gif='animation.gif',duration=40,subset_images=10)
    
    inference_output_folder = run_yolo_inference(image_folder, output_folder, limit=50)
    create_gif_from_images(inference_output_folder, output_gif="yolo_inference.gif", duration=100)

if __name__ == "__main__":
    main()