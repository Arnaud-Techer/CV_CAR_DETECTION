import os
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import yaml
import ultralytics
from ultralytics import YOLO


# definition of labels
LABELS = {0:'car',
          1:'truck',
          2:'person',
          3:'bicycle',
          4: 'traffic_light'}


def convert_to_yolo(img_width:int, img_height:int, xmin:int, xmax:int, ymin:int, ymax:int) -> tuple[int]:
    """Convert the format of the images into a format requiered for the training of the YOLO model.

    Args:
        img_width (int): width of the image
        img_height (int): height of the image
        xmin (int): smalest value of x in the image
        xmax (int): highest value of x in the image 
        ymin (int): smalest value of y in the image
        ymax (int): highest value of y in the image

    Returns:
        tuple[int]: tuple of 4 values (x position of the center point, y position of the center point, width, height)
    """
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

# Function to save YOLO format annotation files
def save_yolo_annotations(df:pd.DataFrame, subset_name:str,images_folder:str,output_folder:str) -> None:
    """Save all the YOLO format images in a good format for the training. Use the .csv file to separate train data and test data
    and associate each image with all the labels inside.

    Args:
        df (pd.DataFrame): for each images show the labels contains in it.
        subset_name (str): define if the image is for training or testing the model.
        images_folder (str): name of the folder that contain the images.
        output_folder (str): name of the folder that contain the images for training the model.
        flag_train_val (str): if True this flag indicates that the annotations and images are used for training. Otherwise the images are for validation.
    """
    subset_folder = os.path.join(output_folder, subset_name)
    labels_subset_folder = os.path.join(subset_folder, 'labels')
    
    if not os.path.exists(labels_subset_folder):
        os.mkdir(labels_subset_folder)
        os.mkdir(f'{labels_subset_folder}/{subset_name}')
    
    if not os.path.exists(subset_folder):
        os.mkdir(subset_folder)
    
    for _, row in df.iterrows():
        annotation_frame = os.path.splitext(row['frame'])[0] + '.txt'
        annotation_path = f'{labels_subset_folder}/{subset_name}/{annotation_frame}'
        if os.path.exists(annotation_path):
            # skip creating this annotation if it already exists
            continue  
        
        # Read the image to get dimensions
        image_path = os.path.join(images_folder, row['frame'])
        image = cv2.imread(image_path)
        # if the image does not exist we skip it
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        
        # convert bounding box to YOLO format
        xmin, xmax, ymin, ymax, class_id = row['xmin'], row['xmax'], row['ymin'], row['ymax'], row['class_id']
        x_center, y_center, width, height = convert_to_yolo(img_width, img_height, xmin, xmax, ymin, ymax)
        
        # save image in the subset folder
        new_image_path = f'{subset_folder}/{row['frame']}'
            
        if not os.path.exists(new_image_path):
            cv2.imwrite(new_image_path, image)
        
        # write annotation in YOLO format
        with open(annotation_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def create_yolo_config(file_path:str='./data.yaml') -> None:
    """Create a config file to train the YOLO model on new data. This file is a YAML file.

    Args:
        file_path (str, optional): Name of the .yalm file containing the config to train YOLO model. Defaults to './data.yaml'.
    """
    yolo_format = {
        'path': "./yolo_dataset",
        'train': "./yolo_dataset/train",
        'val': "./yolo_dataset/val",
        'nc': 5,
        'names': {0:"car",1:'truck',2:'person',3:'bicycle',4:'traffic_light'}
    }
    
    with open(file_path, 'w') as outfile:
        yaml.dump(yolo_format, outfile, default_flow_style=False)
    print(f"YOLO config saved at {file_path}")
    return file_path

def train_yolo_model(data_yaml:str, model_path:str='yolov8m.pt', epochs:int=6, imgsz:int=640) -> ultralytics.models.yolo.model.YOLO:
    """Train a YOLO model on the images and labels in the training dataset.

    Args:
        data_yaml (str): path to the configuration file YAML to train the model
        model_path (str, optional): path to the YOLO model to be used. Defaults to 'yolov8m.pt'.
        epochs (int, optional): number of epochs to tain the model. Defaults to 50.
        imgsz (int, optional): input size of the image. Defaults to 640.

    Returns:
        ultralytics.models.yolo.model.YOLO: YOLO model after training
    """
    model = YOLO(model_path)
    model.train(
        data=data_yaml, 
        epochs=epochs, 
        imgsz=imgsz,
        workers=4  
    )
    model.save(filename='yolo_trained.pt')
    return model 

def test_yolo_model(model:ultralytics.models.yolo.model.YOLO, data_yaml:str, test_images_folder:str=None):
    """Test the model trained on the train dataset.

    Args:
        model (ultralytics.models.yolo.model.YOLO): model trained on the dataset.
        data_yaml (str): configuration file name for the training of a YOLO model.
        test_images_folder (str, optional): folder name that contains the test images. Defaults to None.

    Returns:
        _type_: prediction made on the test dataset with the trained YOLO model.
    """
    results = model.val(data=data_yaml)
    if test_images_folder:
        predictions = model.predict(source=test_images_folder, save=True)
        return predictions
    
def main():
    images_folder = './data/images/images'
    labels_train_csv = './data/labels_train.csv'
    output_folder = './yolo_dataset/yolo_dataset'
    # load the data in csv
    labels_train = pd.read_csv(labels_train_csv)
    train_labels, test_labels = train_test_split(labels_train,test_size=0.2,shuffle=False)

    # Save train and test annotations   
    save_yolo_annotations(train_labels, 'train',images_folder,output_folder)
    save_yolo_annotations(test_labels, 'val',images_folder,output_folder)
    # Path to dataset configuration file
    data_yaml = create_yolo_config()
    # data_yaml = './data.yaml'

    # Train the YOLO model
    trained_model = train_yolo_model(data_yaml, epochs=7, imgsz=640)
    # Test the model on validation set and optionally on test images
    test_yolo_model(trained_model, data_yaml, test_images_folder=f'{output_folder}/val/images/')

if __name__ == '__main__':
    main()