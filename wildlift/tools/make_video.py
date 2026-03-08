import cv2
import os
import glob
from pathlib import Path

def images_to_video(input_folder, output_path, fps=30, img_format='jpg'):
    """
    Convert images in a folder to a video file
    
    Args:
        input_folder (str): Path to folder containing images
        output_path (str): Path for output video file
        fps (int): Frames per second for the video
        img_format (str): Image file extension (jpg, png, etc.)
    """
    
    # Get all image files and sort them
    img_pattern = os.path.join(input_folder, f"*.{img_format}")
    img_files = sorted(glob.glob(img_pattern))
    
    if not img_files:
        print(f"No {img_format} files found in {input_folder}")
        return
    
    # Read first image to get dimensions
    first_img = cv2.imread(img_files[0])
    height, width, layers = first_img.shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating video from {len(img_files)} images...")
    
    # Add each image to video
    for i, img_file in enumerate(img_files):
        img = cv2.imread(img_file)
        if img is not None:
            video.write(img)
            if (i + 1) % 10 == 0:  # Progress indicator
                print(f"Processed {i + 1}/{len(img_files)} images")
    
    # Release everything
    video.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved as: {output_path}")

# Example usage
if __name__ == "__main__":
    # Configuration
    input_folder = "/home/shuklva/CUT3R/results/tmp-zebr-15_1-revisit-1/combined_orientation_arrows"  # Folder containing your images
    output_video = "/home/shuklva/CUT3R/results/tmp-zebr-15_1-revisit-1/combined_orientation_arrows/output_video.mp4"
    frame_rate = 10  # Adjust as needed
    image_extension = "png"  # Change to "png" if needed
    
    # Create video
    images_to_video(input_folder, output_video, frame_rate, image_extension)