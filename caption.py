from transformers import pipeline
import argparse
import re
import os



# Generate a caption for the given image using a pre-trained model
def generate_caption(image_path):
    cap = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    return cap(image_path)[0]["generated_text"]


# Add a prefix string to the caption
def prefix(caption: str, string: str) -> str:
    return string + caption


# Add a suffix string to the caption
def suffix(caption: str, string: str) -> str:
    return caption + string


# Replace spaces in the caption with the given string
def glue(caption: str, string: str) -> str:
    return caption.replace(" ", string)


# Change the case of the caption based on the case_type
def case(caption: str, case_type: str) -> str:
    if case_type == "upper":
        return caption.upper()
    elif case_type == "lower":
        return caption.lower()
    elif case_type == "title":
        return caption.title()
    elif case_type == "sentence":
        return caption.capitalize()
    else:
        return caption 


# Remove punctuation, special characters, and extra spaces from the caption
def normalise(caption: str) -> str:
    caption = re.sub(r"[^A-Za-z0-9\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption)
    return caption.strip()


# Parse command-line arguments
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('image', help='Path to the image file or folder')
    p.add_argument('--prefix', type=str, default=None)
    p.add_argument('--suffix', type=str, default=None)
    p.add_argument('--glue', type=str, default=None)
    p.add_argument('--case', type=str, choices=['upper', 'lower', 'title', 'sentence'], required=None)
    return p.parse_args()


# Rename the file, appending a number if the target name already exists
def rename_file(path: str, new_name: str) -> None:
    dir_name = os.path.dirname(path)
    base, ext = os.path.splitext(new_name)
    new_path = os.path.join(dir_name, new_name)
    counter = 1
    while os.path.exists(new_path):
        new_name_candidate = f"{base}{counter}{ext}"
        new_path = os.path.join(dir_name, new_name_candidate)
        counter += 1
    os.rename(path, new_path)



# Main script execution
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Determine if input is a folder or a single image
    if os.path.isdir(args.image):
        image_paths = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_paths = [args.image]

    # Process each image
    for image_path in image_paths:
        # Generate caption
        caption = generate_caption(image_path)
        # Normalise caption
        caption = normalise(caption)
        # Apply transformations based on flags
        if args.glue:
            caption = glue(caption, args.glue)
        if args.prefix:
            caption = prefix(caption, args.prefix)
        if args.suffix:
            caption = suffix(caption, args.suffix)
        if args.case:
            caption = case(caption, args.case)
        # Rename the image file
        rename_file(image_path, caption + os.path.splitext(image_path)[1])
        # Print the final caption
        print(caption)

