import os
import cv2
import base64
import requests
import json
import csv
import glob
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def extract_frames(video_path, num_frames=10):
    """
    Extract equally spaced frames from a video

    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to extract

    Returns:
        list: List of extracted frames as numpy arrays
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError(f"Could not read frames from video: {video_path}")

    # Calculate frame indices to extract (equally spaced)
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    # Extract frames
    frames = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            frames.append(frame)

    # Release the video
    video.release()

    return frames


def save_frames(frames, output_dir):
    """
    Save frames as image files

    Args:
        frames (list): List of frames as numpy arrays
        output_dir (str): Directory to save images

    Returns:
        list: List of paths to saved images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save frames as images
    image_paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(path, frame)
        image_paths.append(path)

    return image_paths


def encode_image(image_path):
    """
    Encode image as base64 string

    Args:
        image_path (str): Path to image file

    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_caption_from_openai(image_paths, context):
    """
    Get caption from OpenAI API using extracted frames

    Args:
        image_paths (list): List of paths to image files
        context: shares the context of the dataset with the ChatGPT prompt

    Returns:
        str: Generated caption
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")

    # Prepare images for the API request
    content = [
        {
            "type": "text",
            "text": f"Write a short caption of video containing 3-4 short sentences according to the provided video. Provide only caption. Video context: {context} dataset video"
        }
    ]

    # Add image content to the API request
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    # Make API request
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 300
        }
    )

    # Check for successful response
    if response.status_code != 200:
        raise Exception(f"Error from OpenAI API: {response.text}")

    # Extract and return the caption
    response_data = response.json()
    return response_data['choices'][0]['message']['content']


def process_video(video_path, output_dir, context, num_frames=10):
    """
    Process a single video to extract frames and get caption

    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save outputs
        num_frames (int): Number of frames to extract

    Returns:
        str: Generated caption
    """
    try:
        # Create video-specific output directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_dir, video_name)
        frames_dir = os.path.join(video_output_dir, "frames")

        # Extract frames from video
        print(f"Extracting {num_frames} frames from video: {video_path}")
        frames = extract_frames(video_path, num_frames)
        print(f"Extracted {len(frames)} frames")

        # Save frames as images
        image_paths = sorted(save_frames(frames, frames_dir))
        print(f"Saved frames to {frames_dir}")

        # Get caption from OpenAI API
        print("Getting caption from OpenAI API...")
        caption = get_caption_from_openai(image_paths, context)

        # Save caption to file
        caption_path = os.path.join(video_output_dir, "caption.txt")
        with open(caption_path, "w") as f:
            f.write(caption)

        print(f"Caption saved to {caption_path}")
        print("\nGenerated Caption:")
        print("-" * 50)
        print(caption)
        print("-" * 50)

        return caption
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return f"Error: {e}"


def find_mp4_files(directory="."):
    """
    Find all MP4 files in the specified directory

    Args:
        directory (str): Directory to search for MP4 files

    Returns:
        list: List of paths to MP4 files
    """
    mp4_pattern = os.path.join(directory, "*.mp4")
    return glob.glob(mp4_pattern)


def save_results_to_csv(results, output_path, context):
    """
    Save results to CSV file

    Args:
        results (dict): Dictionary mapping video paths to captions
        output_path (str): Path to save CSV file
        context(str): Context of the video is written it the end of the file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['content_path', 'caption'])
        for video_path, caption in results.items():
            writer.writerow([video_path, caption])


def main():
    """Main function to run the script"""
    import argparse

    parser = argparse.ArgumentParser(description="Process all MP4 files in current directory and generate captions")
    parser.add_argument("--output", default="output", help="Directory to save frames and captions")
    parser.add_argument("--frames", type=int, default=15, help="Number of frames to extract per video")
    parser.add_argument("--csv", default="captions.csv", help="Output CSV file name")
    parser.add_argument("--context", default="Running fast", help="Context for the video")

    args = parser.parse_args()

    # Find all MP4 files in the current directory
    mp4_files = find_mp4_files()

    if not mp4_files:
        print("No MP4 files found in the current directory")
        return

    print(f"Found {len(mp4_files)} MP4 files")

    # Process each video
    results = {}
    for video_path in mp4_files:
        print(f"\nProcessing {video_path}...")
        caption = process_video(video_path, args.output, args.context, args.frames)
        results[video_path] = caption

    # Save results to CSV
    csv_path = args.csv
    save_results_to_csv(results, csv_path, args.context)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()