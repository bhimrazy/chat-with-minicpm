import base64
import os
import re
from io import BytesIO

import requests
from decord import VideoReader, cpu
from litserve.specs.openai import ChatCompletionRequest, ChatMessage
from PIL import Image
import concurrent.futures

from src.config import IMAGE_EXTENSIONS, MAX_NUM_FRAMES, VIDEO_EXTENSIONS


def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()


def is_image(filename):
    return get_file_extension(filename) in IMAGE_EXTENSIONS


def is_video(filename):
    return get_file_extension(filename) in VIDEO_EXTENSIONS


def read_image(source):
    """
    Read an image from a real image URL or a base64-encoded URL.

    Parameters:
    source (str): The image source. Can be a real image URL or a base64 URL string.

    Returns:
    Image or None: The Image object if the source is valid, otherwise None.
    """
    try:
        if re.match(r"^https?://", source):
            # It's a real image URL
            return Image.open(requests.get(source, stream=True).raw)
        elif re.match(r"^data:image/.+;base64,", source):
            # It's a base64 image URL
            base64_image = source.split(",")[1]
            image_data = base64.b64decode(base64_image)
            return Image.open(BytesIO(image_data))
        else:
            return Image.open(source)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def encode_image(image_source):
    """
    Encode an image to a base64 data URL based object.

    Parameters:
    image_source (str or Image): The image source. Can be a real image URL, an Image instance, or a base64 URL string.

    Returns:
    str or None: The base64-encoded data URL of the image if successful, otherwise None.
    """
    try:
        if isinstance(image_source, str):
            image = read_image(image_source)
            if image is None:
                return None
        elif isinstance(image_source, Image.Image):
            image = image_source
        else:
            image = Image.open(image_source)

        # resize to max_size
        max_size = 448 * 16
        if max(image.size) > max_size:
            w, h = image.size
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)

        buffered = BytesIO()
        # Use image format or default to "JPEG"
        image_format = image.format if image.format else "JPEG"
        image.save(buffered, format=image_format)
        mime_type = f"image/{image_format.lower()}"
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = f"data:{mime_type};base64,{encoded_image}"

        image_object = {
            "type": "image_url",
            "image_url": {
                "url": url,
            },
        }
        return image_object
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def encode_video(video):
    """Encode a video to a list of base64 data URLs.

    Adapted from: https://huggingface.co/spaces/openbmb/MiniCPM-V-2_6/blob/52bef569e63422a5c3e5913b148af1be9a6d5188/app.py#L188
    """
    print("Got video", video)
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    if hasattr(video, "path"):
        vr = VideoReader(video.path, ctx=cpu(0))
    else:
        vr = VideoReader(video, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    video = vr.get_batch(frame_idx).asnumpy()
    # video = [Image.fromarray(v.astype("uint8")) for v in video]
    # video = [encode_image(v) for v in video]

    def process_frame(frame):
        print("processing frame", frame.shape)
        return encode_image(Image.fromarray(frame.astype("uint8")))
    print("Processing video frames...")
    # Use ThreadPoolExecutor to parallelize the encoding process
    with concurrent.futures.ThreadPoolExecutor() as executor:
        video = list(executor.map(process_frame, video))

    return video


def parse_messages(request: ChatCompletionRequest):
    """
    Parse messages from a ChatCompletionRequest object, extracting text content and images.

    Parameters:
    request (ChatCompletionRequest): The request object containing messages.

    Returns:
    Tuple[List[ChatMessage], List[Image]]: A tuple containing a list of ChatMessage objects with updated content
    and a list of PIL Image objects representing the images extracted from the messages.
    """
    messages = []
    images = []

    for message in request.messages:
        if isinstance(message.content, list):
            text_content = ""

            for content_item in message.content:
                if content_item.type == "text":
                    text_content += content_item.text
                elif content_item.type == "image_url":
                    image_url = content_item.image_url
                    if image_url:
                        image = read_image(image_url)
                        if image:
                            images.append(image)
            messages.append(ChatMessage(role=message.role, content=text_content))
        else:
            messages.append(message)

    # dump to json
    messages = [message.model_dump_json(exclude_none=True) for message in messages]
    print(messages)
    return messages, images
