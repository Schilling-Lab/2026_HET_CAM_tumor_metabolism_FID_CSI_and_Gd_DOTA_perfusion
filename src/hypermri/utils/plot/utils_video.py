import os
import re

from PIL import Image


def create_gif(
    dir_path,
    name_structure,
    start_index=None,
    end_index=None,
    duration_s=1.0,
    output_filename="None",
    extension=".png",
    save_path=None,
):
    """
    Create a GIF from a sequence of images.

    Parameters
    ----------
    dir_path : str
        Directory where images are stored.
    name_structure : str
        Base name of images. If it includes a format placeholder (e.g., 'image_{:02d}'),
        the function will use it to construct filenames. If it does not include a placeholder
        (e.g., 'image_'), the function will automatically detect images starting with this
        name and ending with the specified extension.
    start_index : int, optional
        Starting index of images. If not provided, the function determines the smallest index
        from the available files.
    end_index : int, optional
        Ending index of images. If not provided, the function determines the largest index
        from the available files.
    duration_s : int, optional
        Duration of each frame in seconds (default is 1).
    output_filename : str, optional
        Filename for the output GIF (default is 'output.gif').
    extension : str, optional
        File extension of images (default is '.png').

    Returns
    -------
    None

    Notes
    -----
    The function loads images from the specified directory, assembles them into a GIF,
    and saves the result to the specified output filename. It handles missing files
    and continues processing the existing ones. If `start_index` or `end_index` are not
    provided, the function automatically determines them based on the files available.

    Examples
    --------
    Create a GIF from images named 'image_01.png' to 'image_10.png':

    >>> create_gif('images', 'image_', duration_s=1, output_filename='output.gif')

    Create a GIF from images named 'slide1.jpg' to 'slide5.jpg':

    >>> create_gif('slides', 'slide', duration_s=2, output_filename='presentation.gif', extension='.jpg')

    """
    # Determine if name_structure includes a format placeholder
    has_placeholder = "{}" in name_structure or "{:" in name_structure
    if has_placeholder:
        # If a placeholder exists, use it to construct filenames
        # Determine start_index and end_index if they are not provided
        if start_index is None or end_index is None:
            # Scan the directory for matching files and extract indices
            indices = []
            for filename in os.listdir(dir_path):
                if filename.endswith(extension):
                    try:
                        # Extract index using the name_structure format
                        pattern = name_structure + extension
                        name_part = filename[: -len(extension)]
                        index_str = name_part.replace(name_structure.format(""), "")
                        index = int(index_str)
                        indices.append(index)
                    except ValueError:
                        continue
            if not indices:
                print("No matching images were found in the directory.")
                return
            indices.sort()
            if start_index is None:
                start_index = indices[0]
            if end_index is None:
                end_index = indices[-1]

        # Load images within the specified index range
        images = []
        for i in range(start_index, end_index + 1):
            filename = f"{name_structure.format(i)}{extension}"
            filepath = os.path.join(dir_path, filename)
            if not os.path.exists(filepath):
                print(f"File {filepath} does not exist.")
                continue
            try:
                with Image.open(filepath) as img:
                    images.append(img.copy())
            except Exception as e:
                print(f"Could not open image {filepath}: {e}")
                continue
    else:
        # If no placeholder, match files starting with name_structure and
        # ending with extension
        pattern = re.escape(name_structure) + r"(\d+)" + re.escape(extension) + r"$"
        regex = re.compile(pattern)

        # Scan the directory for matching files and extract indices
        files = []
        for filename in os.listdir(dir_path):
            match = regex.match(filename)
            if match:
                index = int(match.group(1))
                files.append((index, filename))

        if not files:
            print("No matching images were found in the directory.")
            return

        # Sort files based on indices
        files.sort(key=lambda x: x[0])

        # Determine start_index and end_index if they are not provided
        indices = [index for index, _ in files]
        if start_index is None:
            start_index = indices[0]
        if end_index is None:
            end_index = indices[-1]

        # Filter files based on the specified index range
        files = [
            filename for index, filename in files if start_index <= index <= end_index
        ]

        # Load images
        images = []
        for filename in files:
            filepath = os.path.join(dir_path, filename)
            try:
                with Image.open(filepath) as img:
                    images.append(img.copy())
            except Exception as e:
                print(f"Could not open image {filepath}: {e}")
                continue

    if not images:
        print("No images were loaded. GIF will not be created.")
        return

    if output_filename == "None":
        output_filename = name_structure + "gif.gif"

    # Convert duration_s to milliseconds for easier handling
    duration_ms = 1000.0 * duration_s

    if save_path is None:
        save_path = dir_path

    try:
        images[0].save(
            os.path.join(save_path, output_filename),
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"GIF saved as {output_filename} in {save_path}/ ")
    except Exception as e:
        print(f"Could not save GIF: {e}")
