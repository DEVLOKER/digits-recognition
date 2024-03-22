import numpy as np
import base64
from PIL import Image, ImageOps, ImageChops
from skimage import exposure
from io import BytesIO
import win32gui
from PIL import ImageGrab, Image

def canvas_to_image(root, canvas):
    canvas.update()
    canvas.update_idletasks()
    # HWND = canvas.winfo_id()
    # rect = win32gui.GetWindowRect(HWND)
    # list_rect = list(rect)
    # list_frame = [-9, -38, 9, 9]
    # final_rect = tuple((map(lambda x,y:x-y,list_rect,list_frame))) #subtracting two lists
    # img = ImageGrab.grab(bbox=final_rect)


    # Get the dimensions of the canvas
    x = canvas.winfo_x() + root.winfo_rootx() + 72
    y = canvas.winfo_y() + root.winfo_rooty() + 70
    w = 604 # canvas.winfo_width()
    h = 606 # canvas.winfo_height()
    img = ImageGrab.grab().crop((x, y, x+w, y+h))
    print((x, y, w, h))


    # Create a PostScript representation of the canvas
    # ps = canvas.postscript(colormode='color', pagewidth=canvas.winfo_width(), pageheight=canvas.winfo_height())
    # eps = canvas.postscript(colormode='color')

    # Convert the PostScript to an image using PIL
    # img = Image.open(BytesIO(eps.encode('utf-8')))
    # img = Image.open(BytesIO(bytes(eps,'ascii')))

    # img.save("processed_image.jpeg")
    return img


def data_uri_to_image(uri):
    encoded_data = uri.split(',')[1]
    image = base64.b64decode(encoded_data)
    return Image.open(BytesIO(image))


def replace_transparent_background(image):
    image_arr = np.array(image)

    has_no_alpha = len(image_arr.shape) < 3 or image_arr.shape[2] < 4
    if has_no_alpha:
        return image

    alpha1 = 0
    r2, g2, b2, alpha2 = 255, 255, 255, 255

    red, green, blue, alpha = image_arr[:, :, 0], image_arr[:, :, 1], image_arr[:, :, 2], image_arr[:, :, 3]
    mask = (alpha == alpha1)
    image_arr[:, :, :4][mask] = [r2, g2, b2, alpha2]

    return Image.fromarray(image_arr)


def trim_borders(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    
    return image


def pad_image(image):
    return ImageOps.expand(image, border=30, fill='#fff')


def resize_image(image, width=8, height=8):
    return image.resize((width, height), Image.LINEAR)


def invert_colors(image):
    return ImageOps.invert(image)


def scale_down_intensity(image):
    image_arr = np.array(image)
    image_arr = exposure.rescale_intensity(image_arr, out_range=(0, 16))
    return Image.fromarray(image_arr)


def to_grayscale(image):
    return image.convert('L')
