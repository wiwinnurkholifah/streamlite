import base64


def encode_image_base64(img_buffer):
    img = img_buffer.getvalue()
    encoded_image = base64.b64encode(img)
    return encoded_image


def decode_image_base64(decoded_img):
    decoded_img = base64.b64decode(decoded_img)
    return decoded_img
