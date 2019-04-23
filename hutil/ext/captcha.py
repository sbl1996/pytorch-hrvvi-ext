# coding: utf-8
"""
    captcha.image
    ~~~~~~~~~~~~~

    Generate Image CAPTCHAs, just the normal image CAPTCHAs you are using.
"""

import os
import random
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
from io import BytesIO

DEFAULT_FONTS = []

__all__ = ['ImageCaptcha']


table = [i * 1.97 for i in range(256)]
m_table = [0] + [255] * 255


class _Captcha(object):
    def generate(self, chars, format='png'):
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars, output, format='png'):
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        im = self.generate_image(chars)
        return im.save(output, format=format)


class ImageCaptcha(_Captcha):
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """

    def __init__(self, width, height, fonts, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts
        self._font_sizes = font_sizes or (42, 50, 56)
        self._truefonts = []

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    @staticmethod
    def create_noise_curve(image, color):
        w, h = image.size
        x1 = random.randint(0, int(w / 5))
        x2 = random.randint(w - int(w / 5), w)
        y1 = random.randint(int(h / 5), h - int(h / 5))
        y2 = random.randint(y1, h - int(h / 5))
        points = [x1, y1, x2, y2]
        end = random.randint(160, 200)
        start = random.randint(0, 20)
        Draw(image).arc(points, start, end, fill=color)
        return image

    @staticmethod
    def create_noise_dots(image, color, width=3, number=30):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image

    def _draw_character(self, draw, c, color, rotate):
        font = random.choice(self.truefonts)
        w, h = draw.textsize(c, font=font)

        dx = random.randint(0, 4)
        dy = random.randint(0, 6)
        im = Image.new('RGBA', (w + dx, h + dy))
        Draw(im).text((dx, dy), c, font=font, fill=color)

        # rotate
        im = im.crop(im.getbbox())
        im = im.rotate(random.uniform(-rotate, rotate),
                       Image.BILINEAR, expand=1)

        # warp
        dx = w * random.uniform(0.1, 0.3)
        dy = h * random.uniform(0.2, 0.3)
        x1 = int(random.uniform(-dx, dx))
        y1 = int(random.uniform(-dy, dy))
        x2 = int(random.uniform(-dx, dx))
        y2 = int(random.uniform(-dy, dy))
        w2 = w + abs(x1) + abs(x2)
        h2 = h + abs(y1) + abs(y2)
        data = (
            x1, y1,
            -x1, h2 - y2,
            w2 + x2, h2 + y2,
            w2 - x2, -y1,
        )
        im = im.resize((w2, h2))
        im = im.transform((w, h), Image.QUAD, data)
        return im

    def create_captcha_image(self, chars, color, background, rotate):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)

        images = []
        all_bboxes = []
        for c in chars:
            if random.random() > 0.5:
                images.append(self._draw_character(draw, " ", color, rotate))
                all_bboxes.append(None)
            img = self._draw_character(draw, c, color, rotate)
            images.append(img)
            bbox = list(img.getbbox())
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]
            all_bboxes.append(bbox)
        images.append(self._draw_character(draw, " ", color, rotate))
        all_bboxes.append(None)

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        x_offset = int(average * 0.1)

        anns = []
        for i, im in enumerate(images):
            w, h = im.size
            x_offset = min(x_offset, width - 1 - w)
            im_l = im.convert('L')
            cmask = im_l.point(table)
            y_offset = (self._height - h) // 2
            image.paste(im, (x_offset, y_offset), cmask)
            if all_bboxes[i]:
                bbox = all_bboxes[i]
                bbox[0] += x_offset
                bbox[1] += y_offset
                ann = {
                    'bbox': bbox,
                }
                c = chars[len(anns)]
                mask = Image.new('1', (width, self._height), 0)
                mask.paste(im_l.point(m_table).convert(
                    '1'), (x_offset, y_offset))
                ann['segmentation'] = mask
                anns.append(ann)

            x_offset += w + random.randint(-rand, rand)

        if width > self._width:
            image = image.resize((self._width, self._height))
            sw = self._width / width
            for ann in anns:
                bbox = ann['bbox']
                bbox[0] *= sw
                bbox[2] *= sw
                mask = ann['segmentation']
                ann['segmentation'] = mask.resize((self._width, self._height))

        return image, anns

    def generate_image(self, chars, noise_dots=1.0, noise_curve=1.0, rotate=30):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = random_color(238, 255)
        color = random_color(10, 200, random.randint(220, 255))
        img, anns = self.create_captcha_image(
            chars, color, background, rotate=rotate)
        if random.random() < noise_dots:
            self.create_noise_dots(img, color)
        if random.random() < noise_dots:
            self.create_noise_curve(img, color)
        img = img.filter(ImageFilter.SMOOTH)

        return img, anns


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)
