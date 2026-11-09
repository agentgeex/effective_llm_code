import random
import string
from io import BytesIO
from PIL import Image, ImageFont, ImageDraw

class ImageCode:


    """生成用于绘制字符串的随机颜色(可以随意指定0-255之间的数字)"""
    def rand_color(self):

        red = random.randint(32, 200)
        green = random.randint(22, 255)
        blue = random.randint(0, 200)
        return red, green, blue

    """生成4位随机字符串"""
    def gen_text(self):

        # sample 用于从一个大的列表或字符串中，随机取得N个字符，来构建出一个子列表
        list = random.sample("abcdeghjkmnpqrstuvwxyABCDEFHJKMNPQRSTUVWXY2345689", 4)
        print(string.ascii_letters)
        return ''.join(list)

    def draw_lines(self, draw, num, width, height):
        """
        绘制干扰线
        :param draw: 图片对象
        :param num: 干扰线数量
        :param width: 图片的宽
        :param height: 图片的高
        :return:
        """
        for num in range(num):
            x1 = random.randint(0, width / 2)
            y1 = random.randint(0, height / 2)
            x2 = random.randint(0, width)
            y2 = random.randint(height / 2, height)
            draw.line(((x1, y1), (x2, y2)), fill='black', width=2)


    """绘制验证码图片"""
    def draw_verify_code(self):

        code = self.gen_text()
        width, height = 90, 20  # 设定图片大小，可根据实际需求调整
        im = Image.new('RGB', (width, height), 'white')  # 创建图片对象，并设定背景色为白色
        draw = ImageDraw.Draw(im)  # 新建ImageDraw对象
        # 绘制字符串
        for i in range(4):
            draw.text((5 + random.randint(-3, 3) + 23 * i, 5 + random.randint(-3, 3)), text=code[i],
                      fill=self.rand_color())
            #self.draw_lines(draw, 4, width, height)  # 绘制干扰线
        # im.show()  # 如需临时调试，可以直接将生成的图片显示出来
        return im, code