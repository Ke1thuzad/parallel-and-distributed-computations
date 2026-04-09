from PIL import Image
import struct
import ctypes

def transform_image(name):
    img = Image.open(name + '.png')
    (w, h) = img.size[0:2]
    pix = img.load()
    buff = ctypes.create_string_buffer(4 * w * h)
    offset = 0
    for j in range(h):
        for i in range(w):
            r = bytes((pix[i, j][0],))
            g = bytes((pix[i, j][1],))
            b = bytes((pix[i, j][2],))
            a = bytes((255,))
            struct.pack_into('cccc', buff, offset, r, g, b, a)
            offset += 4
    out = open('in.data', 'wb')
    out.write(struct.pack('ii', w, h))
    out.write(buff.raw)
    out.close()

def revert_image(name):
    fin = open(name + '.data', 'rb')
    (w, h) = struct.unpack('ii', fin.read(8))
    buff = ctypes.create_string_buffer(4 * w * h)
    fin.readinto(buff)
    fin.close()
    img = Image.new('RGBA', (w, h))
    pix = img.load()
    offset = 0
    for j in range(h):
        for i in range(w):
            (r, g, b, a) = struct.unpack_from('cccc', buff, offset)
            pix[i, j] = (ord(r), ord(g), ord(b), ord(a))
            offset += 4
    img.save('out.png')

def main():
    # transform_image('field')
    revert_image('out')

if __name__ == '__main__':
    main()