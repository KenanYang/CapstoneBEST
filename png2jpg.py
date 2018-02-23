from PIL import Image
def png2jpg(page):
    im = Image.open(r"/Users/kenanyang/PycharmProjects/CombineP6/upload/"+str(page)+"/test.png")
    bg = Image.new("RGB", im.size, (255,255,255))
    bg.paste(im,im)
    # tran = bg.transpose(Image.FLIP_LEFT_RIGHT)
    # tran.rotate(180).save(r"/Users/kenanyang/PycharmProjects/CombineP6/upload/"+str(page)+"/test.jpg")
    bg.save(r"/Users/kenanyang/PycharmProjects/CombineP6/upload/"+str(page)+"/test.jpg")