
import glob
import imageio
import os

# pwd = os.path.abspath(__file__)
pwd = os.path.dirname(__file__)

imgs = []

imgs_path = glob.glob( pwd + '/../output/*.jpg')

imgs_path = sorted(imgs_path)
print(len(imgs_path))

for pp in imgs_path:

    imgs.append( imageio.imread(pp) )

imageio.mimsave(pwd + '/generator.gif', imgs, fps=5)