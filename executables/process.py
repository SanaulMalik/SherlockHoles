#Changes made to the original process.py from https://github.com/techzizou/yolov4-tiny-custom_Training
#Need to compare the output with the original process.py on Colab
import os,glob

main_dir = os.path.dirname(os.path.abspath(__file__))

print(main_dir)

main_dir = 'data/Images'

file_train = open('data/train.txt', 'w')
file_test = open('data/test.txt','w')

current_dir = os.path.join(main_dir, 'train')
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*jpg")):
    title,ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(main_dir + "/train" + "/"+ title + ".jpg" + "\n")

current_dir = os.path.join(main_dir, 'test')
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*jpg")):
    title,ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_test.write(main_dir + "/test" + "/"+ title + ".jpg" + "\n")

