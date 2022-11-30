import sys,os,re
import glob


def _paddingFileInFolder(folderPath):
    for filePath in glob.glob(folderPath + '/*.png'):
        filename = os.path.basename(filePath)
        split = re.split('\.',filename)
        paddedNumber = split[0].zfill(10)
        newName = paddedNumber + '.' + split[1]
        os.rename(filePath, os.path.join(folderPath, newName))
    return

if __name__ == "__main__":
    if os.path.isdir(sys.argv[1]):
        _paddingFileInFolder(sys.argv[1])
    else:
        raise ValueError("Input folder does not exist: {}".format(sys.argv[1]))
