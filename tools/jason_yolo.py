from PIL import Image
import glob
import json


label_path = './yolo_train_data/fashion2/data/labels/';
image_list = './Yolo_image_list.txt'
outfile_name = './Yolo_train_data.json'
one_decimal = "{0:0.1f}"
final_json = []

num = 0;
with open(image_list) as f:
    for path in f:
        image_path = path[:-1]
        image_name = image_path[image_path.rfind("/")+1:-4]
        image = Image.open(image_path)
        width, height = image.size
        print num
        num = num +1

        label = label_path + image_name + ".txt"
        with open(label) as f_l:
            #print image_name
            json_dict = {}
            json_dict["image_path"] = image_path
            json_dict["labels"] = []
            json_dict["rects"] = []
            for line in f_l:

                box = line.split()
                label = box[0]
                json_dict["labels"].append(label)
                xc = float(box[1])
                yc = float(box[2])
                w = float(box[3])
                h = float(box[4])
                x1 = (xc - 0.5*w)*float(width)
                x2 = (xc + 0.5*w )*float(width)
                y1 = (yc - 0.5*h)*float(height)
                y2 = (yc + 0.5*h)*float(height)

                if (x1<x2) and (y1<y2):
                    rects = {}
                    rects["x1"] = float(one_decimal.format(x1))
                    rects["x2"] = float(one_decimal.format(x2))
                    rects["y1"] = float(one_decimal.format(y1))
                    rects["y2"] = float(one_decimal.format(y2))
                    json_dict["rects"].append(rects)
                else:
                    pass
            final_json.append(json_dict)

outfile = open(outfile_name, 'w')
json.dump(final_json, outfile, sort_keys = True, indent = 4)
outfile.close()
