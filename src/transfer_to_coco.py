import numpy as np
import json
import glob


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class tococo(object):
    def __init__(self, jsonfile, save_path):
        self.images = []
        self.categories = []
        self.annotations = []
        self.jsonfile = jsonfile
        self.save_path = save_path  # 保存json的路径
        self.class_ids = {"BG": 0}
        self.class_id = 0
        self.coco = {}

    def labelme_to_coco(self):
        annid = 0
        for num, json_file in enumerate(self.jsonfile):
            print('\r', '{}/{}, file_Dir:{}'.format(num, len(self.jsonfile), json_file), end='')

            data = open(json_file, "r")
            data = json.load(data)
            self.images.append(self.get_images(data["imagePath"], data["imageHeight"], data["imageWidth"], num))
            shapes = data["shapes"]
            for shape in shapes:
                label = shape["label"]
                if shape["label"] in self.class_ids:
                    pass
                if shape["label"] not in self.class_ids and shape["label"] in ["people", 'peopel']:
                    label = 'people'
                    self.class_id = 1

                    self.class_ids[label] = self.class_id
                    self.categories.append(self.get_categories(label, self.class_id))

                self.annotations.append(self.get_annotations(shape["points"], num, annid, label))

                annid = annid + 1

        self.coco["images"] = self.images

        self.coco["categories"] = [dict(t) for t in set([tuple(d.items()) for d in self.categories])]

        [dict(t) for t in set([tuple(d.items()) for d in self.categories])]
        self.coco["annotations"] = self.annotations

    def get_images(self, filename, height, width, image_id):
        image = {}
        image["height"] = height
        image['width'] = width
        image["id"] = image_id
        image["file_name"] = filename
        return image

    def get_categories(self, name, class_id):
        category = {}
        category["supercategory"] = "pedestrian"
        category['id'] = class_id
        category['name'] = name

        return category

    def get_annotations(self, points, image_id, ann_id, calss_name):
        annotation = {}
        mins = np.amin(points, axis=0)
        maxs = np.amax(points, axis=0)
        wh = maxs - mins
        x = mins[0]
        y = mins[1]
        w = wh[0]
        h = wh[1]
        area = w * h
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = image_id
        annotation['bbox'] = [x, y, w, h]
        annotation['area'] = area
        annotation['category_id'] = self.class_ids[calss_name]
        annotation['id'] = ann_id
        return annotation

    def save_json(self):
        self.labelme_to_coco()
        coco_data = self.coco
        # 保存json文件
        json.dump(coco_data, open(self.save_path, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示


labelme_json = glob.glob('../data/abnormal/*.json')
c = tococo(labelme_json, '../output/vis.json')
c.save_json()
