import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt

class Image_localization():
    def __init__(self, path, model_type=1):
        self.image_path=path
        image=tf.io.read_file(filename=self.image_path)
        self.image=tf.image.decode_image(image)
        self.model_type=model_type

        def model_exceptor (model_path):
            try:
                main_model=hub.load(handle=model_path)
            except Exception as e:
                print ("An error occured {}".format(e))
            return main_model
        
        if self.model_type==1:
            self.main_model=model_exceptor(model_path="https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")

        if self.model_type==2:
            self.main_model=model_exceptor(model_path='https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1')
        self.main_model.signatures.keys()
        self.detector=self.main_model.signatures['default']

    def draw_boxes_around_image (self, 
                                    image, 
                                    ymin, 
                                    xmin, 
                                    ymax, 
                                    xmax, 
                                    color, 
                                    main_display_string=(),
                                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",25)):
        pass
        draw=PIL.ImageDraw.Draw(image,
                                mode=None)

        image_width, image_height = image.size()

        (left, right, top, bottom)=(xmin*image_width, 
                                    xmax*image_width,
                                    ymin*image_height,
                                    ymax*image_height)

        draw.line([(left, top), 
                    (left, bottom),
                    (right, bottom),
                    (right, top)],
                    fill=color,
                    width=2,
                    joint=None)

        display_string=[font.getsize(ds)[1] for ds in main_display_string]
        total_string_height=(1 + 2 *0.05) * sum(display_string)

        if top > total_string_height:
            text_bottom=top
        else:
            text_bottom=top+total_string_height

        for display_str in main_display_string[::-1]:
            width, height=font.getsize(display_str)
            margin=np.ceil(0.05 * height)

            draw.rectangle([(left, text_bottom - height - 2 * margin),
                            (left + width , text_bottom)],
                            fill=color)

            draw.text((left + margin, text_bottom - height - margin), 
                        text= display_str,
                        fill='black',
                        font=font)

            text_bottom=text_bottom-height -2 * margin

    def draw_boxes (self,image, boxes, classes, predicts, max_boxes=10, min_score=0.5):

        colors=list(ImageColor.colormap.values())

        for i in range (min(boxes.shape[0], max_boxes)):
            if predicts[i] >= min_score:
                print ("Predicted {}: {}".format(classes[i].decode('ascii'),
                                                    predicts[i]))
                
                ymin, xmin, ymax, xmax = tuple(boxes[i])

                color=colors[hash(classes[i]) % len(colors)]
                image_pil=Image.fromarray(np.uint8(image)).convert("RGB")

                self.draw_boxes_around_image(image=image_pil, 
                                            ymin=ymin,
                                            xmin=xmin, 
                                            ymax=ymax,
                                            xmax=xmax, 
                                            color=color,
                                            main_display_string= "{} : {}".format(
                                                classes[i].decode('ascii'),
                                                int (100*predicts[i])
                                            ))
                np.copyto(np.array(image), np.array(image_pil))

        return image

    def __call__ (self):
        pass
        image_converted=tf.image.convert_image_dtype(image=self.image, 
                                                    dtype=tf.float32)[tf.newaxis, ...]
        results=self.detector(image_converted)

        results={keys : values.numpy() for keys, values in results.items()}

        Image_drew=self.draw_boxes(image=self.image, 
                                    boxes=results['detection_boxes'], 
                                    classes=results['detection_class_entites'], 
                                    predicts=results['detection_scores'])

        plt.figure(figsize=(20, 15))
        plt.grid(False)
        plt.imshow(Image_drew)


Image_localization(path='ANY.jpeg',
                    model_type=1)
