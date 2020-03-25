import tkinter as tk
from tkinter import *
import datetime
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter.messagebox import showinfo
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
import os
import matplotlib.pyplot as plt

class goods_counting_gui:
    def __init__(self):
        super().__init__()
        self.create_gui()
    def create_gui(self):
        self.window = tk.Tk()
        self.window.title('Lung_disease_Recognition_GUI')
        self.label = tk.Label(self.window, text='Current time：', bg='green', font=30)  # text是要显示的内容
        self.label.grid(row=0, column=0)
        self.cur_time = tk.Label(self.window, text='%s%d'%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:'),
                                                          datetime.datetime.now().microsecond//100000), font=30)
        self.cur_time.grid(row=0, column=1)
        self.window.after(100, self.update_time)

        self.img_path = StringVar()
        self.label_yolo_obj = tk.Label(self.window, text='choose image: ')
        self.label_yolo_obj.grid(row=1, column=0, pady=6)
        self.entry2 = tk.Entry(self.window, textvariable=self.img_path, width=50)
        self.entry2.grid(row=1, column=1, padx=5, pady=6)

        self.button_yolo_obj = tk.Button(self.window, text='path of image', bg='pink', relief=tk.RAISED, width=14,
                                        height=1, command=self.get_img_path)
        self.button_yolo_obj.grid(row=1, column=2, padx=10, pady=6)

        self.button_start = tk.Button(self.window, text='start recognition', bg='gold', relief=tk.RAISED,
                                      width=13, height=1, command=self.start)
        self.button_start.grid(row=1, column=3, padx=10, pady=6)

        self.button_get_result = tk.Button(self.window, text='get detail', bg='orange', relief=tk.RAISED,
                                      width=14, height=1, command=self.get_result)
        self.button_get_result.grid(row=2, column=2, padx=10, pady=6)

        self.button_start = tk.Button(self.window, text='clear output', bg='red', relief=tk.RAISED, width=13, height=1,
                                      command=self.stop)
        self.button_start.grid(row=2, column=3, padx=10, pady=10)

        self.frm_ = tk.Frame()
        self.frm_.grid(row=3, column=1, padx=5)

        self.result_show = tk.Label(self.frm_)
        self.result_show.grid(row=4, column=1, padx=5)

        self.output = tk.Label(self.window, text='output display', font=50)
        self.output.grid(row=3, column=2, padx=5, pady=5, sticky=tk.NW)

        self.text = tk.Text(self.window, width=24, height=10)
        self.text.grid(row=3, column=3, pady=5, sticky=tk.NW)

    def get_img_path(self):
        # self.path1 = askdirectory()
        self.path1 = askopenfilename()
        self.img_path.set(self.path1)
        frame = cv2.imread(self.path1)
        B, G, R = cv2.split(frame)
        frame = cv2.merge([R, G, B])
        img = Image.fromarray(frame)  # 类型是PIL.Image.Image
        img = self.resize(img)
        imgtk = ImageTk.PhotoImage(img)
        self.result_show.config(image=imgtk)
        self.result_show.img = imgtk



    def update_time(self):
        self.cur_time.config(text=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.window.after(500, self.update_time)

    def start(self):
        # self.stop_flag = False
        t = threading.Thread(target=self.run_program())
        t.setDaemon(True)
        t.start()

    def stop(self):
        # self.stop_flag = True
        # self.window.quit() 退出窗体
        # self.result_show.quit()
        # frame = cv2.imread(self.path1)
        # B, G, R = cv2.split(frame)
        # frame = cv2.merge([R, G, B])
        # img = Image.fromarray(frame)  # 类型是PIL.Image.Image
        # img = self.resize(img)
        # imgtk = ImageTk.PhotoImage(img)
        # self.result_show.config(image=imgtk)
        # self.result_show.img = imgtk
        # for widget in self.frm_.winfo_children():
        #     widget.destroy()
        # self.result_show.grid_forget()
        self.text.delete('1.0', 'end')
        # exit()
        #pass



    def resize(self, image):
        im = image
        self.new_size = (600, 600)
        im.thumbnail(self.new_size,Image.ANTIALIAS)  # thumbnail() 函数是制作当前图片的缩略图, 参数size指定了图片的最大的宽度和高度。
        return im

    def run_program(self):
        weightsPath = "lung_x.weights"
        configPath = "lung_x.cfg"
        labelsPath = "lung_x.names"
        self.LABELS = open(labelsPath).read().strip().split("\n")  # 物体类别
        COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")  # 颜色
        boxes = []
        confidences = []
        self.classIDs = []
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        image = cv2.imread(self.path1)
        # print(self.path1)
        (H, W) = image.shape[:2]

        # 得到 YOLO需要的输出层
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # 从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        for i in os.listdir('./Atelectasis'):
            os.remove('./Atelectasis/' + i)
        for i in os.listdir('./Cardiomegaly'):
            os.remove('./Cardiomegaly/' + i)
        for i in os.listdir('./Effusion'):
            os.remove('./Effusion/' + i)
        for i in os.listdir('./Infiltrate'):
            os.remove('./Infiltrate/' + i)
        for i in os.listdir('./Mass'):
            os.remove('./Mass/' + i)
        for i in os.listdir('./Nodule'):
            os.remove('./Nodule/' + i)
        for i in os.listdir('./Pneumonia'):
            os.remove('./Pneumonia/' + i)
        for i in os.listdir('./Pneumothorax'):
            os.remove('./Pneumothorax/' + i)

        # 在每层输出上循环
        for output in layerOutputs:
            # 对每个检测进行循环
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # 过滤掉那些置信度较小的检测结果
                if confidence > 0.5:
                    # 框后接框的宽度和高度
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # 边框的左上角
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # 更新检测出来的框
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    self.classIDs.append(classID)

        # 极大值抑制
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
        # print('idxs =', idxs)
        self.num = 0
        self.get_num = []
        self.Atelectasis_num = 0
        self.Cardiomegaly_num = 0
        self.Effusion_num = 0
        self.Infiltrate_num = 0
        self.Mass_num = 0
        self.Nodule_num = 0
        self.Pneumonia = 0
        self.Pneumothorax = 0
        if len(idxs) > 0:
            for i in idxs.flatten():  # 把一列拉成一行,比如[[1] \ [0] \[2] ]  变成[1 0 2 ]
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # 在原图上绘制边框和类别
                color = [int(c) for c in COLORS[self.classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                # text = "{}: {:.4f}".format(self.LABELS[self.classIDs[i]], confidences[i])
                text = "{}".format((x + w, y + h))
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                self.get_num.append(self.num)
                self.num += 1

                if self.LABELS[self.classIDs[i]] == 'Atelectasis':
                    self.Atelectasis_num += 1
                    self.crop = image[y:y + h, x:x + w]
                    cv2.imwrite('./Atelectasis/Atelectasis{}.jpg'.format(self.Atelectasis_num), self.crop)

                if self.LABELS[self.classIDs[i]] == 'Cardiomegaly':
                    self.Cardiomegaly_num += 1
                    self.crop = image[y+50:y + h, x:x + w]
                    cv2.imwrite('./Cardiomegaly/Cardiomegaly{}.jpg'.format(self.Cardiomegaly_num), self.crop)

                if self.LABELS[self.classIDs[i]] == 'Effusion':
                    self.Effusion_num += 1
                    self.crop = image[y:y + h, x:x + w]
                    cv2.imwrite('./Effusion/Effusion{}.jpg'.format(self.Effusion_num), self.crop)

                if self.LABELS[self.classIDs[i]] == 'Infiltrate':
                    self.Infiltrate_num += 1
                    self.crop = image[y:y + h, x:x + w]
                    cv2.imwrite('./Infiltrate/Infiltrate{}.jpg'.format(self.Infiltrate_num), self.crop)

                if self.LABELS[self.classIDs[i]] == 'Mass':
                    self.Mass_num += 1
                    self.crop = image[y:y + h, x:x + w]
                    cv2.imwrite('./Mass/Mass{}.jpg'.format(self.Mass_num), self.crop)

                if self.LABELS[self.classIDs[i]] == 'Nodule':
                    self.Nodule_num += 1
                    self.crop = image[y:y + h, x:x + w]
                    cv2.imwrite('./Nodule/Nodule{}.jpg'.format(self.Nodule_num), self.crop)

                if self.LABELS[self.classIDs[i]] == 'Pneumonia':
                    self.Nodule_num += 1
                    self.crop = image[y:y + h, x:x + w]
                    cv2.imwrite('./Pneumonia/Pneumonia{}.jpg'.format(self.Pneumonia_num), self.crop)

                if self.LABELS[self.classIDs[i]] == 'Pneumothorax':
                    self.Nodule_num += 1
                    self.crop = image[y:y + h, x:x + w]
                    cv2.imwrite('./Pneumothorax/Pneumothorax{}.jpg'.format(self.Pneumothorax_num), self.crop)
        # cv2.imshow("Image", image)
        # cv2.imwrite('result.jpg', image)

        self.second_page = image
        frame = image # ndarray
        B, G, R = cv2.split(frame)
        frame = cv2.merge([R, G, B])
        img = Image.fromarray(frame)  # 类型是PIL.Image.Image
        img = self.resize(img)
        imgtk = ImageTk.PhotoImage(img)
        self.result_show.config(image=imgtk)
        self.result_show.img = imgtk

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # self.summary.append(self.LABELS[self.classIDs[i]])
                # print('self.summary =', self.summary)

                if self.LABELS[self.classIDs[i]] == 'Atelectasis':
                    self.text.insert(INSERT,'The location in{}'.format((x + w, y + h)) + 'contains  pathology：Atelectasis' + '\n'+ '\n')
                if self.LABELS[self.classIDs[i]] == 'Cardiomegaly':
                    self.text.insert(INSERT, 'The location in{}'.format((x + w, y + h)) + 'contains  pathology：Cardiomegaly' + '\n'+ '\n')
                if self.LABELS[self.classIDs[i]] == 'Effusion':
                    self.text.insert(INSERT, 'The location in{}'.format((x + w, y + h)) + 'contains  pathology：Effusion'+ '\n'+ '\n')
                if self.LABELS[self.classIDs[i]] == 'Infiltrate':
                    # (x, y) = (boxes[i][0], boxes[i][1])
                    self.text.insert(INSERT, 'The location in{}'.format((x + w, y + h)) + 'contains  pathology：Infiltrate'+ '\n'+ '\n')
                if self.LABELS[self.classIDs[i]] == 'Mass':
                    # (x, y) = (boxes[i][0], boxes[i][1])
                    self.text.insert(INSERT, 'The location in{}'.format((x + w, y + h)) + 'contains  pathology：Mass' + '\n'+ '\n')
                if self.LABELS[self.classIDs[i]] == 'Nodule':
                    # (x, y) = (boxes[i][0], boxes[i][1])
                    self.text.insert(INSERT,'The location in{}'.format((x + w, y + h)) + 'contains  pathology: Nodule' + '\n'+ '\n')
                if self.LABELS[self.classIDs[i]] == 'Pneumonia':
                    # (x, y) = (boxes[i][0], boxes[i][1])
                    self.text.insert(INSERT,'The location in{}'.format((x + w, y + h)) + 'contains  pathology: Pneumonia' + '\n'+ '\n')
                if self.LABELS[self.classIDs[i]] == 'Pneumothorax':
                    # (x, y) = (boxes[i][0], boxes[i][1])
                    self.text.insert(INSERT,'The location in{}'.format((x + w, y + h)) + 'contains  pathology: Pneumothorax' + '\n'+ '\n')

        # cv2.imwrite('result.jpg', image)
        # cv2.waitKey(5)
    def get_result(self):
        # self.Atelectasis_num = 0
        # self.Cardiomegaly_num = 0
        # self.Effusion_num = 0
        # self.Infiltrate_num = 0
        # self.Mass_num = 0
        # self.Nodule_num = 0
        # dic = {}
        # dic['Atelectasis'] = 'self.Atelectasis_num'
        # dic['Cardiomegaly'] = 'self.Cardiomegaly_num'
        # dic['Effusion'] = 'self.Effusion_num'
        # dic['Infiltrate'] = 'self.Infiltrate_num'
        # dic['Mass'] = 'self.Mass_num'
        # dic['Nodule'] = 'self.Nodule_num'

        Atelectasis_img = len(os.listdir('./Atelectasis'))
        Cardiomegaly_img = len(os.listdir('./Cardiomegaly'))
        Effusion_img = len(os.listdir('./Effusion'))
        Infiltrate_img = len(os.listdir('./Infiltrate'))
        Mass_img = len(os.listdir('./Mass'))
        Nodule_img = len(os.listdir('./Nodule'))
        Pneumonia_img = len(os.listdir('./Pneumonia'))
        Pneumothorax_img = len(os.listdir('./Pneumothorax'))

        if Atelectasis_img != 0:
            self.merge_into_one_pic('./Atelectasis/')
            # for i in os.listdir('./Atelectasis'):
            pic = cv2.imread('./Atelectasis/'+ 'result.jpg')
            name = 'Atelectasis：lung collapse (atelectasis) refers to the reduction in capacity or gas content in one or more ' \
                   'lung segments or lobes. Because of the absorption of gas in the alveoli, the pulmonary depression is often accompanied by a decrease in the transmittance of the affected area, and the adjacent structures (bronchial, pulmonary vascular, pulmonary interstitial) gather towards the atelectasis, sometimes with alveolar cavity consolidation and compensatory emphysema in other lung tissues. The flow of collateral gas between the lobules and segments of the lungs (occasionally the lobes of the lungs) can still give a degree of light to the completely blocked area. '
            category = 'Atelectasis'
            self.another_window(Atelectasis_img, category, name, pic)

        if Cardiomegaly_img != 0:
            # for i in os.listdir('./Cardiomegaly'):
            self.merge_into_one_pic('./Cardiomegaly/')
            pic = cv2.imread('./Cardiomegaly/'+ 'result.jpg')
            name = 'Cardiomegaly：Ventricular hypertrophy (ventricular hypertrophy) is caused by overload of the ventricles (diastolic or systolic). Including ventricular hypertrophy and enlargement, the patients with stress overload are mainly ventricular hypertrophy, and those with volume overload are mainly ventricular enlargement. ' \
                   'The atrium wall is thin, whichever kind of excessive load is generally shown as enlargement. Atrial or ventricular hypertrophy is a common consequence of organic heart disease and can be shown in ECG when reaching a certain degree. To treat ventricular hypertrophy, we need to find out the underlying causes and treat different causes. '
            category = 'Cardiomegaly'
            self.another_window(Cardiomegaly_img, category, name, pic)

        if Effusion_img != 0:
            # for i in os.listdir('./Effusion'):
            self.merge_into_one_pic('./Effusion/')
            pic = cv2.imread('./Effusion/' + 'result.jpg')
            name = 'Effusion：Pulmonary hydrocephalus is commonly referred to as "pleural hydronephrosis" in medicine. Water is accumulated outside the lungs. It can be caused by inflammation of infection (e.g. pneumonia, tuberculosis... can be associated with pleural hydronephrosis), or some autoimmune diseases (e.g. lupus erythematosus), and many pulmonary diseases can be associated with pleural hydronephrosis. '
            category = 'Effusion'
            self.another_window(Effusion_img, category, name, pic)

        if Infiltrate_img != 0:
            # for i in os.listdir('./Infiltrate'):
            self.merge_into_one_pic('./Infiltrate/')
            pic = cv2.imread('./Infiltrate/'+ 'result.jpg')
            name = 'Infiltrate:This type is more common in adults, the lesions are mostly in the upper and lower collarbone, with flaky or flocculent shape, the boundary is blurred, the lesions may be cheese-like necrotic focus, causing more serious toxic symptoms, but cheese-like (tuberculous) pneumonia, ' \
                   'necrotic focus wrapped in fiber to form tuberculous ball.'
            category = 'Infiltrate'
            self.another_window(Infiltrate_img, category , name, pic)

        if Mass_img != 0:
            # for i in os.listdir('./Mass'):
            #     pic = cv2.imread('./Infiltrate/'+ i)
            self.merge_into_one_pic('./Mass/')
            pic = cv2.imread('./Mass/'+ 'result.jpg')
            name = 'Mass: Lung atrophy refers to a disease of pulmonary ventilation disorder caused by many causes and must be treated immediately. '
            category = 'Mass'
            self.another_window(Mass_img, category, name, pic)

        if Nodule_img != 0:
            # for i in os.listdir('./Nodule'):
            self.merge_into_one_pic('./Nodule/')
            pic = cv2.imread('./Nodule/'+ 'result.jpg')
            name = 'Nodule:Pulmonary tuberculosis (sarcoidosis) is a multi-system multi-organ granulomatous disease with unknown etiology, which often invades lung, ' \
                   'bilateral hilum lymph nodes, eyes, skin and other organs. '
            category = 'Nodule'
            self.another_window(Nodule_img, category, name, pic)

        if Pneumothorax_img != 0:
            # for i in os.listdir('./Cardiomegaly'):
            self.merge_into_one_pic('./Pneumothorax/')
            pic = cv2.imread('./Pneumothorax/' + 'result.jpg')
            name = 'Pneumothorax：Pneumothorax refers to the gas entering the pleural cavity, resulting in gas accumulation state, called pneumothorax. pulmonary tissue and visceral pleura ruptured due to pulmonary disease or external force effects, or microswelling vesicles near the surface of the lung ruptured, ' \
                   'and air in the lungs and bronchi escaped into the pleural cavity. '
            category = 'Pneumothorax'
            self.another_window(Cardiomegaly_img, category, name, pic)

        if Pneumonia_img != 0:
            # for i in os.listdir('./Cardiomegaly'):
            self.merge_into_one_pic('./Pneumonia/')
            pic = cv2.imread('./Pneumonia/' + 'result.jpg')
            name = 'Pneumonia：Senile pneumonia often lacks obvious respiratory symptoms, symptoms are more atypical, the disease progress quickly, prone to missed diagnosis, wrong diagnosis. The first symptoms are shortness of breath and dyspnea, ' \
                   'or conscious disorders, lethargy, dehydration, loss of appetite, etc. '
            category = 'Pneumonia'
            self.another_window(Cardiomegaly_img, category, name, pic)
        # self.another_window()
        # self.another_window()
        mainloop()

        # recognition_result = Toplevel()
        # recognition_result.title('海藻自动分析与定位识别结果')
        #
        # # IMG = tk.Text(recognition_result)
        # # IMG.grid(row=1, column=2)
        #
        # image = tk.Frame(recognition_result)
        # image.grid(columnspan=2, sticky=W)
        #
        #
        # img_label = tk.Label(image)
        # img_label.grid(columnspan=2, sticky=W)
        #
        # category = tk.Label(recognition_result, text='微藻种类: ', font=60)
        # category.grid(row=2, column=0)
        #
        # entry1= tk.Text(recognition_result, width =26, height=1)
        # entry1.grid(row=2, column=1, sticky=W)
        #
        # category_name = tk.Label(recognition_result, text='数量: ', font=60)
        # category_name.grid(row=3, column=0)
        #
        # entry2 = tk.Text(recognition_result, width=26, height=1)
        # entry2.grid(row=3, column=1, sticky=W)
        #
        # category_name = tk.Label(recognition_result, text='简介: ', font=60)
        # category_name.grid(row=4, column=0)
        #
        # entry2 = tk.Text(recognition_result, width=26, height=8)
        # entry2.grid(row=4, column=1, sticky=W)


        # my_photo = PhotoImage(file="result.png")
        # IMG.image_create(END, image=my_photo)


        # self.Atelectasis_num = 0
        # self.Cardiomegaly_num = 0
        # self.Effusion_num = 0
        # self.Infiltrate_num = 0
        # self.Mass_num = 0
        # self.Nodule_num  = 0



        # B, G, R = cv2.split(second_frame)
        # second_frame = cv2.merge([R, G, B])
        # img = Image.fromarray(second_frame)  # 类型是PIL.Image.Image
        # img = self.resize(img)
        # imgtk = ImageTk.PhotoImage(img)
        # img_label.config(image=imgtk)
        # img_label.img = imgtk
        # mainloop()

    def another_window(self,count,cat,introduction,pic):
        recognition_result = Toplevel()
        recognition_result.title('Automatic Analysis and Mapping of Lung Diseases')

        # canvas = tk.Canvas(recognition_result, bg='white',height=768,width=1024)
        # for i in pic:
        #     image_file = tk.PhotoImage(file=i)
        #     canvas.create_image()

        image = tk.Frame(recognition_result)
        image.grid(columnspan=2, sticky=W)

        img_label = tk.Label(image)
        img_label.grid(columnspan=2, sticky=W)

        category = tk.Label(recognition_result, text='Type of disease : ', font=60)
        category.grid(row=2, column=0)

        entry1 = tk.Text(recognition_result, width=26, height=1)
        entry1.grid(row=2, column=1, sticky=W)


        category_name = tk.Label(recognition_result, text='number: ', font=60)
        category_name.grid(row=3, column=0)

        entry2 = tk.Text(recognition_result, width=26, height=1)
        entry2.grid(row=3, column=1, sticky=W)


        category_name = tk.Label(recognition_result, text='brief introduction: ', font=60)
        category_name.grid(row=4, column=0)

        entry3 = tk.Text(recognition_result, width=26, height=8)
        entry3.grid(row=4, column=1, sticky=W)
        entry1.insert(INSERT, cat)
        entry2.insert(INSERT, str(count))
        entry3.insert(INSERT,introduction)

        # for i in pic:
        second_frame = pic # ndarray
        B, G, R = cv2.split(second_frame)
        second_frame = cv2.merge([R, G, B])
        img = Image.fromarray(second_frame)  # 类型是PIL.Image.Image
        img = self.resize(img)
        imgtk = ImageTk.PhotoImage(img)
        img_label.config(image=imgtk)
        img_label.img = imgtk

    def text_insert(self):
        pass

    def merge_into_one_pic(self,path):
        dir_imgs = []
        count = len(os.listdir(path))
        for i in os.listdir(path):
            dir_img = cv2.imread(path + '/' + i)
            B, G, R = cv2.split(dir_img)
            dir_img = cv2.merge([R, G, B])
            dir_imgs.append(dir_img)

        for i in range(len(dir_imgs)):
            geshu = len(dir_imgs)
            frame = dir_imgs[i]
            plt.subplot(1, geshu, i + 1)
            plt.imshow(frame)
        plt.savefig(path + 'result.jpg')
        # plt.show()

    # @staticmethod
    # def thread_it(func, *args):
    #     t = threading.Thread(target=func, args=args)
    #     t.setDaemon(True)
    #     t.start()
if __name__ == '__main__':
    goods_counting_gui()
    mainloop()
