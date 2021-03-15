
import numpy as np
from sklearn.metrics import precision_recall_curve,average_precision_score,precision_score,recall_score
class SegmentationMetric():

    def __init__(self,as_binary=False,class_idx=1,as_mean=True,class_name=None):
        super(SegmentationMetric,self).__init__()
        if isinstance(class_idx,int):
            self.class_idx=[class_idx]
        else:
            self.class_idx=class_idx

        self.a11=np.zeros([len(self.class_idx)])
        self.a12 = np.zeros([len(self.class_idx)])
        self.a21=np.zeros([len(self.class_idx)])
        self.a22=np.zeros([len(self.class_idx)])
        self.as_bianry=as_binary
        self.as_mean=as_mean

    def forward(self,pred_img,label_img):
        if not self.as_bianry:
            pred_img=np.argmax(pred_img,dim=1)
        for idx,i in enumerate(self.class_idx):
            a11_mask = (pred_img!=i) & (label_img !=i)
            a12_mask = (pred_img == i) & (label_img !=i)
            a21_mask = (pred_img !=i) & (label_img == i)
            a22_mask = (pred_img == i) & (label_img == i)

            self.a11[idx] = self.a11[idx] + np.sum(a11_mask)
            self.a12[idx] = self.a12[idx] + np.sum(a12_mask)
            self.a21[idx] = self.a21[idx] + np.sum(a21_mask)
            self.a22[idx] = self.a22[idx] + np.sum(a22_mask)

    def get_metric(self):
        self.a22=self.a22.astype(np.float)
        iou = self.a22 / (self.a12 + self.a21 + self.a22+1e-6)
        precision = self.a22 / (self.a12 + self.a22+1e-6)
        recall = self.a22 / (self.a21 + self.a22+1e-6)
        F1 = 2 * precision * recall / (precision + recall+1e-6)
        acc = (self.a11 + self.a22) / (self.a11 + self.a12 + self.a21 + self.a22+1e-6)
        if self.as_mean:
            iou=np.mean(iou)
            precision=np.mean(precision)
            recall=np.mean(recall)
            F1=np.mean(F1)
            acc=np.mean(acc)
            return iou,precision,recall,F1,acc
        else:
            return iou,precision,recall,F1,acc


    def reset(self):
        self.a11 = np.zeros([len(self.class_idx)])
        self.a12 = np.zeros([len(self.class_idx)])
        self.a21 = np.zeros([len(self.class_idx)])
        self.a22 = np.zeros([len(self.class_idx)])


class PrcMetric():
    def __init__(self):
        super(PrcMetric,self).__init__()
        self.pred_list=[]
        self.label_list=[]


    def forward(self,pred_img:np.ndarray,label_img:np.ndarray):
        pred_img=np.mean(pred_img,axis=1)
        label_img=label_img/255.
        label_img=label_img.astype(np.int)
        self.pred_list.extend(np.reshape(pred_img,[-1,]))
        self.label_list.extend(np.reshape(label_img,[-1,]))

    def get_metric(self):
        prc=average_precision_score(self.label_list,self.pred_list)
        return prc


    def reset(self):
        self.pred_list=[]
        self.label_list=[]