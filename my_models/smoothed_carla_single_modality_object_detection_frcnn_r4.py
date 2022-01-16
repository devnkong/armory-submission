"""
PyTorch Faster-RCNN Resnet50-FPN object detection model
"""
import logging
from typing import Optional

from art.estimators.object_detection import PyTorchFasterRCNN
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DetectionsAcc:
    OBJECT_SORT=0
    CENTER_SORT=1
    CORNER_SORT=2
    SINGLE_BIN=0
    LABEL_BIN=1
    LOCATION_BIN=2
    LOCATION_LABEL_BIN=3

    def __init__(self, bin=SINGLE_BIN, sort=OBJECT_SORT, loc_bin_count=None):
        self.detections_list = []
        self.max_num_detections = 0
        #count the number of classes in each class bin
        self.bin_counts = {}
        self.detections_tensor = None
        self.id_index_map = {}

        self.sort = sort
        self.bin = bin
        self.loc_bin_count = loc_bin_count
    def track(self, detections):
        #dim of detections (# of simulations, tensor((#of detections, 7)))
        self.detections_list.extend(detections)
        for detection in detections:
            if detection is not None:

                temp_count = {}
                if self.bin == DetectionsAcc.SINGLE_BIN:
                    box_count = detection.size(0)
                    if box_count > self.max_num_detections:
                        self.max_num_detections = box_count
                elif (self.bin ==  DetectionsAcc.LABEL_BIN
                      or self.bin == DetectionsAcc.LOCATION_BIN
                      or self.bin == DetectionsAcc.LOCATION_LABEL_BIN):
                    if self.bin == DetectionsAcc.LABEL_BIN:
                        # for label binning
                        ids = detection[:, -1].tolist()
                    elif self.bin == DetectionsAcc.LOCATION_BIN:
                        # for location binning
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx/416*self.loc_bin_count).floor()
                        yids = (midy/416*self.loc_bin_count).floor()
                        ids = (xids+yids*10).tolist()
                    elif self.bin == DetectionsAcc.LOCATION_LABEL_BIN:
                        # for location+label binning
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx / 416 * self.loc_bin_count).floor()
                        yids = (midy / 416 * self.loc_bin_count).floor()
                        labels = detection[:, -1]
                        ids = (xids + yids * 10 + labels * 100).tolist()

                    for id in ids:
                        if id not in temp_count:
                            temp_count[id] = 1
                        else:
                            temp_count[id] += 1
                    for id, count in temp_count.items():
                        if id not in self.bin_counts:
                            self.bin_counts[id] = count
                        elif self.bin_counts[id] < count:
                            self.bin_counts[id] = count

    def tensorize(self):
        if self.bin == DetectionsAcc.SINGLE_BIN:
            self.detection_len = self.max_num_detections
        elif (self.bin == DetectionsAcc.LABEL_BIN or
                self.bin == DetectionsAcc.LOCATION_BIN or
                self.bin == DetectionsAcc.LOCATION_LABEL_BIN):
            self.detection_len = 0
            for id, count in self.bin_counts.items():
                self.id_index_map[id] = self.detection_len
                self.detection_len += count
        else:
            raise ValueError("Invalid bin parameter")


        self.detections_tensor = torch.ones(
            (len(self.detections_list), self.detection_len, 7)
        )*float('inf')
        # self.detections_tensor[0:len(self.detections_list)//2] *= -1
        for i, detection in enumerate(self.detections_list):
            if detection is not None:
                if self.sort == DetectionsAcc.OBJECT_SORT:
                    detection_count = detection.size(0)
                elif self.sort == DetectionsAcc.CENTER_SORT:
                    detection_count = detection.size(0)
                    midy = (detection[:, 1]+detection[:, 3])/2
                    _, sort_idx = midy.sort(dim=0)
                    detection = detection[sort_idx]
                    midx = (detection[:, 0]+detection[:, 2])/2
                    _, sort_idx = midx.sort(dim=0)
                    detection = detection[sort_idx]

                if self.bin == DetectionsAcc.SINGLE_BIN:
                    self.detections_tensor[i, 0:detection_count] = detection
                elif (self.bin == DetectionsAcc.LABEL_BIN or
                        self.bin == DetectionsAcc.LOCATION_BIN or
                        self.bin == DetectionsAcc.LOCATION_LABEL_BIN):
                    if self.bin == DetectionsAcc.LABEL_BIN:
                        ids = detection[:, -1]
                        unique_ids = detection[:, -1].unique()
                    elif self.bin == DetectionsAcc.LOCATION_BIN:
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx / 416 * self.loc_bin_count).floor()
                        yids = (midy / 416 * self.loc_bin_count).floor()
                        ids = xids + yids * 10
                        unique_ids = ids.unique()
                    elif self.bin == DetectionsAcc.LOCATION_LABEL_BIN:
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx / 416 * self.loc_bin_count).floor()
                        yids = (midy / 416 * self.loc_bin_count).floor()
                        labels = detection[:, -1]
                        ids = xids + yids * 10 + labels * 100
                        unique_ids = ids.unique()

                    for id in unique_ids:
                        filtered_detection = detection[ids == id]
                        filtered_len = filtered_detection.size(0)
                        idx_st = self.id_index_map[id.cpu().item()]
                        self.detections_tensor[i, idx_st:idx_st+filtered_len]= filtered_detection





        self.detections_tensor, _ = self.detections_tensor.sort(dim=0)
    def median(self):
        result = self.detections_tensor[len(self.detections_list) // 2]
        return result
    def upper(self, alpha=.05):
        result = self.detections_tensor[int(len(self.detections_list)*(alpha))]
        return result
    def lower(self, alpha=.05):
        result = self.detections_tensor[int(len(self.detections_list)*(1-alpha))]
        return result
    def k(self, q):
        result = self.detections_tensor[q]
        return result
    def clear(self):
        self.detections_list = []
        self.max_num_detections = 0
        self.detections_tensor = None

class SmoothMedianNMS(nn.Module):
    def __init__(self, base_detector, sigma, accumulator):
        super(SmoothMedianNMS, self).__init__()
        self.base_detector = base_detector
        self.sigma = sigma
        self.detection_acc = accumulator

    def predict_range(self, x: torch.tensor, n: int, batch_size: int, q_u: int, q_l: int) -> (torch.tensor, torch.tensor, torch.tensor):

        input_imgs = x.repeat((batch_size, 1, 1, 1))
        for i in range(n//batch_size):
            # Get detections
            with torch.no_grad():
                detections = self.base_detector(input_imgs + torch.randn_like(input_imgs) * self.sigma)
                # detections, _ = non_max_suppression(detections, conf_thres, nms_thres)
                self.detection_acc.track(detections)

        self.detection_acc.tensorize()
        detections = [self.detection_acc.median()]
        detections_l = [self.detection_acc.k(q_l)]
        detections_u = [self.detection_acc.k(q_u)]
        self.detection_acc.clear()
        return detections, detections_u, detections_l

    def forward(self, x, n=2000, batch_size=20) :

        # # x = torch.tensor(x)
        # input_imgs = x.repeat((batch_size, 1, 1, 1))
        # for i in range(n//batch_size):
        #     # Get detections
        #     with torch.no_grad():
        #         detections = self.base_detector(input_imgs + torch.randn_like(input_imgs) * self.sigma)
        #         # detections, _ = non_max_suppression(detections, conf_thres, nms_thres)
        #         self.detection_acc.track(detections)

        # self.detection_acc.tensorize()
        # detections = [self.detection_acc.median()]
        # self.detection_acc.clear()
        # return detections

        return self.base_detector(x)

# NOTE: PyTorchFasterRCNN expects numpy input, not torch.Tensor input
def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchFasterRCNN:

    if weights_path:
        assert model_kwargs.get("num_classes", None) == 4, (
            "model trained on CARLA data outputs predictions for 4 classes, "
            "set model_kwargs['num_classes'] to 4."
        )
        assert not model_kwargs.get("pretrained", False), (
            "model trained on CARLA data should not use COCO-pretrained weights, set "
            "model_kwargs['pretrained'] to False."
        )

    model = models.detection.fasterrcnn_resnet50_fpn(**model_kwargs)
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    # q_u, q_l = estimated_qu_ql(opt.eps, opt.smooth_count, opt.sigma, conf_thres=opt.cert_conf)
    accumulator = DetectionsAcc(bin=3, sort = 1, loc_bin_count=3)
    model = SmoothMedianNMS(model, 0.25, accumulator)

    wrapped_model = PyTorchFasterRCNN(
        model, clip_values=(0.0, 1.0), channels_first=False, **wrapper_kwargs,
    )
    return wrapped_model