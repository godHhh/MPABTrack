from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import copy
import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from trackers.MPABSort_tracker.mice_tracker import MPABSort
from trackers.MPABSort_tracker.mice_tracker_reid import MPABSort_ReID

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
from utils.utils import write_results, write_results_no_score, write_results_no_score_have_class


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
            self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args


    def evaluate_mice(
            self,
            model,
            distributed=False,
            half=False,
            trt_file=None,
            decoder=None,
            test_size=None,
            result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = MPABSort(det_thresh=self.args.track_thresh, iou_threshold=self.args.iou_thresh,
                         asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia,
                         use_second=self.args.use_second)

        detections = dict()

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
                progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if frame_id == 1:
                    tracker = MPABSort(det_thresh=self.args.track_thresh, iou_threshold=self.args.iou_thresh,
                                     asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia,
                                     use_second=self.args.use_second)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                ckt_file = "dance_detections1/{}_detetcion.pkl".format(video_name)
                if os.path.exists(ckt_file):
                    # outputs = [torch.load(ckt_file)]
                    if not video_name in detections:
                        dets = torch.load(ckt_file)
                        detections[video_name] = dets

                    all_dets = detections[video_name]
                    outputs = [all_dets[all_dets[:, 0] == frame_id][:, 1:]]
                else:
                    imgs = imgs.type(tensor_type)

                    # skip the the last iters since batchsize might be not enough for batch inference

                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                    # we should save the detections here !
                    # os.makedirs("dance_detections/{}".format(video_name), exist_ok=True)
                    # torch.save(outputs[0], ckt_file)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_mice_reid(
            self,
            args,
            model,
            distributed=False,
            half=False,
            trt_file=None,
            decoder=None,
            test_size=None,
            result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        # for reid
        self.encoder = reidInterface(self.args.MPBReID_config, self.args.MPBReID_weights, 'cuda')

        detections = dict()

        # for cur_iter, (imgs, target, info_imgs, ids, raw_image) in enumerate(
        for cur_iter, (imgs, _, info_imgs, ids, raw_image) in enumerate(
                progress_bar(self.dataloader)
        ):
            raw_image = raw_image.numpy()[0, ...]
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if frame_id == 1:
                    tracker = MPABSort_ReID(args, det_thresh=self.args.track_thresh, iou_threshold=self.args.iou_thresh,
                                          asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia,
                                          use_second=self.args.use_second)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        # write_results_no_score_have_class(result_filename, results)
                        results = []

                ckt_file = "dance_detections1/{}_detetcion.pkl".format(video_name)
                if os.path.exists(ckt_file):
                    # outputs = [torch.load(ckt_file)]
                    if not video_name in detections:
                        dets = torch.load(ckt_file)
                        detections[video_name] = dets

                    all_dets = detections[video_name]
                    outputs = [all_dets[all_dets[:, 0] == frame_id][:, 1:]]
                else:
                    imgs = imgs.type(tensor_type)

                    # skip the the last iters since batchsize might be not enough for batch inference
                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

                    # pic_number = img_file_name[0].split('/')[-1].split('.')[0]
                    # if pic_number.__eq__('0666') or pic_number.__eq__('0667') or pic_number.__eq__('0668') or pic_number.__eq__('0669') or pic_number.__eq__('0670') or pic_number.__eq__('0671'):
                    #     outputs_666 = outputs
                    if outputs[0] == None:
                        id_feature = np.array([]).reshape(0, 2048)
                    else:
                        bbox_xyxy = copy.deepcopy(outputs[0][:, :4])
                        # we should save the detections here !
                        # os.makedirs("dance_detections/{}".format(video_name), exist_ok=True)
                        # torch.save(outputs[0], ckt_file)
                        scale = min(self.img_size[0] / float(info_imgs[0]), self.img_size[1] / float(info_imgs[1]))
                        bbox_xyxy /= scale
                        id_feature = self.encoder.inference(raw_image,
                                                            bbox_xyxy.cpu().detach().numpy(),img_file_name)  # normalization and numpy included
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            if self.args.ECC:
                # compute warp matrix with ECC, when frame_id is not 1.
                # raw_image = raw_image.numpy()[0, ...]       # sequeeze batch dim, [bs, H, W, C] ==> [H, W, C]
                if frame_id != 1:
                    warp_matrix, src_aligned = self.ECC(self.former_frame, raw_image, align=True)
                else:
                    warp_matrix, src_aligned = None, None
                self.former_frame = raw_image  # update former_frame
            else:
                warp_matrix, src_aligned = None, None

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, id_feature=id_feature,
                                            warp_matrix=warp_matrix)
            online_tlwhs = []
            online_ids = []
            online_class = []
            # class_labels = target[0,:,4]
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                # tclass = class_labels[int(tid)-1]
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    # online_class.append(tclass.item())
            # save results
            # results.append((frame_id, online_tlwhs, online_ids,online_class))
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                # write_results_no_score_have_class(result_filename, results)
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
                outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                ["forward", "track", "inference"],
                [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
            )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
