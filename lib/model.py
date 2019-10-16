import os
import datetime
import logging
import time
import numpy as np
import torch
import cv2
from typing import List
from lib.scene_parser.rcnn.structures.bounding_box import BoxList
from lib.scene_parser.rcnn.structures.bounding_box_pair import BoxPairList
from .data.build import build_data_loader
from .scene_parser.parser import build_scene_parser
from .scene_parser.parser import build_scene_parser_optimizer
from .scene_parser.rcnn.utils.metric_logger import MetricLogger
from .scene_parser.rcnn.utils.timer import Timer, get_time_str
from .scene_parser.rcnn.utils.comm import synchronize, all_gather, is_main_process, get_world_size
from .scene_parser.rcnn.utils.visualize import select_top_predictions, select_top_relations, overlay_boxes, overlay_class_names
from .data.evaluation import evaluate, evaluate_sg
from .utils.box import bbox_overlaps

class SceneGraphGeneration:
    """
    Scene graph generation
    """
    def __init__(self, cfg, arguments, local_rank, distributed):
        """
        initialize scene graph generation model
        """
        self.cfg = cfg
        self.arguments = arguments.copy()
        self.device = torch.device("cuda")

        # build data loader
        if cfg.inference:
            self.data_loader_test = build_data_loader(
                cfg, split="test", is_distributed=distributed)
        else:
            self.data_loader_train = build_data_loader(
                    cfg, split="train", is_distributed=distributed)
            self.data_loader_test = build_data_loader(
                    cfg, split="test", is_distributed=distributed)

        cfg.DATASET.IND_TO_OBJECT = self.data_loader_test.dataset.ind_to_classes
        cfg.DATASET.IND_TO_PREDICATE = self.data_loader_test.dataset.ind_to_predicates

        logger = logging.getLogger("scene_graph_generation.trainer")
        if not cfg.inference:
            logger.info("Train data size: {}".format(
                    len(self.data_loader_train.dataset)))
        logger.info("Test data size: {}".format(
                    len(self.data_loader_test.dataset)))

        if not os.path.exists("freq_prior.npy"):
            logger.info("Computing frequency prior matrix...")
            fg_matrix, bg_matrix = self._get_freq_prior()
            prob_matrix = fg_matrix.astype(np.float32)
            prob_matrix[:,:,0] = bg_matrix

            prob_matrix[:,:,0] += 1
            prob_matrix /= np.sum(prob_matrix, 2)[:,:,None]
            # prob_matrix /= float(fg_matrix.max())
            np.save("freq_prior.npy", prob_matrix)

        # build scene graph generation model
        self.scene_parser = build_scene_parser(cfg); self.scene_parser.to(self.device)
        self.sp_optimizer, self.sp_scheduler, self.sp_checkpointer, self.extra_checkpoint_data = \
            build_scene_parser_optimizer(cfg, self.scene_parser, local_rank=local_rank, distributed=distributed)

        self.arguments.update(self.extra_checkpoint_data)

    def _get_freq_prior(self, must_overlap=False):

        fg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
            ), dtype=np.int64)

        bg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
        ), dtype=np.int64)

        for ex_ind in range(len(self.data_loader_train.dataset)):
            gt_classes = self.data_loader_train.dataset.gt_classes[ex_ind].copy()
            gt_relations = self.data_loader_train.dataset.relationships[ex_ind].copy()
            gt_boxes = self.data_loader_train.dataset.gt_boxes[ex_ind].copy()

            # For the foreground, we'll just look at everything
            o1o2 = gt_classes[gt_relations[:, :2]]
            for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
                fg_matrix[o1, o2, gtr] += 1

            # For the background, get all of the things that overlap.
            o1o2_total = gt_classes[np.array(
                self._box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1

            if ex_ind % 20 == 0:
                print("processing {}/{}".format(ex_ind, len(self.data_loader_train.dataset)))

        return fg_matrix, bg_matrix

    def _box_filter(self, boxes, must_overlap=False):
        """ Only include boxes that overlap as possible relations.
        If no overlapping boxes, use all of them."""
        n_cands = boxes.shape[0]

        overlaps = bbox_overlaps(torch.from_numpy(boxes.astype(np.float)), torch.from_numpy(boxes.astype(np.float))).numpy() > 0
        np.fill_diagonal(overlaps, 0)

        all_possib = np.ones_like(overlaps, dtype=np.bool)
        np.fill_diagonal(all_possib, 0)

        if must_overlap:
            possible_boxes = np.column_stack(np.where(overlaps))

            if possible_boxes.size == 0:
                possible_boxes = np.column_stack(np.where(all_possib))
        else:
            possible_boxes = np.column_stack(np.where(all_possib))
        return possible_boxes

    def train(self):
        """
        main body for training scene graph generation model
        """
        start_iter = self.arguments["iteration"]
        logger = logging.getLogger("scene_graph_generation.trainer")
        logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        max_iter = len(self.data_loader_train)
        self.scene_parser.train()
        start_training_time = time.time()
        end = time.time()
        for i, data in enumerate(self.data_loader_train, start_iter):
            data_time = time.time() - end
            self.arguments["iteration"] = i
            self.sp_scheduler.step()
            imgs, targets, _ = data
            imgs = imgs.to(self.device)
            targets = [target.to(self.device) for target in targets]
            loss_dict = self.scene_parser(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = loss_dict
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            self.sp_optimizer.zero_grad()
            losses.backward()
            self.sp_optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if i % 20 == 0 or i == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "model: {tag}",
                            "eta: {eta}",
                            "iter: {iter}/{max_iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        tag="scene_parser",
                        eta=eta_string,
                        iter=i, max_iter=max_iter,
                        meters=str(meters),
                        lr=self.sp_optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if (i + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                self.sp_checkpointer.save("checkpoint_{:07d}".format(i), **self.arguments)
            if (i + 1) == max_iter:
                self.sp_checkpointer.save("checkpoint_final", **self.arguments)

    def _accumulate_predictions_from_multiple_gpus(self, predictions_per_gpu):
        all_predictions = all_gather(predictions_per_gpu)
        if not is_main_process():
            return
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
        # convert a dict where the key is the index in a list
        image_ids = list(sorted(predictions.keys()))
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("scene_graph_generation.inference")
            logger.warning(
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the evaluation"
            )

        # convert to a list
        predictions = [predictions[i] for i in image_ids]
        return predictions

    def visualize_detection(self,
                            img: torch.Tensor,
                            det_img_fname: str,
                            det_txt_fname: str,
                            detection: BoxList
                            ):
        """
        Display the detected objects on the original image and save it. Also
        save the detected objects in text form together with their
        probabilities.

        :param img: Tensor with image
        :param det_img_fname: path to image where detected objects are overlayed
        :param det_txt_fname: path to text file where detected objects are saved
        : param detection: BoxList with detected objects
        :return:
        """
        #visualize_folder = "visualize"
        #if not os.path.exists(visualize_folder):
        #    os.mkdir(visualize_folder)
        predictions = detection
        idx_to_obj = self.data_loader_test.dataset.ind_to_classes

        for i, prediction in enumerate(predictions):
            all_scores = prediction.get_field("scores").tolist()
            all_labels = prediction.get_field("labels").tolist()

            top_prediction = select_top_predictions(prediction)

            # save top predicted objects and their probabilities
            scores = top_prediction.get_field("scores").tolist()
            labels = top_prediction.get_field("labels").tolist()

            with open(det_txt_fname, "w") as f:
                for label, score in zip(all_labels, all_scores):
                    label = idx_to_obj[label]
                    f.write(label + " " + str(score) + "\n")

            # w x h x c
            img = img.permute(1, 2, 0).contiguous().cpu().numpy() + \
                  np.array(self.cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3)
            # BGR -> RGB or the other way around
            img = img[:, :, [2, 1, 0]]

            result = img.copy()
            result = overlay_boxes(result, top_prediction)
            result = overlay_class_names(
                    result,
                    top_prediction,
                    idx_to_obj
            )
            cv2.imwrite(det_img_fname, result)

    def get_triplets_as_string(self,
                               top_obj: BoxList,
                               top_pred: BoxPairList) -> List[str]:
        """
        Given top detected objects and top predicted relationships, return
        the triplets in human-readable form.
        :param top_obj: BoxList containing top detected objects
        :param top_pred: BoxPairList containing the top detected triplets
        :return: List of triplets (in decreasing score order)
        """
        # num_detected_objects
        obj_indices = top_obj.get_field("labels")

        # 100 x 2 (indices in obj_indices)
        obj_pairs_indices = top_pred.get_field("idx_pairs")

        # 100 (indices in GLOBAL relationship indices list)
        rel_indices = top_pred.get_field("scores").max(1)[1]

        # 100 x 3
        top_triplets = torch.stack((
            obj_indices[obj_pairs_indices[:, 0]],
            obj_indices[obj_pairs_indices[:, 1]],
            rel_indices), 1).tolist()

        idx_to_obj = self.data_loader_test.dataset.ind_to_classes
        idx_to_rel = self.data_loader_test.dataset.ind_to_predicates

        # convert integers to labels
        top_triplets_str = []
        for t in top_triplets:
            top_triplets_str.append(idx_to_obj[t[0]] + " " +
                                    idx_to_rel[t[2]] + " " +
                                    idx_to_obj[t[1]])

        return top_triplets_str

    def predict(self, visualize=False):
        """
        Make predictions using pre-trained SG solution on unseen images from
        non-VG dataset (test but without the evaluation).
        :param visualize:
        :return:
        """
        logger = logging.getLogger("scene_graph_generation")
        logger.info("Start predicting")

        test_size = len(self.data_loader_test.dataset)
        data_dir = self.data_loader_test.dataset.data_dir
        self.scene_parser.eval()
        cpu_device = torch.device("cpu")
        results_dict = {}
        if self.cfg.MODEL.RELATION_ON:
            results_pred_dict = {}
        rel_embs_dict = {}

        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()

        reg_recalls = []
        for i, data in enumerate(self.data_loader_test, 0):
            imgs, boxes, objects, image_ids = data
            imgs = imgs.to(self.device)
            boxes = boxes.to(self.device)

            #print("[model.py:233] imgs = ", imgs.size())
            if i % 10 == 0:
                logger.info("prediction on batch {}/{}...".
                            format(i, len(self.data_loader_test)))

            with torch.no_grad():
                # predict relations between objects
                output = self.scene_parser(imgs)
                if self.cfg.MODEL.RELATION_ON:
                    output, output_pred, rel_embs = output
                    output_pred = [o.to(cpu_device) for o in output_pred]
                    output = [o.to(cpu_device) for o in output]

                # there are B examples in output (most likely B=1 at inference)
                # top_objs = [BoxList_1(num_boxes = 47),..., BoxList_B]
                # top_preds = [BoxPairList_1(num_boxes = 50), ... BoxPairList_B]
                # top_scores = [[50 top scores for example 1], ..., example B]
                # top_order = [[100 indices for example 1], ..., example B]
                top_objs, top_preds, top_scores, top_orders = \
                    self.scene_parser._post_processing((output,
                                                        output_pred
                                                        ))

                # save triplets (text + embedding) in dataset folder:
                # dataset/$movie_name/
                # 			          $img@0.detection.jpg - detected objects
                # 			          $img@0.detection.txt - top K detections
                # 			          $img@0.triplets.txt - top K triplets
                # 			          $img@0.triplets.pth - top K triplet embs
                for idx, (top_obj, top_pred, top_order) in enumerate(
                        zip(top_objs, top_preds, top_orders)):
                    fname = self.data_loader_test.dataset.get_img_fname(image_ids[idx])

                    sgg_txt_fname = fname.replace('.jpg', '.triplets.txt')
                    sgg_emb_fname = fname.replace('.jpg', '.triplets.pth')

                    # write top K triplets plus their probabilities
                    with open(sgg_txt_fname, 'w') as f:
                        top_triplets = self.get_triplets_as_string(top_obj,
                                                                   top_pred)
                        for triplet, score in zip(top_triplets,
                                                  top_scores[idx]):
                            f.write(triplet + " " + str(score) + "\n")

                    # write top K embeddings
                    top_rel_embs = rel_embs[top_order].to(cpu_device)
                    torch.save(top_rel_embs, sgg_emb_fname)

                # save detected objects and their probabilities
                if visualize:
                   fname = self.data_loader_test.dataset.get_img_fname(image_ids[0])
                   det_img_fname = fname.replace('.jpg', '.detection.jpg')
                   det_txt_fname = fname.replace('.jpg', '.detection.txt')
                   self.visualize_detection(imgs[0],
                                            det_img_fname,
                                            det_txt_fname,
                                            output)

            # {1: BoxList(), ... }
            # results_dict.update(
            #     {img_id: result for img_id, result in zip(image_ids, output)}
            # )
            # {1: T(#num_proposals, 2048)}
            # rel_embs_dict.update(
            #     {img_id: rel_embs for img_id in image_ids }
            # )
            # if self.cfg.MODEL.RELATION_ON:
            #     # {1: BoxPairList(), ... }
            #     results_pred_dict.update(
            #         {img_id: result for img_id, result in zip(image_ids,
            #                                                   output_pred)}
            #     )

            # if cfg.instance > 0, break after 1 batch?
            if self.cfg.instance > 0 and i > self.cfg.instance:
                break

        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        num_devices = get_world_size()

        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str,
                total_time * num_devices / test_size,
                num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img/device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / test_size,
                num_devices
            )
        )

        # predictions = self._accumulate_predictions_from_multiple_gpus(
        #                 results_dict)
        # relation_embeddings = self._accumulate_predictions_from_multiple_gpus(
        #             rel_embs_dict)
        # if self.cfg.MODEL.RELATION_ON:
        #     predictions_pred = self._accumulate_predictions_from_multiple_gpus(
        #                 results_pred_dict)
        if not is_main_process():
            return


    def test(self, timer=None, visualize=False):
        """
        main body for testing scene graph generation model
        """
        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start evaluating")
        self.scene_parser.eval()
        targets_dict = {}
        results_dict = {}
        if self.cfg.MODEL.RELATION_ON:
            results_pred_dict = {}
        cpu_device = torch.device("cpu")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        reg_recalls = []
        for i, data in enumerate(self.data_loader_test, 0):
            imgs, targets, image_ids = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]
            if i % 10 == 0:
                logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
            with torch.no_grad():
                if timer:
                    timer.tic()
                output = self.scene_parser(imgs)
                if self.cfg.MODEL.RELATION_ON:
                    output, output_pred = output
                    output_pred = [o.to(cpu_device) for o in output_pred]
                ious = bbox_overlaps(targets[0].bbox, output[0].bbox)
                reg_recall = (ious.max(1)[0] > 0.5).sum().item() / ious.shape[0]
                reg_recalls.append(reg_recall)
                if timer:
                    torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output]
                if visualize:
                    self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs, output)
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
            targets_dict.update(
                {img_id: target for img_id, target in zip(image_ids, targets)}
            )
            if self.cfg.MODEL.RELATION_ON:
                results_pred_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output_pred)}
                )
            if self.cfg.instance > 0 and i > self.cfg.instance:
                break
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        num_devices = get_world_size()
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(self.data_loader_test.dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(self.data_loader_test.dataset),
                num_devices,
            )
        )
        predictions = self._accumulate_predictions_from_multiple_gpus(results_dict)
        if self.cfg.MODEL.RELATION_ON:
            predictions_pred = self._accumulate_predictions_from_multiple_gpus(results_pred_dict)
        if not is_main_process():
            return

        output_folder = "results"
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
            if self.cfg.MODEL.RELATION_ON:
                torch.save(predictions_pred, os.path.join(output_folder, "predictions_pred.pth"))

        extra_args = dict(
            box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
            iou_types=("bbox",),
            expected_results=self.cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        )
        eval_det_results = evaluate(dataset=self.data_loader_test.dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        **extra_args)

        if self.cfg.MODEL.RELATION_ON:
            eval_sg_results = evaluate_sg(dataset=self.data_loader_test.dataset,
                            predictions=predictions,
                            predictions_pred=predictions_pred,
                            output_folder=output_folder,
                            **extra_args)

def build_model(cfg, arguments, local_rank, distributed):
    return SceneGraphGeneration(cfg, arguments, local_rank, distributed)
