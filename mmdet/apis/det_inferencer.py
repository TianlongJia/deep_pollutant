# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Union

import mmcv
import mmengine
import numpy as np
import torch.nn as nn
from mmengine.dataset import Compose
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from mmengine.visualization import Visualizer
from rich.progress import track

from mmdet.evaluation import INSTANCE_OFFSET
from mmdet.registry import DATASETS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import encode_mask_results, mask2bbox
from mmdet.utils import ConfigType
from ..evaluation import get_classes

# new added packages
import cv2
import pandas as pd
import torch
import utils.bbox_segm as bs

try:
    from panopticapi.evaluation import VOID
    from panopticapi.utils import id2rgb
except ImportError:
    id2rgb = None
    VOID = None

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = List[DetDataSample]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


class DetInferencer(BaseInferencer):
    """Object Detection Inferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "rtmdet-s" or 'rtmdet_s_8xb32-300e_coco' or
            "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py".
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to mmdet.
        palette (str): Color palette used for visualization. The order of
            priority is palette -> config -> checkpoint. Defaults to 'none'.
        show_progress (bool): Control whether to display the progress
            bar during the inference process. Defaults to True.
    """

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        'return_vis',
        'show',
        'wait_time',
        'draw_pred',
        'pred_score_thr',
        'img_out_dir',
        'no_save_vis',
    }
    postprocess_kwargs: set = {
        'print_result',
        'pred_out_dir',
        'return_datasample',
        'no_save_pred',
    }

    def __init__(self,
                 model: Optional[Union[ModelType, str]] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmdet',
                 palette: str = 'none',
                 show_progress: bool = True) -> None:
        # A global counter tracking the number of images processed, for
        # naming of the output images
        self.num_visualized_imgs = 0
        self.num_predicted_imgs = 0
        self.palette = palette
        init_default_scope(scope)
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)
        self.model = revert_sync_batchnorm(self.model)
        self.show_progress = show_progress
        

    def _load_weights_to_model(self, model: nn.Module,
                               checkpoint: Optional[dict],
                               cfg: Optional[ConfigType]) -> None:
        """Loading model weights and meta information from cfg and checkpoint.

        Args:
            model (nn.Module): Model to load weights and meta information.
            checkpoint (dict, optional): The loaded checkpoint.
            cfg (Config or ConfigDict, optional): The loaded config.
        """

        if checkpoint is not None:
            _load_checkpoint_to_model(model, checkpoint)
            checkpoint_meta = checkpoint.get('meta', {})
            # save the dataset_meta in the model for convenience
            if 'dataset_meta' in checkpoint_meta:
                # mmdet 3.x, all keys should be lowercase
                model.dataset_meta = {
                    k.lower(): v
                    for k, v in checkpoint_meta['dataset_meta'].items()
                }
            ############################################################################################################
                self.CLASSES = list({k.lower(): v for k, v in checkpoint_meta['dataset_meta'].items()}['classes'])
                print(self.CLASSES)
            ############################################################################################################
            elif 'CLASSES' in checkpoint_meta:
                # < mmdet 3.x
                classes = checkpoint_meta['CLASSES']
                model.dataset_meta = {'classes': classes}
            else:
                warnings.warn(
                    'dataset_meta or class names are not saved in the '
                    'checkpoint\'s meta data, use COCO classes by default.')
                model.dataset_meta = {'classes': get_classes('coco')}
        else:
            warnings.warn('Checkpoint is not loaded, and the inference '
                          'result is calculated by the randomly initialized '
                          'model!')
            warnings.warn('weights is None, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}

        # Priority:  args.palette -> config -> checkpoint
        if self.palette != 'none':
            model.dataset_meta['palette'] = self.palette
        else:
            test_dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
            # lazy init. We only need the metainfo.
            test_dataset_cfg['lazy_init'] = True
            metainfo = DATASETS.build(test_dataset_cfg).metainfo
            cfg_palette = metainfo.get('palette', None)
            if cfg_palette is not None:
                model.dataset_meta['palette'] = cfg_palette
            else:
                if 'palette' not in model.dataset_meta:
                    warnings.warn(
                        'palette does not exist, random is used by default. '
                        'You can also set the palette to customize.')
                    model.dataset_meta['palette'] = 'random'

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        # For inference, the key of ``img_id`` is not used.
        if 'meta_keys' in pipeline_cfg[-1]:
            pipeline_cfg[-1]['meta_keys'] = tuple(
                meta_key for meta_key in pipeline_cfg[-1]['meta_keys']
                if meta_key != 'img_id')

        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFile')
        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'mmdet.InferencerLoader'
        return Compose(pipeline_cfg)

    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] == name:
                return i
        return -1

    def _init_visualizer(self, cfg: ConfigType) -> Optional[Visualizer]:
        """Initialize visualizers.

        Args:
            cfg (ConfigType): Config containing the visualizer information.

        Returns:
            Visualizer or None: Visualizer initialized with config.
        """
        visualizer = super()._init_visualizer(cfg)
        visualizer.dataset_meta = self.model.dataset_meta
        return visualizer

    def _inputs_to_list(self, inputs: InputsType) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - str:
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (InputsType): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, str):
            backend = get_file_backend(inputs)
            if hasattr(backend, 'isdir') and isdir(inputs):
                # Backends like HttpsBackend do not implement `isdir`, so only
                # those backends that implement `isdir` could accept the inputs
                # as a directory
                filename_list = list_dir_or_file(
                    inputs, list_dir=False, suffix=IMG_EXTENSIONS)
                inputs = [
                    join_path(inputs, filename) for filename in filename_list
                ]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        
        # print("inputs: ", inputs)
        return list(inputs)

    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an iterable object, of which each item will be used as the
        input of ``model.test_step``.

        ``BaseInferencer.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        chunked_data = self._get_chunk_data(inputs, batch_size)
        yield from map(self.collate_fn, chunked_data)


    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from inputs.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    inputs_ = next(inputs_iter)
                    if isinstance(inputs_, dict):
                        if 'img' in inputs_:
                            ori_inputs_ = inputs_['img']
                        else:
                            ori_inputs_ = inputs_['img_path']
                        chunk_data.append(
                            (ori_inputs_,
                             self.pipeline(copy.deepcopy(inputs_))))
                    else:
                        chunk_data.append((inputs_, self.pipeline(inputs_)))
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    # TODO: Video and Webcam are currently not supported and
    #  may consume too much memory if your input folder has a lot of images.
    #  We will be optimized later.
    def __call__(
            self,
            inputs: InputsType,
            batch_size: int = 1,
            return_vis: bool = False,
            show: bool = False,
            wait_time: int = 0,
            no_save_vis: bool = False,
            draw_pred: bool = True,
            pred_score_thr: float = 0.3,
            return_datasample: bool = False,
            print_result: bool = False,
            no_save_pred: bool = True,
            out_dir: str = '',
            # by open image task
            texts: Optional[Union[str, list]] = None,
            # by open panoptic task
            stuff_texts: Optional[Union[str, list]] = None,
            # by GLIP
            custom_entities: bool = False,
            **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Inference batch size. Defaults to 1.
            show (bool): Whether to display the visualization results in a
                popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            no_save_vis (bool): Whether to force not to save prediction
                vis results. Defaults to False.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            return_datasample (bool): Whether to return results as
                :obj:`DetDataSample`. Defaults to False.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            no_save_pred (bool): Whether to force not to save prediction
                results. Defaults to True.
            out_file: Dir to save the inference results or
                visualization. If left as empty, no file will be saved.
                Defaults to ''.
            texts (str | list[str]): Text prompts. Defaults to None.
            stuff_texts (str | list[str]): Stuff text prompts of open
                panoptic task. Defaults to None.
            custom_entities (bool): Whether to use custom entities.
                Defaults to False. Only used in GLIP.
            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)
        #####################################################################################
        mask_name_list = []
        for mask_name in ori_inputs:
            print("reading image: ", mask_name)
            mask_name_list.append(mask_name.split('/')[-1].split('.')[0])
        
        # print("mask_name_list: ", mask_name_list)
        # print(ori_inputs.split('/')[1].split('.')[0])

        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)

        results_dict = {'predictions': [], 'visualization': []}
        tmp = 0

        all_img_name_list = []
        all_cls_name_list = []
        all_bbox_area_list = []
        all_scores_list = []
        all_mask_area_list = []
        all_xmin_list = []
        all_ymin_list = []
        all_xmax_list = []
        all_ymax_list = []

        for ori_inputs, data in track(inputs, description='Inference'):
            preds = self.forward(data, **forward_kwargs)
            visualization = self.visualize(
                ori_inputs,
                preds,
                return_vis=return_vis,
                show=show,
                wait_time=wait_time,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                no_save_vis=no_save_vis,
                img_out_dir=out_dir,
                **visualize_kwargs)
            # print("pred_score_thr: ", pred_score_thr)
            results, bbox_mask_inf_results = self.postprocess(
                                      preds,
                                      visualization,
                                      return_datasample=return_datasample,
                                      print_result=print_result,
                                      no_save_pred=no_save_pred,
                                      pred_out_dir=out_dir,
                                      mask_name=mask_name_list[tmp],
                                      pred_score_thr=pred_score_thr,
                                      **postprocess_kwargs)
            
            img_name_list = bbox_mask_inf_results[0]
            cls_name_list = bbox_mask_inf_results[1]
            bbox_area_list = bbox_mask_inf_results[2]
            scores_list = bbox_mask_inf_results[3]
            mask_area_list = bbox_mask_inf_results[4]
            xmin_list = bbox_mask_inf_results[5]
            ymin_list = bbox_mask_inf_results[6]
            xmax_list = bbox_mask_inf_results[7]
            ymax_list = bbox_mask_inf_results[8]
            

            for img_name in img_name_list:
                all_img_name_list.append(img_name)
            for cls_name in cls_name_list:
                all_cls_name_list.append(cls_name)
            for bbox_area in bbox_area_list:
                all_bbox_area_list.append(bbox_area)
            for scores in scores_list:
                all_scores_list.append(scores)
            for mask_area in mask_area_list:
                all_mask_area_list.append(mask_area)
            
            for xmin in xmin_list:
                all_xmin_list.append(xmin)
            for ymin in ymin_list:
                all_ymin_list.append(ymin)
            for xmax in xmax_list:
                all_xmax_list.append(xmax)
            for ymax in ymax_list:
                all_ymax_list.append(ymax)
            

            # all_img_name_list.append(img_name_list)
            
            results_dict['predictions'].extend(results['predictions'])
            if results['visualization'] is not None:
                results_dict['visualization'].extend(results['visualization'])
            tmp += 1

        # print("all_img_name_list: ", all_img_name_list)
        # print("all_cls_name_list: ", all_cls_name_list)
        # print("all_bbox_area_list: ", all_bbox_area_list)
        # print("all_scores_list: ", all_scores_list)
        # print("all_mask_area_list: ", all_mask_area_list)

        bs.save_bbox_info_in_excel(all_img_name_list, 
                                   all_xmin_list, 
                                   all_ymin_list, 
                                   all_xmax_list, 
                                   all_ymax_list, 
                                   all_scores_list, 
                                   all_cls_name_list, 
                                   all_bbox_area_list, 
                                   all_mask_area_list)

        return results_dict

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '',
                  **kwargs) -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[:obj:`DetDataSample`]): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            no_save_vis (bool): Whether to force not to save prediction
                vis results. Defaults to False.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if self.visualizer is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            out_file = osp.join(img_out_dir, 'vis',
                                img_name) if img_out_dir != '' else None

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                out_file=out_file,
            )
            results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results

    def postprocess(
        self,
        preds: PredType,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasample: bool = False,
        print_result: bool = False,
        no_save_pred: bool = False,
        pred_out_dir: str = '',
        mask_name: str = '',
        pred_score_thr: float = 0.3,
        **kwargs,
    ) -> Dict:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[:obj:`DetDataSample`]): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.
            return_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            no_save_pred (bool): Whether to force not to save prediction
                results. Defaults to False.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``.

            - ``visualization`` (Any): Returned by :meth:`visualize`.
            - ``predictions`` (dict or DataSample): Returned by
                :meth:`forward` and processed in :meth:`postprocess`.
                If ``return_datasample=False``, it usually should be a
                json-serializable dict containing only basic data elements such
                as strings and numbers.
        """
        if no_save_pred is True:
            pred_out_dir = ''

        result_dict = {}
        results = preds
        # print("#############")
        # print("Image name: ", mask_name)
        # print("#############")


        if not return_datasample:
            results = []
            # pred each image
            for pred in preds:
                result, bbox_mask_inf_results = self.pred2dict(pred, pred_out_dir, mask_name, pred_score_thr)
                results.append(result)

                
        
        elif pred_out_dir != '':
            warnings.warn('Currently does not support saving datasample '
                          'when return_datasample is set to True. '
                          'Prediction results are not saved!')
        # Add img to the results after printing and dumping
        result_dict['predictions'] = results
        if print_result:
            print(result_dict)
        result_dict['visualization'] = visualization

        # print("all_img_name_list: ", all_img_name_list)


        return result_dict, bbox_mask_inf_results

    # @ 将一张图中推理出的多个mask存在一张mask上，根据像素值来区分
    # masks -> mask 
    # input: labels = ['bed', 'chair', ...]、input_labels = [0,0,2,5,1,...]、masks = [mask1,mask2,...]
    # output: mask
    # def Save_masks(self, masks, labels, scores, output_dir, mask_name, conf_thres=0.5):
    def Save_masks(self, masks, labels, scores, conf_thres=0.5):
        # print(len(labels))
        # print(len(scores))
        # print(len(masks))
        
        index_dict =  {}
        print(len(self.CLASSES))
        # 筛选出高置信度的mask
        kept_inds = []
        for i in range(len(scores)):
            if scores[i]>conf_thres:
                kept_inds.append(i)
        print(kept_inds)
        # 设定每个mask合并后对应的像素值
        for i in range(len(self.CLASSES)):
            index_dict[self.CLASSES[i]] = i * 50  # 不同类别的像素值的间隔为10
        height = masks[0].shape[0]
        width = masks[0].shape[1]
        imgmask = np.zeros((height, width), np.uint16)  # 初始化待合并的mask
        # 遍历每个mask进行合并
        for idx in kept_inds:
            cls_name = self.CLASSES[labels[idx]]  # 获取当前mask对应的label
            print(cls_name, index_dict[cls_name])
            imgmask = np.where(masks[i].cpu().numpy()==True, index_dict[cls_name], imgmask)
            index_dict[cls_name] += 1    # 下一个同label的id自增1
        print(imgmask.shape)
        print(pd.Series(imgmask.flatten()).describe())
        cv2.imwrite('./outputs/preds/Amask.png', imgmask)



    #################################### Core code: to save mask (begin) #####################################
    
    def Save_masks_pics(self, masks, labels, scores, output_dir, mask_name, pred_score_thr):
        mask_area_list = []
        # cls_name_list = []
        # score_list = []

        index_dict =  {}
        # print("pred_score_thr: ", pred_score_thr)
        # print("Num of CLASSES: ", len(self.CLASSES))
        # 筛选出高置信度的mask
        kept_inds = []
        for i in range(len(scores)):
            if scores[i]>pred_score_thr:
                kept_inds.append(i)
                # score_list.append(scores[i])
        # print("kept_inds_mask: ", kept_inds)
        # 设定每个mask合并后对应的像素值
        for i in range(len(self.CLASSES)):
            index_dict[self.CLASSES[i]] = i * 50  # 不同类别的像素值的间隔为10
            index_dict[self.CLASSES[0]]= 20
        height = masks[0].shape[0]
        width = masks[0].shape[1]
        imgmask = np.zeros((height, width), np.uint16)  # 初始化待合并的mask
        # review each mask, and output final mask
        for idx in kept_inds:
            cls_name = self.CLASSES[labels[idx]]  # 获取当前mask对应的label
            # print(cls_name, index_dict[cls_name])
            
            # output mask area for each mask
            mask_area = torch.sum(masks[idx]).cpu().numpy()
            mask_area= int(mask_area)
            # print("mask_area: ", mask_area)

            mask_area_list.append(mask_area)
            # cls_name_list.append(cls_name)            
            imgmask = np.where(masks[idx].cpu().numpy()==True, index_dict[cls_name], imgmask)
            index_dict[cls_name] += 1    # 下一个同label的id自增1


        # print("imgmask.shape: ", imgmask.shape)
        # print(pd.Series(imgmask.flatten()).describe())

        # save mask files (.png)
        # cv2.imwrite(output_dir+'/masks/'+mask_name+'.png', imgmask)

        # print("mask_cls_name_list: ", cls_name_list)
        # print("mask_score_list: ", score_list)
        # print("mask_area_list: ", mask_area_list)

        return mask_area_list
    
     #################################### Core code: to save mask (end) #####################################





    # TODO: The data format and fields saved in json need further discussion.
    #  Maybe should include model name, timestamp, filename, image info etc.
    def pred2dict(self,
                  data_sample: DetDataSample,
                  pred_out_dir: str = '',
                  mask_name:str = '',
                  pred_score_thr:float=0.3
                  ) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Prediction results.
        """
        # mask_name_list=mask_name
        # print("####### mask_name: ", mask_name)
        is_save_pred = True
        if pred_out_dir == '':
            is_save_pred = False

        if is_save_pred and 'img_path' in data_sample:
            img_path = osp.basename(data_sample.img_path)
            img_path = osp.splitext(img_path)[0]
            out_img_path = osp.join(pred_out_dir, 'preds',
                                    img_path + '_panoptic_seg.png')
            out_json_path = osp.join(pred_out_dir, 'preds', img_path + '.json')
        elif is_save_pred:
            out_img_path = osp.join(
                pred_out_dir, 'preds',
                f'{self.num_predicted_imgs}_panoptic_seg.png')
            out_json_path = osp.join(pred_out_dir, 'preds',
                                     f'{self.num_predicted_imgs}.json')
            self.num_predicted_imgs += 1

        result = {}
        bbox_area_list = []
        cls_name_list = []
        scores_list = []
        img_name_list = []
        xmin_list = []
        ymin_list = []
        xmax_list = []
        ymax_list = []

        if 'pred_instances' in data_sample:
            masks = data_sample.pred_instances.get('masks')
            pred_instances = data_sample.pred_instances.numpy()
            result = {
                'bboxes': pred_instances.bboxes.tolist(),
                'labels': pred_instances.labels.tolist(),
                'scores': pred_instances.scores.tolist()
            }

            bboxes = pred_instances.bboxes.tolist()
            labels = pred_instances.labels.tolist()
            scores = pred_instances.scores.tolist()
            
            # obtain the bbox id with "score" above "pred_score_thr" 
            kept_inds = []
            for i in range(len(scores)):
              if scores[i] > pred_score_thr:
                  kept_inds.append(i)
                  scores_list.append(scores[i])
            # print("kept_inds_bbox: ", kept_inds)
            
            # Review and output final bboxes with "score" above "pred_score_thr" 
            for idx in kept_inds:
              cls_name = self.CLASSES[labels[idx]]  # obtain the bbox label
              bbox = bboxes[idx] # obtain the bbox location
              xmin = bbox[0]
              ymin = bbox[1]
              xmax = bbox[2]
              ymax = bbox[3]
              area_bbox = (xmax-xmin)*(ymax-ymin)

              xmin_list.append(xmin)
              ymin_list.append(ymin)
              xmax_list.append(xmax)
              ymax_list.append(ymax)
              bbox_area_list.append(area_bbox)
              cls_name_list.append(cls_name)
              img_name_list.append(mask_name)
            
            #####################################################################################################
            if masks is not None:
                # self.Save_masks(masks, labels, scores)
                mask_area_list = self.Save_masks_pics(masks, labels, scores, pred_out_dir, mask_name, pred_score_thr)
                
                if pred_instances.bboxes.sum() == 0:
                    # Fake bbox, such as the SOLO.
                    bboxes = mask2bbox(masks.cpu()).numpy().tolist()
                    result['bboxes'] = bboxes
                # print("masks.shape: ", masks.shape)
                # print("pred_instances.bboxes.shape: ", pred_instances.bboxes.shape)
                # print("len(pred_instances.labels): ", len(pred_instances.labels))
                encode_masks = encode_mask_results(pred_instances.masks)
                for encode_mask in encode_masks:
                    if isinstance(encode_mask['counts'], bytes):
                        encode_mask['counts'] = encode_mask['counts'].decode()
                result['masks'] = encode_masks

        if 'pred_panoptic_seg' in data_sample:
            if VOID is None:
                raise RuntimeError(
                    'panopticapi is not installed, please install it by: '
                    'pip install git+https://github.com/cocodataset/'
                    'panopticapi.git.')

            pan = data_sample.pred_panoptic_seg.sem_seg.cpu().numpy()[0]
            pan[pan % INSTANCE_OFFSET == len(
                self.model.dataset_meta['classes'])] = VOID
            pan = id2rgb(pan).astype(np.uint8)

            if is_save_pred:
                mmcv.imwrite(pan[:, :, ::-1], out_img_path)
                result['panoptic_seg_path'] = out_img_path
            else:
                result['panoptic_seg'] = pan

        if is_save_pred:
            mmengine.dump(result, out_json_path)

        # print("img_name_list:", img_name_list)
        # print("cls_name_list: ", cls_name_list)
        # print("bbox_area_list: ", bbox_area_list)
        # print("bbox_scores_list: ", scores_list)
        # print("mask_area_list: ", mask_area_list)
        
        bbox_mask_inf_results=[img_name_list, 
                               cls_name_list, 
                               bbox_area_list, 
                               scores_list, 
                               mask_area_list, 
                               xmin_list,
                               ymin_list,
                               xmax_list,
                               ymax_list
                               ]

        return result, bbox_mask_inf_results