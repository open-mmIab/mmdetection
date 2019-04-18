from __future__ import division
import os
from collections import OrderedDict

import torch
from mmcv.runner import Runner, DistSamplerSeedHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import wget
from mmdet.core import (DistOptimizerHook, DistEvalmAPHook,
						CocoDistEvalRecallHook, CocoDistEvalmAPHook)
from mmdet.datasets import build_dataloader
from mmdet.models import RPN
from .env import get_root_logger


# https://github.com/open-mmIab/mmdetection/blob/master/MODEL_ZOO.md
tmp = '.tmp'
weights_urls = {
	'rpn_r50_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_fpn_1x_20181010-4a9c0712.pth',
	'rpn_r50_fpn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_fpn_2x_20181010-88a4a471.pth',
	'rpn_r101_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r101_fpn_1x_20181129-f50da4bd.pth',
	'rpn_r101_fpn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r101_fpn_2x_20181129-e42c6c9a.pth',
	'rpn_x101_32x4d_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_32x4d_fpn_1x_20181218-7e379d26.pth',
	'rpn_x101_32x4d_fpn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_32x4d_fpn_2x_20181218-0510af40.pth',
	'rpn_x101_64x4d_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_64x4d_fpn_1x_20181218-c1a24f1f.pth',
	'rpn_x101_64x4d_fpn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_x101_64x4d_fpn_2x_20181218-c22bdd70.pth',
	'retinanet_r50_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-3d3c2142.pth',
	'retinanet_r50_fpn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_2x_20181125-e0dbec97.pth',
	'retinanet_r101_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r101_fpn_1x_20181129-f738a02f.pth',
	'retinanet_r101_fpn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r101_fpn_2x_20181129-f654534b.pth',
	'retinanet_x101_32x4d_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_32x4d_fpn_1x_20181218-c140fb82.pth',
	'retinanet_x101_32x4d_fpn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_32x4d_fpn_2x_20181218-605dcd0a.pth',
	'retinanet_x101_64x4d_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_64x4d_fpn_1x_20181218-2f6f778b.pth',
	'retinanet_x101_64x4d_fpn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_64x4d_fpn_2x_20181218-2f598dc5.pth',
	'mask_rcnn_r50_fpn_gn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_gn_2x_20180113-86832cf2.pth',
	'mask_rcnn_r50_fpn_gn_3x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_gn_3x_20180113-8e82f48d.pth',
	'mask_rcnn_r101_fpn_gn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r101_fpn_gn_2x_20180113-9598649c.pth',
	'mask_rcnn_r101_fpn_gn_3x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r101_fpn_gn_3x_20180113-a14ffb96.pth',
	'mask_rcnn_r50_fpn_gn_contrib_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_gn_contrib_2x_20180113-ec93305c.pth',
	'mask_rcnn_r50_fpn_gn_contrib_3x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_gn_contrib_3x_20180113-9d230cab.pth',
	'faster_rcnn_dconv_c3-c5_r50_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-e41688c9.pth',
	'faster_rcnn_mdconv_c3-c5_r50_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_mdconv_c3-c5_r50_fpn_1x_20190125-1b768045.pth',
	'faster_rcnn_dpool_r50_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dpool_r50_fpn_1x_20190125-f4fc1d70.pth',
	'faster_rcnn_mdpool_r50_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_mdpool_r50_fpn_1x_20190125-473d0f3d.pth',
	'faster_rcnn_dconv_c3-c5_r101_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-a7e31b65.pth',
	'faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_20190201-6d46376f.pth',
	'mask_rcnn_dconv_c3-c5_r50_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/mask_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-4f94ff79.pth',
	'mask_rcnn_mdconv_c3-c5_r50_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/mask_rcnn_mdconv_c3-c5_r50_fpn_1x_20190125-c5601dc3.pth',
	'mask_rcnn_dconv_c3-c5_r101_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/mask_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-decb6db5.pth',
	'cascade_rcnn_dconv_c3-c5_r50_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmleb/mmdetection/models/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth',
	'cascade_rcnn_dconv_c3-c5_r101_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth',
	'cascade_mask_rcnn_dconv_c3-c5_r50_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_mask_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-09d8a443.pth',
	'cascade_mask_rcnn_dconv_c3-c5_r101_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_mask_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-0d62c190.pth',
	'cascade_rcnn_r50_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_1x_20181123-b1987c4a.pth',
	'cascade_rcnn_r50_fpn_20e': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth',
	'cascade_rcnn_r101_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth',
	'cascade_rcnn_r101_fpn_20e': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r101_fpn_20e_20181129-b46dcede.pth',
	'cascade_rcnn_x101_32x4d_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_1x_20181218-941c0925.pth',
	'cascade_rcnn_x101_32x4d_fpn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_2x_20181218-28f73c4c.pth',
	'cascade_rcnn_x101_64x4d_fpn_1x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth',
	'cascade_rcnn_x101_64x4d_fpn_2x': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth',
}


def parse_losses(losses):
	log_vars = OrderedDict()
	for loss_name, loss_value in losses.items():
		if isinstance(loss_value, torch.Tensor):
			log_vars[loss_name] = loss_value.mean()
		elif isinstance(loss_value, list):
			log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
		else:
			raise TypeError(
				'{} is not a tensor or list of tensors'.format(loss_name))

	loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

	log_vars['loss'] = loss
	for name in log_vars:
		log_vars[name] = log_vars[name].item()

	return loss, log_vars


def batch_processor(model, data, train_mode):
	losses = model(**data)
	loss, log_vars = parse_losses(losses)

	outputs = dict(
		loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

	return outputs


def train_detector(model,
				   dataset,
				   cfg,
				   distributed=False,
				   validate=False,
				   logger=None):
	if logger is None:
		logger = get_root_logger(cfg.log_level)

	if not os.path.exists(tmp):
		wget.download(weights_urls[cfg.model_name], tmp)
	# start training
	if distributed:
		_dist_train(model, dataset, cfg, validate=validate)
	else:
		_non_dist_train(model, dataset, cfg, validate=validate)


def _dist_train(model, dataset, cfg, validate=False):
	# prepare data loaders
	data_loaders = [
		build_dataloader(
			dataset,
			cfg.data.imgs_per_gpu,
			cfg.data.workers_per_gpu,
			dist=True)
	]
	# put model on gpus
	model = MMDistributedDataParallel(model.cuda())
	# build runner
	runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
					cfg.log_level)
	# register hooks
	optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
	runner.register_training_hooks(cfg.lr_config, optimizer_config,
								   cfg.checkpoint_config, cfg.log_config)
	runner.register_hook(DistSamplerSeedHook())
	# register eval hooks
	if validate:
		if isinstance(model.module, RPN):
			# TODO: implement recall hooks for other datasets
			runner.register_hook(CocoDistEvalRecallHook(cfg.data.val))
		else:
			if cfg.data.val.type == 'CocoDataset':
				runner.register_hook(CocoDistEvalmAPHook(cfg.data.val))
			else:
				runner.register_hook(DistEvalmAPHook(cfg.data.val))

	if cfg.resume_from:
		runner.resume(cfg.resume_from)
	elif cfg.load_from:
		runner.load_checkpoint(cfg.load_from)
	else:
		runner.load_checkpoint(tmp)
	runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False):
	# prepare data loaders
	data_loaders = [
		build_dataloader(
			dataset,
			cfg.data.imgs_per_gpu,
			cfg.data.workers_per_gpu,
			cfg.gpus,
			dist=False)
	]
	# put model on gpus
	model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
	# build runner
	runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
					cfg.log_level)
	runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
								   cfg.checkpoint_config, cfg.log_config)

	if cfg.resume_from:
		runner.resume(cfg.resume_from)
	elif cfg.load_from:
		runner.load_checkpoint(cfg.load_from)
	else:
		runner.load_checkpoint(tmp)
	runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
