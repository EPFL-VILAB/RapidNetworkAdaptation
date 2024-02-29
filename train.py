import torch
import torch.nn as nn

from utils import *
from energy import get_energy_loss
from graph import TaskGraph
from logger import Logger, VisdomLogger
from datasets import load_train_val, load_test, load_ood
from task_configs import tasks, RealityTask
from transfers import functional_transfers

import wandb
from fire import Fire

def main(
	loss_config="baseline_depth_loop_film",
	fast=False, batch_size=None, lr=3e-5, resume=False, run_name=None,
	subset_size=None, max_epochs=5000, dataaug=False, target_task='depth_zbuffer', noise_max_std=0.17, 
	baseline_model_path='pretrained_models/rgb2depth.pth',
	base_save_dir='./results/', **kwargs,
):

	# CONFIG
	wandb.init(project=f"rna-{target_task}", entity="rna", name=run_name, job_type="train")
	batch_size = batch_size or (4 if fast else 64)
	wandb.config.update({"loss_config":loss_config,"batch_size":batch_size,"lr":lr,"target_model":run_name,"baseline_model_path":baseline_model_path})

	energy_loss = get_energy_loss(config=loss_config, **kwargs)

	# DATA LOADERS
	train_dataset, val_dataset, train_step, val_step = load_train_val(
		energy_loss.get_tasks("train"),
		batch_size=batch_size, fast=fast,
		subset_size=subset_size,
		dataaug=dataaug,
	)
	test_set = load_test(energy_loss.get_tasks("test"))

	train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
	val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
	test = RealityTask.from_static("test", test_set, energy_loss.get_tasks("test"))

	# GRAPH
	realities = [train, val, test] 
	graph = TaskGraph(tasks=energy_loss.tasks + realities, pretrained=True, finetuned=False,
		freeze_list=energy_loss.freeze_list, target_task=target_task, noise_max_std=noise_max_std
	)
	graph.compile(torch.optim.Adam, lr=lr, weight_decay=2e-6, amsgrad=True)

	results_dir = f"results_{run_name}"
	if resume:
		graph.load_weights(f'{base_save_dir}{results_dir}/graph.pth')
	else:
		os.system(f"mkdir -p {base_save_dir}{results_dir}")
		tmp = torch.load(baseline_model_path) 
		print(f"loading rgb2{target_task}")
		graph.edge_map[str(('rgb', target_task))].load_state_dict(tmp, strict=False)

	# LOGGING
	logger = VisdomLogger("train", env=run_name)    # fake visdom logger
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	energy_loss.logger_hooks(logger)

	######## baseline computation
	if not resume:
		graph.eval()
		with torch.no_grad():
			for _ in range(0, val_step//4):
				val_loss = energy_loss(graph, realities=[val])
				val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
				val.step()
				logger.update("loss", val_loss)

			for _ in range(0, train_step//4):
				train_loss = energy_loss(graph, realities=[train])
				train_loss = sum([train_loss[loss_name] for loss_name in train_loss])
				train.step()
				logger.update("loss", train_loss)

		energy_loss.logger_update(logger)
		data=logger.step()
		del data['loss']
		data = {k:v[0] for k,v in data.items()}
		wandb.log(data, step=0)

		path_values = energy_loss.plot_paths(graph, logger, realities, prefix="")
		for reality_paths, reality_images in path_values.items():
			wandb.log({reality_paths: [wandb.Image(reality_images)]}, step=0)
	###########


	# TRAINING
	for epochs in range(0, max_epochs):

		logger.update("epoch", epochs)

		graph.eval()
		for _ in range(0, val_step):
			with torch.no_grad():
				val_loss = energy_loss(graph, realities=[val])
				val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
			val.step()
			logger.update("loss", val_loss)

		graph.train()
		for _ in range(0, train_step):
			train_loss = energy_loss(graph, realities=[train])
			train_loss = sum([train_loss[loss_name] for loss_name in train_loss])
			graph.step(train_loss)
			train.step()
			logger.update("loss", train_loss)

		energy_loss.logger_update(logger)

		data=logger.step()
		del data['loss']
		del data['epoch']
		data = {k:v[0] for k,v in data.items()}
		wandb.log(data, step=epochs+1)

		if epochs % 10 == 0:
			graph.save(f"{base_save_dir}{results_dir}/graph.pth")
			torch.save(graph.optimizer.state_dict(),f"{base_save_dir}{results_dir}/opt.pth")

		if (epochs<=50 and epochs % 10 == 0) or (epochs>=100 and epochs % 200==0):
			path_values = energy_loss.plot_paths(graph, logger, realities, prefix="")
			for reality_paths, reality_images in path_values.items():
				wandb.log({reality_paths: [wandb.Image(reality_images)]}, step=epochs+1)


	graph.save(f"{base_save_dir}{results_dir}/graph.pth")
	torch.save(graph.optimizer.state_dict(),f"{base_save_dir}{results_dir}/opt.pth")

if __name__ == "__main__":
	Fire(main)
