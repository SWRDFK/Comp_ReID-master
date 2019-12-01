import argparse
import os
import ast
import torchvision.transforms as transforms
from core import Loaders, Base, train_an_epoch, test, ensemble
from tools import make_dirs, Logger, os_walk, time_now


def main(config):

	# init loaders and base
	loaders = Loaders(config)
	base = Base(config, loaders)

	# make directions
	make_dirs(base.output_path)
	make_dirs(base.save_model_path)
	make_dirs(base.save_log_path)

	# init logger
	logger = Logger(os.path.join(os.path.join(os.path.join(config.output_path, config.model_name), 'logs/'), 'log.txt'))
	logger('\n')
	logger(config)

	# train mode
	if config.mode == 'train':

		# resume model from the resume_train_epoch
		if config.resume_train_epoch >= 0:
			base.resume_model(config.resume_train_epoch)
			start_train_epoch = config.resume_train_epoch
		else:
			start_train_epoch = 0

		# automatically resume model from the latest one
		if config.auto_resume_training_from_lastest_steps:
			root, _, files = os_walk(base.save_model_path)
			if len(files) > 0:
				# get indexes of saved models
				indexes = []
				for file in files:
					indexes.append(int(file.replace('.pkl', '').split('_')[-1]))
				indexes = sorted(list(set(indexes)), reverse=False)
				# resume model from the latest model
				base.resume_model(indexes[-1])
				start_train_epoch = indexes[-1]
				logger('Time: {}, automatically resume training from the latest step (model {})'.format(time_now(), indexes[-1]))

		# main loop
		for current_epoch in range(start_train_epoch, config.total_train_epochs):

			# save model
			base.save_model(current_epoch)

			# train
			base.lr_scheduler.step(current_epoch)
			_, results = train_an_epoch(config, base, loaders)
			logger('Time: {};  Epoch: {};  {}'.format(time_now(), current_epoch, results))


	# test mode
	elif config.mode == 'test':

		test(config, base, loaders)


	# ensemble mode
	elif config.mode == 'ensemble':

		ensemble(config, base, loaders)



if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	# overall configuration
	parser.add_argument('--cuda', type=str, default='cuda')
	parser.add_argument('--mode', type=str, default='train', help='train, test or ensemble')
	parser.add_argument('--output_path', type=str, default='output', help='path to save models')
	parser.add_argument('--model_name', type=str, default='resnet101a_SA',
						help='resnet101a_SA, resnet101a_RLL or densenet161_CBL')

	# dataset configuration
	parser.add_argument('--dataset_path', type=str, default='dataset')
	parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
	parser.add_argument('--p', type=int, default=16, help='persons count in a batch')
	parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')

	# model configuration
	parser.add_argument('--pid_num', type=int, default=4768, help='labels count of train set')
	parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')

	# train configuration
	parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
						help='milestones for the learning rate decay')
	parser.add_argument('--base_learning_rate', type=float, default=0.00035)
	parser.add_argument('--weight_decay', type=float, default=0.0005)
	parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
	parser.add_argument('--total_train_epochs', type=int, default=120)
	parser.add_argument('--auto_resume_training_from_lastest_steps', type=ast.literal_eval, default=True)
	parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')


	# main
	config = parser.parse_args()
	main(config)
