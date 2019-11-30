import torch
import numpy as np
from tools import CatMeter, cosine_dist, euclidean_dist, re_ranking


def generate_jsonfile(distmat, dataset, topk):
	"""
	Args:
		distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
		dataset (tuple): a 2-tuple containing (query, gallery), each of which contains tuples of (img_path(s), pid).
		topk: return topk ranks.
	"""

	num_q, num_g = distmat.shape
	indices = np.argsort(distmat, axis=1)

	query, gallery = dataset
	assert num_q == len(query)
	assert num_g == len(gallery)

	print('Compute result with top-{} ranks'.format(topk))
	print('# query: {}\n# gallery {}'.format(num_q, num_g))

	result_dict = {}
	qlist = []

	for q_idx in range(num_q):
		qimg_path, qpid = query[q_idx]
		# choose test_set A or B
		# query_name = qimg_path.replace('/home/kangning/Competition/query_a/', '')
		query_name = qimg_path.replace('/home/kangning/Competition/query_b/', '')

		g_num = 0
		glist = []
		glist.append(query_name)
		for g_idx in indices[q_idx, :]:
			gimg_path, gpid = gallery[g_idx]
			# choose test_set A or B
			# gallery_name = gimg_path.replace('/home/kangning/Competition/gallery_a/', '')
			gallery_name = gimg_path.replace('/home/kangning/Competition/gallery_b/', '')

			if g_num < topk:
				glist.append(gallery_name)
			g_num += 1
		qlist.append(glist)

	for i in range(len(qlist)):
		for j in range(1, len(qlist[i])):
			result_dict.setdefault(qlist[i][0], []).append(qlist[i][j])

	# generate json
	import json

	json = json.dumps(result_dict)
	jsonfile = 'TestB_result_4768_one_ibn101a_SA_rerank+4768_one_ibn101a_rankedloss_rerank+4768_one_dense161_cbloss_beta_rerank0.48.json'

	with open(jsonfile, 'w') as f:
		f.write(json)

	print("Successfully generate jsonfile: {}".format(jsonfile))


def test(config, base, loaders):

	base.set_eval()

	# meters
	query_features_meter, gallery_features_meter = CatMeter(), CatMeter()

	# init dataset
	_datasets = [loaders.comp_query_samples.samples, loaders.comp_gallery_samples.samples]
	_loaders = [loaders.comp_query_loader, loaders.comp_gallery_loader]

	# compute query and gallery features
	with torch.no_grad():
		for loader_id, loader in enumerate(_loaders):
			for data in loader:
				# compute feautres
				images, _ = data

				# features = base.model(images)
				features, _ = base.model(images)      # For Spatial Attention

				# save as query features
				if loader_id == 0:
					query_features_meter.update(features.data)
				# save as gallery features
				elif loader_id == 1:
					gallery_features_meter.update(features.data)

	# get torch.Tensor
	query_features = query_features_meter.get_val()
	gallery_features = gallery_features_meter.get_val()

	# compute distance: cosine, euclidean distance or re-ranking
	# distance = -cosine_dist(query_features, gallery_features).data.cpu().numpy()
	# distance = euclidean_dist(query_features, gallery_features).data.cpu().numpy()
	distance = re_ranking(query_features, gallery_features)
	np.save("TestB_distance_4768_one_ibn101a_SA_rerank.npy", distance)

	# generate submission file containing top-200 ranks
	generate_jsonfile(distance, _datasets, 200)


def ensemble(config, base, loaders):

	base.set_eval()

	# init dataset
	_datasets = [loaders.comp_query_samples.samples, loaders.comp_gallery_samples.samples]

	# test_set A
	# d1 = np.load("/home/kangning/AI-baseline-master/distance_4768_one_ibn101a_SA_rerank.npy")
	# d2 = np.load("/home/kangning/AI-baseline-master/distance_4768_one_ibn101a_rankedloss_rerank.npy")
	# d3 = np.load("/home/kangning/AI-baseline-master/distance_4768_one_dense161_cbloss_beta_rerank.npy")

	# test_set B
	d1 = np.load("/home/kangning/AI-baseline-master/TestB_distance_4768_one_ibn101a_SA_rerank.npy")
	d2 = np.load("/home/kangning/AI-baseline-master/TestB_distance_4768_one_ibn101a_rankedloss_rerank.npy")
	d3 = np.load("/home/kangning/AI-baseline-master/TestB_distance_4768_one_dense161_cbloss_beta_rerank.npy")

	distance = d1 + d2 + 0.48 * d3
	generate_jsonfile(distance, _datasets, 200)
