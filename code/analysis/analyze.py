import scipy.stats as ss
from scipy.stats import norm
import glob
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 150
import pandas as pd
import seaborn as sns
from collections import defaultdict
import os
sns.set()

fig_dir = os.path.join("..", '..','results',"figures")
storage_dir = os.path.join("..","..","storage")

from collections import defaultdict
model_stats = defaultdict(dict)

def calculate_ci(datapoints, alpha=.95):
	mean = np.mean(datapoints)
	total_points = datapoints

	data_greater = sorted(datapoints[datapoints > mean])
	data_less = sorted(datapoints[datapoints < mean], reverse=True)
	curr_d = int(len(datapoints) * alpha)
	'''
	print("====")
	print(len(datapoints))
	print(len(data_greater))
	print(len(data_less))
	print(int(curr_d/2))
	48000
	22414
	25586
	22800
	'''
	cur_greater = int(curr_d/2)
	cur_less = int(curr_d/2)
	if int(curr_d/2) > len(data_greater):
		cur_greater = len(data_greater) - 1
		cur_less += int(curr_d/2) - len(data_greater)
	if int(curr_d/2) > len(data_less):
		cur_less = len(data_less) - 1
		cur_greater += int(curr_d/2) - len(data_less)

	return data_less[cur_less], data_greater[cur_greater]

def plot_posteriors(model_name, chains):
	''' Plot the main posteriors for this model
	'''
	if "09" in model_name:
		return
	plt.figure()

	for param in ['alpha', 'tau', 'beta']:
		plt.figure()	
		param_data = np.array(chains.get(param)).flatten()

		# Plot Young
		plt.hist(param_data, bins=50, density=True, label='Young', alpha=.5)
	
		diff_data = np.array(chains.get(param+'diff')).flatten() + param_data
		# Plot Old
		plt.hist(diff_data, bins=50, density=True, label='Old', alpha=.5)


		if param == 'beta':
			model_stats[model_name]['stats_beta_.5_young'] = sum(param_data < .5) / len(param_data)
			model_stats[model_name]['stats_beta_.5_old'] = sum(diff_data < .5) / len(diff_data)
		model_stats[model_name]['stats_{}_young'.format(param)] = '{:.2f} [{:.2f}, {:.2f}]'.format(np.mean(param_data), *calculate_ci(param_data))
		model_stats[model_name]['stats_{}_old'.format(param)] = '{:.2f} [{:.2f}, {:.2f}]'.format(np.mean(diff_data), *calculate_ci(diff_data))

		plt.title(param.capitalize() + " Posteriors")
		plt.legend()
		plt.savefig(os.path.join(fig_dir, 'posteriors', "{}_{}_posteriors.png".format(model_name, param)))
		plt.close()

		plt.figure()
		old_data = np.array(chains.get("{}subjrep_1".format(param))).flatten()
		young_data   = np.array(chains.get("{}subjrep_2".format(param))).flatten()
		model_stats[model_name]['stats_{}_young_greater_than_old'.format(param)] = sum(young_data > old_data) / len(old_data)
		
		plt.hist(young_data, bins=50, density=True, label='Young', alpha=.5)
		plt.hist(old_data, bins=50, density=True, label='Old', alpha=.5)
		plt.title(param + " Subj Rep")
		plt.legend()
		plt.savefig(os.path.join(fig_dir, 'posteriors', '{}_{}_rep.png'.format(model_name, param)))
		plt.close()

	# for violin plots
	young_data_violin = {}
	old_data_violin = {}

	for cond_num, label in enumerate(['Targ', 'HSim', 'LSim', 'Foil']):
		plt.figure()
		if "delta_1_{}_1".format(cond_num+1) in chains:
			param_data = np.array(chains.get("delta_1_{}_1".format(cond_num+1))).flatten()
			diff_data = np.array(chains.get("delta_2_{}_1".format(cond_num+1))).flatten()			
		else:
			param_data = np.array(chains.get("delta_1_{}".format(cond_num+1))).flatten()
			diff_data = np.array(chains.get("delta_2_{}".format(cond_num+1))).flatten()			
		# Plot Old
		plt.hist(param_data, bins=50, density=True, label='Old', alpha=.5)
		# Plot Young
		plt.hist(diff_data, bins=50, density=True, label='Young', alpha=.5)

		young_data_violin[label] = diff_data
		old_data_violin[label] = param_data

		model_stats[model_name]['stats_delta_{}_old'.format(label)] = '{:.2f}, CI: [{:.2f}, {:.2f}]'.format(np.mean(param_data), *calculate_ci(param_data))
		model_stats[model_name]['stats_delta_{}_young'.format(label)] = '{:.2f}, CI: [{:.2f}, {:.2f}]'.format(np.mean(diff_data), *calculate_ci(diff_data))
	
		plt.title("Delta " + label.capitalize() + " Posteriors")
		plt.legend()
		plt.savefig(os.path.join(fig_dir, 'posteriors', "{}_delta_{}_posteriors.png".format(model_name, label)))

		plt.close()	

		if "deltasubjrep_1_1" not in chains:
			continue
		plt.figure()
		old_data = np.array(chains.get("deltasubjrep_1_{}".format(cond_num+1))).flatten()
		young_data   = np.array(chains.get("deltasubjrep_2_{}".format(cond_num+1))).flatten()
		model_stats[model_name]['stats_{}_young_greater_than_old'.format(label)] = sum(young_data > old_data) / len(old_data)
		
		plt.hist(young_data, bins=150, density=True, label='Young', alpha=.5)
		plt.hist(old_data, bins=150, density=True, label='Old', alpha=.5)
		if 'Sim' in label:
			plt.xlim([-5,6])
		if 'Foil' in label:
			plt.xlim([0,11])
		plt.title("Delta " + label + " Subj Rep")
		plt.legend()
		plt.savefig(os.path.join(fig_dir, 'posteriors', '{}_delta_{}_rep.png'.format(model_name, label)))
		plt.close()


	#### violin plot for posteriors of delta
	old_df = pd.DataFrame(old_data_violin)
	young_df = pd.DataFrame(young_data_violin)
	old_df['Group'] = 'Old'
	young_df['Group'] = 'Young'
	violin_df = pd.concat([old_df, young_df])
	violin_df = pd.melt(violin_df, id_vars=['Group'], value_vars=['Targ', 'HSim', 'LSim', 'Foil'], var_name='Condition', value_name='Drift Rate')
	sns.violinplot(x='Condition', y='Drift Rate', hue='Group', data=violin_df)
	plt.savefig(os.path.join(fig_dir, 'posteriors', '{}_delta_violin.png'.format(model_name)))
	plt.close()


	for group_num, group_label in enumerate(['Old', 'Young']):
		plt.figure()
		for cond_num, label in enumerate(['Targ', 'HSim', 'LSim', 'Foil']):
			if "delta_1_{}_1".format(cond_num+1) in chains:
				param_data = np.array(chains.get("delta_{}_{}_1".format(group_num+1, cond_num+1))).flatten()
			else:
				param_data = np.array(chains.get("delta_{}_{}".format(group_num+1, cond_num+1))).flatten()
			plt.hist(param_data, bins=50, density=True, label=label, alpha=.5)
		plt.title(group_label)
		plt.legend()
		plt.savefig(os.path.join(fig_dir, 'posteriors', '{}_{}_delta_all.png'.format(model_name, group_label)))
		plt.close()

	plt.figure()
	data = np.array(chains.get("deviance")).flatten()
	plt.hist(data, bins=50, density=True)
	plt.title("Deviance")
	plt.savefig(os.path.join(fig_dir, 'posteriors', "{}_deviance.png".format(model_name)))
	plt.close()	

def plot_diff(model_name, label, vals, prior_mean, prior_sd, title, fig_name):
	''' Plot the difference distributions along with priors
	'''
	plt.figure()

	# Plot difference posterior
	plt.hist(vals, bins=50, density=True, label='Posterior')

	cur_xlim = plt.gca().get_xlim()
	min_x = min(cur_xlim[0], prior_mean-prior_sd*3)
	max_x = max(cur_xlim[1], prior_mean+prior_sd*3)

	# Plot priors
	x = np.linspace(prior_mean-prior_sd*4,prior_mean+prior_sd*4, 100)
	y_pdf = ss.norm.pdf(x, prior_mean, prior_sd) # the normal pdf
	plt.plot(x, y_pdf, label='Prior')

	mu, sd = norm.fit(vals)	
	x = np.linspace(mu-sd*4,mu+sd*4, 100)
	y_pdf = ss.norm.pdf(x, mu, sd) # the normal pdf
	plt.plot(x, y_pdf, label='Normal Posterior Fit')

	model_stats[model_name]["{}_{}".format(fig_name.split(".")[0],'BF_zero')] = ss.norm.pdf([0], prior_mean, prior_sd)[0] / ss.norm.pdf([0], mu, sd)[0]

	plt.xlim([min_x,max_x])
	plt.title(title)
	plt.legend()
	plt.savefig(os.path.join(fig_dir, 'posteriors', "{}_{}".format(model_name, fig_name)))
	plt.close()

def calculate_posterior_predictives(model_name, f):

	n_subj = int(np.array(f.get("data").get("nSubjects"))[0][0])
	n_trials = int(np.array(f.get("data").get("nAllTrials"))[0][0])
	trial_sim = np.array(f.get("data").get("subList")).T
	group_list = np.array(f.get("data").get("groupList")).T.flatten()

	n_chains = int(np.array(f.get("info").get("options").get("nchains"))[0][0])
	n_samples = int(np.array(f.get("info").get("options").get("nsamples"))[0][0])
	n_values = n_chains * n_samples

	print("Filling ypred....")
	# Load in all the y_pred simulated values (lots of values)
	chains = f.get('chains')
	y_pred = np.zeros((n_subj,n_trials,n_values))
	for s in range(n_subj):
		for tr in range(n_trials):
			exec('y_pred[s,tr,:] = np.array(chains.get("ypred_' + str(s+1) + '_' + str(tr+1) + '")).flatten()')
	print("Loading y")
	# Load in original y
	y = np.array(f.get("data").get("y")).T

	y_vals = np.zeros((n_subj, n_trials))
	ypred_vals = np.zeros((n_subj, n_trials))
	ypred_exp = np.zeros((n_subj, n_trials))
	# Set ypred_vals to -1 or 1 based on a majority vote of the simulated values
	for s in range(n_subj):
		for tr in range(n_trials):
			# If the original trial is not nan
			if not np.isnan(y[s,tr]):
				# Change y_vals to be -1 or 1
				if y[s,tr] > 0:
					y_vals[s,tr] = 1
				else:
					y_vals[s,tr] = -1
				# Set ypred_vals to be -1 or 1 by a vote from chains
				if np.count_nonzero(y_pred[s,tr,:] > 0) > len(y_pred[s,tr,:]) / 2:
					ypred_vals[s,tr] = 1
				else:
					ypred_vals[s,tr] = -1
				ypred_exp[s,tr] = np.count_nonzero(y_pred[s,tr,:] > 0) / len(y_pred[s,tr,:])
			# original trial is nan, so don't look at predicted
			else:
				ypred_vals[s,tr] = np.nan
				y_vals[s,tr] = np.nan
				ypred_exp[s,tr] = np.nan

	matching = ypred_vals == y_vals
	matching = matching.astype(int)
	matching = matching.astype(float)
	matching[np.isnan(y)] = np.nan
	model_stats[model_name]['PP_total_accuracy'] = np.nanmean(matching)
	print("Analyzing accuracy")
	labels = ['Targ', 'HSim', 'LSim', 'Foil']
	for i in range(4):
		matching = ypred_vals == y_vals
		matching = matching.astype(int)
		matching = matching.astype(float)
		matching[np.isnan(y)] = np.nan
		matching[trial_sim != i+1] = np.nan
		model_stats[model_name]['PP_{}_accuracy'.format(labels[i])] = np.nanmean(matching)

	accs = [[],[],[],[]]
	pred_accs = [[],[],[],[]]
	pred_exps = [[],[],[],[]]
	for i in range(4):
		y_sim = y_vals[trial_sim == i+1].reshape(y.shape[0],40)
		ypred_sim = ypred_vals[trial_sim == i+1].reshape(y.shape[0],40)
		ypred_exp_sim = ypred_exp[trial_sim == i+1].reshape(y.shape[0], 40)
		for subj in range(y.shape[0]):
			if i == 0:
				corr_resp = -1
			else:
				corr_resp = 1
			subj_num_corr = sum((y_sim == corr_resp)[subj,:])
			subj_num_resp = sum(np.invert(np.isnan(y_sim))[subj,:])
			accs[i].append(subj_num_corr/subj_num_resp)
			subjpred_num_corr = sum((ypred_sim == corr_resp)[subj,:])
			subjpred_num_resp = sum(np.invert(np.isnan(y_sim))[subj,:])
			pred_accs[i].append(subjpred_num_corr/subjpred_num_resp)
	
			pred_exps[i].append(np.nanmean(ypred_exp_sim[subj,:]))
	# Normal acc vs modeled acc
	plt.figure()
	for i, label in enumerate(['Targ', 'HSim', 'LSim', 'Foil']):
		plt.scatter(accs[i], pred_accs[i], label=label, marker='+')
	plt.plot([0,1], [0,1])
	plt.xlim([-.05,1.05])
	plt.ylim([-.05,1.05])
	plt.title("Posterior Predictive Accuracy")
	plt.legend()
	plt.xlabel("Subject Accuracy")
	plt.ylabel("Modeled Accuracy")
	plt.savefig(os.path.join(fig_dir, 'posterior_predictives', "{}_pp_acc.png".format(model_name)))
	plt.close()

	# Normal acc vs expected acc
	plt.figure()
	# Target we want to switch since we calculated it the same as lures+foil
	pred_exps[0] = [1-x for x in pred_exps[0]]
	for i, label in enumerate(['Targ', 'HSim', 'LSim', 'Foil']):
		plt.scatter(accs[i], pred_exps[i], label=label, marker='+')
	plt.plot([0,1], [0,1])
	plt.xlim([-.05,1.05])
	plt.ylim([-.05,1.05])
	plt.title("Posterior Predictive Accuracy")
	plt.legend()
	plt.xlabel("Subject Accuracy")
	plt.ylabel("Modeled Accuracy")
	plt.savefig(os.path.join(fig_dir, 'posterior_predictives', "{}_pp_exp_acc.png".format(model_name)))
	plt.close()


		
	print("Analyzing RT")
	# Analyze mean absolute error on RTs for each subj
	rt_err_pos = [defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)]
	rt_prd_pos = [defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)]
	rt_act_pos = [defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)]
	rt_err_neg = [defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)]
	rt_prd_neg = [defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)]
	rt_act_neg = [defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)]
	# For quantile graphs

	#point_1_quantile = [[[],[]], [[],[]]]
	#point_5_quantile = [[[],[]], [[],[]]]
	#point_9_quantile = [[[],[]], [[],[]]]
	point_1_quantile = {}
	point_5_quantile = {}
	point_9_quantile = {}
	for i in range(4):
		point_1_quantile[i] = {-1: [[[],[]],[[],[]]], 1:[[[],[]],[[],[]]]}
		point_5_quantile[i] = {-1: [[[],[]],[[],[]]], 1:[[[],[]],[[],[]]]}
		point_9_quantile[i] = {-1: [[[],[]],[[],[]]], 1:[[[],[]],[[],[]]]}

	for i in range(4):
		y_sim = y[trial_sim == i+1].reshape(y.shape[0],40)
		rt_sim = y_pred[trial_sim == i+1].reshape(y_pred.shape[0], 40, y_pred.shape[2])
		# for each subj
		for subj in range(rt_sim.shape[0]):
			rt_err_pos[i][subj] += []
			rt_prd_pos[i][subj] += []
			rt_act_pos[i][subj] += []
			rt_err_neg[i][subj] += []
			rt_prd_neg[i][subj] += []
			rt_act_neg[i][subj] += []

			subj_resp = y_sim[subj,:]
			pred_subj_resp = rt_sim[subj,:]
			
			# positive response
			subj_resp_pos = subj_resp[subj_resp > 0]
			preds_subj_resp_pos = pred_subj_resp[subj_resp>0]

			# quantile shit
			if len(subj_resp_pos) > 10:
				sub_group = group_list[subj]-1
				quantile_pred = np.array(preds_subj_resp_pos)
				quantile_pred[quantile_pred < 0] = np.nan
				quantile_pred = np.abs(quantile_pred).flatten()
				quantile_pred = quantile_pred[~np.isnan(quantile_pred)]
				print(subj_resp_pos.shape)
				print(quantile_pred.shape)
				x1 = np.quantile(subj_resp_pos, .1)
				y1 = np.quantile(quantile_pred, .1)
				print(x1, y1)
				point_1_quantile[i][1][sub_group][0].append(x1)
				point_1_quantile[i][1][sub_group][1].append(y1)
				x1 = np.quantile(subj_resp_pos, .5)
				y1 = np.quantile(quantile_pred, .5)
				point_5_quantile[i][1][sub_group][0].append(x1)
				point_5_quantile[i][1][sub_group][1].append(y1)
				x1 = np.quantile(subj_resp_pos, .9)
				y1 = np.quantile(quantile_pred, .9)
				point_9_quantile[i][1][sub_group][0].append(x1)
				point_9_quantile[i][1][sub_group][1].append(y1)


			for t in range(len(subj_resp_pos)): 
				trial_pred = np.mean(np.abs(preds_subj_resp_pos[t,:][preds_subj_resp_pos[t,:] > 0]))
				trial_err = abs(subj_resp_pos[t] - trial_pred)		
				rt_err_pos[i][subj].append(trial_err)
				rt_prd_pos[i][subj].append(trial_pred)
				rt_act_pos[i][subj].append(abs(subj_resp_pos[t]))

			# negative response
			subj_resp_neg = subj_resp[subj_resp < 0]
			preds_subj_resp_neg = pred_subj_resp[subj_resp<0]

			# quantile shit
			if len(subj_resp_neg) > 10:
				sub_group = group_list[subj]-1
				quantile_pred = np.array(preds_subj_resp_neg)
				quantile_pred[quantile_pred < 0] = np.nan
				quantile_pred = np.abs(quantile_pred).flatten()
				quantile_pred = quantile_pred[~np.isnan(quantile_pred)]
				x1 = np.quantile(abs(subj_resp_neg), .1)
				y1 = np.quantile(quantile_pred, .1)
				point_1_quantile[i][-1][sub_group][0].append(abs(x1))
				point_1_quantile[i][-1][sub_group][1].append(abs(y1))
				x1 = np.quantile(abs(subj_resp_neg), .5)
				y1 = np.quantile(quantile_pred, .5)
				point_5_quantile[i][-1][sub_group][0].append(abs(x1))
				point_5_quantile[i][-1][sub_group][1].append(abs(y1))
				x1 = np.quantile(abs(subj_resp_neg), .9)
				y1 = np.quantile(quantile_pred, .9)
				point_9_quantile[i][-1][sub_group][0].append(abs(x1))
				point_9_quantile[i][-1][sub_group][1].append(abs(y1))

			for t in range(len(subj_resp_neg)): 
				trial_pred = np.mean(np.abs(preds_subj_resp_neg[t,:][preds_subj_resp_neg[t,:] < 0]))
				trial_err = abs(abs(subj_resp_neg[t]) - trial_pred)		
				rt_err_neg[i][subj].append(trial_err)
				rt_prd_neg[i][subj].append(trial_pred)
				rt_act_neg[i][subj].append(abs(subj_resp_neg[t]))

	# QUANTILE GRAPH
	o_p1_corr = []
	o_p1_incorr = []
	o_p5_corr = []
	o_p5_incorr = []
	o_p9_corr = []
	o_p9_incorr = []
	y_p1_corr = []
	y_p1_incorr = []
	y_p5_corr = []
	y_p5_incorr = []
	y_p9_corr =[]
	y_p9_incorr = []

	for cond_idx, cond_label in enumerate(['Targ', 'HSim', 'LSim', 'Foil']):
		for resp_label, resp_idx in [['New', 1], ['Old', -1]]:
			for group_idx, group_label in enumerate(['Old', 'Young']):
				fig, (ax0, ax1, ax2) = plt.subplots(1,3, sharey=True, figsize=(20,5))
				fig.suptitle(group_label)
				ymin = .5
				ymax = 2
				xmin = .5
				xmax = 2
				if group_label == 'Old':
					if (cond_label != 'Targ' and resp_label == 'New') or (cond_label == 'Targ' and resp_label == 'Old'):
						o_p1_corr.append([point_1_quantile[cond_idx][resp_idx][group_idx][0], point_1_quantile[cond_idx][resp_idx][group_idx][1]])
						o_p5_corr.append([point_5_quantile[cond_idx][resp_idx][group_idx][0], point_5_quantile[cond_idx][resp_idx][group_idx][1]])
						o_p9_corr.append([point_9_quantile[cond_idx][resp_idx][group_idx][0], point_9_quantile[cond_idx][resp_idx][group_idx][1]])
					else:
						o_p1_incorr.append([point_1_quantile[cond_idx][resp_idx][group_idx][0], point_1_quantile[cond_idx][resp_idx][group_idx][1]])
						o_p5_incorr.append([point_5_quantile[cond_idx][resp_idx][group_idx][0], point_5_quantile[cond_idx][resp_idx][group_idx][1]])
						o_p9_incorr.append([point_9_quantile[cond_idx][resp_idx][group_idx][0], point_9_quantile[cond_idx][resp_idx][group_idx][1]])
				else:
					if (cond_label != 'Targ' and resp_label == 'New') or (cond_label == 'Targ' and resp_label == 'Old'):
						y_p1_corr.append([point_1_quantile[cond_idx][resp_idx][group_idx][0], point_1_quantile[cond_idx][resp_idx][group_idx][1]])
						y_p5_corr.append([point_5_quantile[cond_idx][resp_idx][group_idx][0], point_5_quantile[cond_idx][resp_idx][group_idx][1]])
						y_p9_corr.append([point_9_quantile[cond_idx][resp_idx][group_idx][0], point_9_quantile[cond_idx][resp_idx][group_idx][1]])
					else:
						y_p1_incorr.append([point_1_quantile[cond_idx][resp_idx][group_idx][0], point_1_quantile[cond_idx][resp_idx][group_idx][1]])
						y_p5_incorr.append([point_5_quantile[cond_idx][resp_idx][group_idx][0], point_5_quantile[cond_idx][resp_idx][group_idx][1]])
						y_p9_incorr.append([point_9_quantile[cond_idx][resp_idx][group_idx][0], point_9_quantile[cond_idx][resp_idx][group_idx][1]])

				ax0.plot([-50,50],[-50,50])
				ax0.scatter(point_1_quantile[cond_idx][resp_idx][group_idx][0], point_1_quantile[cond_idx][resp_idx][group_idx][1], marker='+', s=5)
				ax0.set_title(".1 Quantile")
				ax0.set_xlim([xmin,xmax])
				ax0.set_ylim([ymin,ymax])

				ax1.plot([-50,50],[-50,50])
				ax1.scatter(point_5_quantile[cond_idx][resp_idx][group_idx][0], point_5_quantile[cond_idx][resp_idx][group_idx][1], marker='+', s=5)
				ax1.set_title(".5 Quantile")
				ax1.set_xlim([xmin,xmax])
				ax1.set_ylim([ymin,ymax])

				ax2.plot([-50,50],[-50,50])
				ax2.scatter(point_9_quantile[cond_idx][resp_idx][group_idx][0], point_9_quantile[cond_idx][resp_idx][group_idx][1], marker='+', s=5)
				ax2.set_title(".9 Quantile")
				ax2.set_xlim([xmin,xmax])
				ax2.set_ylim([ymin,ymax])

				plt.savefig(os.path.join(fig_dir, 'quantiles', "{}_{}_{}_{}.png".format(model_name,cond_label, resp_label,group_label)))
				plt.close()

	for quantiles, labels in [[[o_p1_corr, o_p5_corr, o_p9_corr], ['Old', 'Correct']], [[o_p1_incorr, o_p5_incorr, o_p9_incorr], ['Old', 'Incorrect']],
							  [[y_p1_corr, y_p5_corr, y_p9_corr], ['Young', 'Correct']], [[y_p1_incorr, y_p5_incorr, y_p9_incorr], ['Young', 'Incorrect']]]:

		# MORE QUANTILE
		fig, (ax0, ax1, ax2) = plt.subplots(1,3, sharey=True, figsize=(20,5))
		fig.suptitle(labels[0])
		ymin = .5
		ymax = 2
		xmin = .5
		xmax = 2

		ax0.plot([-50,50],[-50,50])
		ax0.scatter(np.hstack([np.array(x[0]) for x in quantiles[0]]).flatten(), np.hstack([x[1] for x in quantiles[0]]).flatten(),marker='+', s=5)
		ax0.set_title(".1 Quantile")
		ax0.set_xlim([xmin,xmax])
		ax0.set_ylim([ymin,ymax])

		ax1.plot([-50,50],[-50,50])
		ax1.scatter(np.hstack([x[0] for x in quantiles[1]]).flatten(), np.hstack([x[1] for x in quantiles[1]]).flatten(),marker='+', s=5)
		ax1.set_title(".5 Quantile")
		ax1.set_xlim([xmin,xmax])
		ax1.set_ylim([ymin,ymax])

		ax2.plot([-50,50],[-50,50])
		ax2.scatter(np.hstack([x[0] for x in quantiles[2]]).flatten(), np.hstack([x[1] for x in quantiles[2]]).flatten(),marker='+', s=5)
		ax2.set_title(".9 Quantile")
		ax2.set_xlim([xmin,xmax])
		ax2.set_ylim([ymin,ymax])

		plt.savefig(os.path.join(fig_dir, 'quantiles', "{}_{}_{}.png".format(model_name, labels[0],labels[1])))
		plt.close()
	


	def plot_violin(group_list, predicted_values, subj_resps, label):
		predicted_values = np.array(predicted_values)
		subj_resps = np.array([np.median(resps) for resps in subj_resps])

		# Sort them by age
		pred_vals_old = predicted_values[group_list == 1][np.argsort(subj_resps[group_list == 1])]
		pred_vals_young = predicted_values[group_list == 2][np.argsort(subj_resps[group_list == 2])]
		predicted_values = np.concatenate([pred_vals_old, pred_vals_young],axis=0)
		subj_resps = np.array(sorted(subj_resps[group_list == 1]) + sorted(subj_resps[group_list == 2]))
		
		predicted_values[np.isnan(subj_resps)] = -100
		subj_resps[np.isnan(subj_resps)] = -100
		plt.figure()
		# Plot the error bars for all the predicted values
		mus = [np.mean(vals) for vals in predicted_values]
		sds = [np.std(vals)*2 for vals in predicted_values]
		
		#plt.violinplot(predicted_values)
		colors = np.where(group_list == 1, 'g', 'b')
		plt.scatter(np.arange(1,len(subj_resps)+1), np.array(subj_resps), c=colors)
		plt.title("Modeled {} RT Distributions".format(label))

		plt.errorbar(np.arange(1,len(subj_resps)+1)+.25, mus, yerr=sds, fmt='none')	

		import matplotlib.lines as mlines
		line = mlines.Line2D([],[], color='#2b7bba')
		s1 = plt.scatter([],[], color='green')
		s2 = plt.scatter([],[], color='blue')
		plt.legend(handles=[line,s1, s2], labels=['Modeled RTs', 'Median Old RTs', 'Median Young RTs'])
		#plt.legend('Predicted RTs', 'Mean Subj RTs')
		plt.ylim([-.1, 2.1])	
		plt.xlabel("Subject")
		plt.ylabel("RT")
		plt.savefig(os.path.join(fig_dir, 'posterior_predictives', "{}_{}_pp_rt.png".format(model_name, label)))
		plt.close()
		
			
	for i, label in enumerate(['Targ', 'HSim', 'LSim', 'Foil']):

		plot_violin(group_list, [np.array(v) for v in rt_prd_pos[i].values()], [np.array(v) for v in rt_act_pos[i].values()], '{} New Response'.format(label))
		plot_violin(group_list, [np.array(v) for v in rt_prd_neg[i].values()], [np.array(v) for v in rt_act_neg[i].values()], '{} Old Response'.format(label))

		model_stats[model_name]['{}_rt_err_mean_pos'.format(label)] = np.mean(np.hstack([np.array(v) for v in rt_err_pos[i].values()]))
		model_stats[model_name]['{}_rt_err_sd_pos'.format(label)] = np.std(np.hstack([np.array(v) for v in rt_err_pos[i].values()]))
		model_stats[model_name]['{}_rt_err_mean_neg'.format(label)] = np.mean(np.hstack([np.array(v) for v in rt_err_neg[i].values()]))
		model_stats[model_name]['{}_rt_err_sd_neg'.format(label)] = np.std(np.hstack([np.array(v) for v in rt_err_neg[i].values()]))


def plot_check_chains(chains, model_name):
	z = [key for key in chains.keys() if 'z_' not in key and 'ypred' not in key and 'trial' not in key]
	for key in z:
		for i in range(6):
			this = np.array(chains.get(key))[i,:].flatten()
			plt.figure()
			plt.title(key)
			plt.plot(np.arange(len(this)), this)
			plt.savefig('../chains/{}/{}_chain_{}.png'.format(model_name, key, i+1))
			plt.close()

def estimate_subj_params(model_name, f):
	data = defaultdict(list)
	
	subj_ids = f.get("data").get("subjList")

	n_subj = int(np.array(f.get("data").get("nSubjects"))[0][0])
	group_list = np.array(f.get("data").get("groupList")).T.flatten()

	for i in range(n_subj):
		subj = i+1	
		data['group'].append(group_list[i])
		data['subj'].append(subj_ids[i])

		for param in ['alpha', 'beta', 'tau']:
			subj_data = np.array(f.get("chains").get("{}subj_{}".format(param, subj))).flatten()
			data[param].append(np.mean(subj_data))
			if param != 'alpha':
				print(param)
				print(np.array(f.get("chains").get("{}subjsd_{}".format(param,subj))))
				data[param+'_sd_mean'].append(np.mean(np.array(f.get("chains").get("{}subjsd_{}".format(param,subj)))).flatten())
				data[param+'_sd_median'].append(np.median(np.array(f.get("chains").get("{}subjsd_{}".format(param,subj))).flatten()))

		for i, param in enumerate(['Targ', 'HSim', 'LSim', 'Foil']):
			subj_data = np.array(f.get("chains").get("deltasubj_{}_{}".format(subj, i+1))).flatten()
			data['delta_'+param].append(np.mean(subj_data))
			data['delta_'+param+'_sd_mean'].append(np.mean(np.array(f.get("chains").get("deltasubjsd_{}".format(subj))).flatten()))
			data['delta_'+param+'_sd_median'].append(np.median(np.array(f.get("chains").get("deltasubjsd_{}".format(subj))).flatten()))
		

	df = pd.DataFrame(data)
	df.to_csv("subj_vals.csv")

def process_model(file_path):
	with h5py.File(mat_file, 'r') as f:
		model_name = os.path.basename(mat_file).split(".mat")[0]
		print("Processing:",model_name)
		chains = f.get('chains')

		####### Plot check chains
		#plot_check_chains(chains, model_name)
		#sys.exit()
		####### Plot Differences
		print(" Plotting differences...")
		plot_diff(model_name, 'delta_diff_targ', np.array(chains.get("deltaconddiff_1")).flatten(), 0, 0.8, "Delta Target SR vs OR", 'delta_diff_targ.png')
		plot_diff(model_name, 'delta_diff_lure', np.array(chains.get("deltaconddiff_2")).flatten(), 0, 0.8, "Delta Lure SR vs OR", 'delta_diff_lure.png')

		'''
		####### Plot posteriors
		print(" Plotting posteriors...")
		plot_posteriors(model_name, chains)	
		'''

		####### Calculate DIC
		print(" Calculating DIC...")
		model_stats[model_name]['DIC'] = np.mean(np.array(chains.get("deviance")).flatten())
		####### Calculate posterior predictives
		print(" Calculating posterior predictives...")
		#calculate_posterior_predictives(model_name, f)	
		
if __name__ == '__main__':
	import sys
	for mat_file in sorted(glob.glob(os.path.join(storage_dir, "model*.mat"))):
		process_model(mat_file)
	# Now let's save the stats to a CSV
	print("Saving results to CSV...")
	for key in model_stats.keys():
		model_stats[key]['model_name'] = key
	data = list(model_stats.values())

	df = pd.DataFrame(data)
	print(df.head(n=50))
	df.to_csv("results.csv")

	

