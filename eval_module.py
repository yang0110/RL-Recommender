import numpy as np 
# from env_module import Env 
import torch 
from utils import to_np, hit_metric, dcg_metric
from sklearn import metrics

def run_evaluation(user, test_env, policy_net, state_repr, training_env_memory):
	test_env.memory = training_env_memory.copy()
	user, memory = test_env.reset(user)
	# print('user', user)
	action_emb = policy_net(state_repr(user, memory))
	item_pool = torch.tensor([item for item in test_env.available_items]).long()
	scores, action, predictions = policy_net.get_action(
		user, 
		torch.tensor(test_env.memory[to_np(user).astype(int), :]), 
		state_repr, 
		action_emb,
		item_pool,
		return_scores=True
		)
	# cutoff = np.min([len(scores), 10])
	# _, ind = scores[:, 0].topk(cutoff)
	# predictions = torch.take(item_pool, ind).cpu().numpy().tolist()
	actual = test_env.related_items
	actual_num = len(actual)
	# print('actual num', actual_num)
	pred_true = []
	for item in predictions:
		if item in actual:
			pred_true.append(1)
		else:
			pred_true.append(0)
	fpr, tpr, thresholds = metrics.roc_curve(pred_true, scores, pos_label=1)
	auc_score = metrics.auc(fpr, tpr)
	precision = np.sum(pred_true)/len(pred_true)
	recall = np.sum(pred_true)/actual_num
	# print('predictions', predictions)
	# print('actual', actual)
	# print('precision, recall', np.round(precision, decimals=2), np.round(recall, decimals=2))
	return precision, recall, auc_score

def final_evaluation(test_env, N, policy_net, state_repr, train_env_memory, train_env_user_history):
	# test_env.memory = train_env_memory
	test_users = list(test_env.user_history.keys())
	# print('test_users', test_users)
	auc_list = []
	for u in test_users:
		# print('user_id', user_id)
		user, memory = test_env.reset(u)
		test_env.available_items = test_env.user_pic_pool[u]
		# print('user_history', train_env_user_history)
		# print('user', to_np(user)[0])
		# print('history', train_env_user_history[to_np(user)[0]])
		history = train_env_user_history[to_np(user)[0]][-N:]
		length = len(history)
		# print('test_env.memory[to_np(user)[0]]', test_env.memory[to_np(user)[0]])
		test_env.memory[to_np(user)[0]] = list(test_env.memory[to_np(user)[0]])[length:]+history
		# print('test_env.memory[to_np(user)[0]]', test_env.memory[to_np(user)[0]])

		action_emb = policy_net(state_repr(user, memory))
		item_pool = torch.tensor([item for item in test_env.available_items]).long()
		# print('item_pool', item_pool)
		scores, action = policy_net.get_action(
			user, 
			torch.tensor(test_env.memory[to_np(user).astype(int), :]), 
			state_repr, 
			action_emb,
			item_pool,
			return_scores=True
			)
		actual = test_env.related_items
		# print('actual', actual)
		# print('item_pool', item_pool)
		actual_num = len(actual)
		if actual_num == 0:
			pass
		else:
			pred_true = [1 if to_np(item) in actual else 0 for item in item_pool]
			# print('pred_true', pred_true)
			if np.sum(pred_true) == len(pred_true):
				pass
			else:
				fpr, tpr, thresholds = metrics.roc_curve(pred_true, to_np(scores), pos_label=1)
				auc_score = metrics.auc(fpr, tpr)
				auc_list.append(auc_score)
		# print('auc_list', auc_list)
	return np.mean(auc_list)



