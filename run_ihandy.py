import torch 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import scipy.sparse as sp
import pandas as pd 
import torch.utils.data as td 
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F 
# from torch.utils.tensorboard import SummaryWriter
from utils import *
from sklearn import metrics
# from eval_module import run_evaluation,  final_evaluation
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data_dir = '../ihandy_data/concat_data_no_feature.csv'
# df = pd.read_csv(data_dir)
train_data, train_matrix, test_data, test_matrix, user_num, item_num, appropriate_users = preprocess_data(data_dir)

params = {
    'batch_size': 256,
    'embedding_dim': 8,
    'hidden_dim': 16,
    'N': 5, # memory size for state_repr
    'ou_noise': True,
    'value_lr': 1e-5,
    'value_decay': 1e-4,
    'policy_lr': 1e-5,
    'policy_decay': 1e-4,
    'state_repr_lr': 1e-4,
    'state_repr_decay': 1e-3,
    'log_dir': '../logs/final/',
    'gamma': 0.8,
    'min_value': -10,
    'max_value': 10,
    'soft_tau': 1e-2,
    'buffer_size': 1000000
}

class Env():
	def __init__(self, user_item_matrix, user_num, item_num, N, train_matrix=None):
		self.matrix = user_item_matrix
		self.item_count = item_num
		self.memory = np.ones([user_num, N])*item_num
		self.train_matrix = train_matrix

	def reset(self, user_id):
		self.user_id = user_id 
		self.viewed_items = []
		self.related_items = np.argwhere(self.matrix[self.user_id]>0)[:, 1]
		# print('self.user_id', self.user_id)
		# print('self.related_items', len(self.related_items))
		if self.train_matrix is not None:
			train_related_items = np.argwhere(self.train_matrix[self.user_id]>0)[:, 1]
			# print('train_related_items', len(train_related_items))
			self.related_items = np.concatenate((self.related_items, train_related_items))
			# print('self.related_items', len(self.related_items))

		self.num_rele = len(self.related_items)
		self.nonrelated_items = np.random.choice(list(set(range(self.item_count))-set(self.related_items)), self.num_rele)
		self.available_items = np.zeros(self.num_rele*2)
		self.available_items[::2] = self.related_items
		self.available_items[1::2] = self.nonrelated_items

		return torch.tensor([self.user_id]), torch.tensor(self.memory[[self.user_id], :])

	def step(self, action, action_emb=None, buffer=None):
		initial_user = self.user_id 
		initial_memory = self.memory[[initial_user], :]
		reward = float(to_np(action)[0] in self.related_items)
		self.viewed_items.append(to_np(action)[0])
		if reward: 
			if len(action) == 1:
				self.memory[self.user_id] = list(self.memory[self.user_id][1:])+[to_np(action)[0]]

		if len(self.viewed_items) == len(self.related_items):
			done = 1
		else:
			done = 0

		if buffer is not None:
			buffer.push(np.array([initial_user]), np.array(initial_memory), to_np(action_emb)[0], np.array([reward]), np.array([self.user_id]), self.memory[[self.user_id], :], np.array([done])
				)

		return torch.tensor([self.user_id]), torch.tensor(self.memory[[self.user_id], :]), reward, done 



		
class Actor_DRR(nn.Module):
	def __init__(self, embedding_dim, hidden_dim):
		super().__init__()

		self.layers = nn.Sequential(
			nn.Linear(embedding_dim*3, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, embedding_dim)
			)
		self.initialize()

	def initialize(self):
		for layer in self.layers:
			if isinstance(layer, nn.Linear):
				nn.init.kaiming_uniform_(layer.weight)

	def forward(self, state):
		return self.layers(state)

	def get_action(self, user, memory, state_repr, action_emb, items, return_scores=False):
		state = state_repr(user, memory)
		# print('state_repr.item_embeddings(items).unsqueeze(0)', state_repr.item_embeddings(items).unsqueeze(0).shape)
		scores = torch.bmm(state_repr.item_embeddings(items).unsqueeze(0), action_emb.T.unsqueeze(0)).squeeze(0)
		if return_scores:
			return scores, torch.gather(items, 0, scores.argmax(0))
		else:
			return torch.gather(items, 0, scores.argmax(0))

class Critic_DRR(nn.Module):
	def __init__(self, state_repr_dim, action_emb_dim, hidden_dim):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(state_repr_dim+action_emb_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
			)
		self.initialize()

	def initialize(self):
		for layer in self.layers:
			if isinstance(layer, nn.Linear):
				nn.init.kaiming_uniform_(layer.weight)

	def forward(self, state, action):
		x = torch.cat([state, action], 1)
		x = self.layers(x)
		# x = torch.sigmoid(x)
		return x 


class State_Repr_Module(nn.Module):
	def __init__(self, user_num, item_num, N, embedding_dim, hidden_dim):
		super().__init__()
		self.user_embeddings = nn.Embedding(user_num, embedding_dim)
		self.item_embeddings = nn.Embedding(item_num+1, embedding_dim, padding_idx=int(item_num))
		self.drr_ave = torch.nn.Conv1d(in_channels=N, out_channels=1, kernel_size=1)
		self.initialize()


	def initialize(self):
		nn.init.normal_(self.user_embeddings.weight, std=0.01)
		nn.init.normal_(self.item_embeddings.weight, std=0.01)
		self.item_embeddings.weight.data[-1].zero_()
		nn.init.uniform_(self.drr_ave.weight)
		self.drr_ave.bias.data.zero_()

	def forward(self, user, memory):
		user_embedding = self.user_embeddings(user.long())
		item_embeddings = self.item_embeddings(memory.long())
		# print('item_embeddings.shape', item_embeddings.shape)
		drr_ave = self.drr_ave(item_embeddings).squeeze(1)
		return torch.cat((user_embedding, user_embedding*drr_ave, drr_ave), 1)



def ddpg_update(training_env, step=0, batch_size=params['batch_size'], gamma=params['gamma'], min_value=params['min_value'], max_value=params['max_value'], soft_tau=params['soft_tau']):
	beta = get_beta(step)
	user, memory, action, reward, next_user, next_memory, done = replay_buffer.sample(batch_size, beta)
	user = torch.FloatTensor(user)
	memory = torch.FloatTensor(memory)
	action = torch.FloatTensor(action)
	reward = torch.FloatTensor(reward)
	next_user = torch.FloatTensor(next_user)
	next_memory = torch.FloatTensor(next_memory)
	done = torch.FloatTensor(done)
	state = state_repr(user, memory)

	policy_loss = value_net(state, policy_net(state))
	policy_loss = - policy_loss.mean()
	next_state = state_repr(next_user, next_memory)
	next_action = target_policy_net(next_state)
	target_value = target_value_net(next_state, next_action.detach())
	expected_value = reward + (1.0 - done)*gamma*target_value
	expected_value = torch.clamp(expected_value, min_value, max_value)

	value = value_net(state, action)
	value_loss = value_criterion(value, expected_value.detach())

	state_repr_optimizer.zero_grad()
	
	policy_optimizer.zero_grad()
	policy_loss.backward(retain_graph=True)
	policy_optimizer.step()

	value_optimizer.zero_grad()
	value_loss.backward(retain_graph=True)
	value_optimizer.step()

	state_repr_optimizer.step()

	for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
		target_param.data.copy_(target_param.data*(1.0-soft_tau)+param.data*soft_tau)

	for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
		target_param.data.copy_(target_param.data*(1.0-soft_tau)+param.data*soft_tau)

	return to_np(value_loss), to_np(policy_loss)


def run_evaluation(net, state_repr, training_env_memory, loader):
    hits = []
    dcgs = []
    test_env = Env(test_matrix, user_num, item_num, params['N'])
    test_env.memory = training_env_memory.copy()
    user, memory = test_env.reset(int(to_np(next(iter(loader))['user'])[0]))
    for batch in loader:
        action_emb = net(state_repr(user, memory))
        scores, action = net.get_action(
            batch['user'], 
            torch.tensor(test_env.memory[to_np(batch['user']).astype(int), :]), 
            state_repr, 
            action_emb,
            batch['item'].long(), 
            return_scores=True
        )
        user, memory, reward, done = test_env.step(action)

        _, ind = scores[:, 0].topk(10)
        predictions = torch.take(batch['item'], ind).cpu().numpy().tolist()
        actual = batch['item'][0].item()
        hits.append(hit_metric(predictions, actual))
        dcgs.append(dcg_metric(predictions, actual))
        
    return np.mean(hits), np.mean(dcgs)



def run_evaluation_2(net, state_repr, training_env_memory):
    hits = []
    dcgs = []
    auc_list = []
    precison_list = []
    recall_list = []
    ap_list = []
    score_list = []
    item_list = []
    test_env = Env(test_matrix, user_num, item_num, params['N'], train_matrix=train_matrix)
    test_env.memory = training_env_memory.copy()
    test_users = np.where(test_matrix.sum(1)>5)[0]
    print('\n test users number', len(test_users))
    for u in test_users:
        # print('test user', u)
        user, memory = test_env.reset(u)
        action_emb = net(state_repr(user, memory))
        items = torch.tensor(test_env.available_items).long()
        scores, action = net.get_action(
        	user, 
            memory, 
            state_repr, 
            action_emb,
            items, 
            return_scores=True
        )
        _, ind = scores[:, 0].topk(len(scores))
        predictions = torch.take(items, ind).cpu().numpy().tolist()
        # print('predictions', predictions)
        actual = test_env.related_items
        actual_num = len(actual)
        # print('actual', actual)
        unrank_pred_true = [1 if to_np(item) in actual else 0 for item in items]
        pred_true = [1 if pred in actual else 0 for pred in predictions]
        ap = average_precision(pred_true)
        ap_list.append(ap)
        pred_true_10 = pred_true[:actual_num]
        # print('pred_true_10', pred_true_10)
        precison = np.sum(pred_true_10)/len(pred_true_10)
        # print('precison', precison)
        recall = np.sum(pred_true_10)/actual_num
        precison_list.append(precison)
        recall_list.append(recall)
        item_list.extend(unrank_pred_true)
        score_list.extend(to_np(scores.squeeze(1)))

        # print('\n scores', to_np(scores.squeeze(1)))
        # print('\n actual', actual)
        # print('\n items', to_np(items))
        # print('\n unrank_pred_true', unrank_pred_true)
        # print('\n pred_true', pred_true)
        # print('\n len(item_list)', len(item_list))
        # print('\n len(score_list)', len(score_list))
        # print('\n item_list', item_list)
        # print('\n score_list', score_list)
    fpr, tpr, thresholds = metrics.roc_curve(item_list, score_list, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    return np.mean(precison_list), np.mean(recall_list), auc_score, np.mean(ap)




# valid_dataset = EvalDataset(
#     np.array(test_data)[np.array(test_data)[:, 0] == 4185], 
#     item_num, 
#     test_matrix)
# valid_loader = td.DataLoader(valid_dataset, batch_size=100, shuffle=False)
# full_dataset = EvalDataset(np.array(test_data), item_num, test_matrix)
# full_loader = td.DataLoader(full_dataset, batch_size=100, shuffle=False)




# torch.manual_seed(2)
state_repr = State_Repr_Module(user_num, item_num, params['N'], params['embedding_dim'], params['hidden_dim'])
policy_net = Actor_DRR(params['embedding_dim'], params['hidden_dim'])
value_net = Critic_DRR(params['embedding_dim']*3, params['embedding_dim'], params['hidden_dim'])
replay_buffer = Prioritized_Buffer(params['buffer_size'])


target_policy_net = Actor_DRR(params['embedding_dim'], params['hidden_dim'])
target_value_net = Critic_DRR(params['embedding_dim']*3, params['embedding_dim'], params['hidden_dim'])

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
	target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
	target_param.data.copy_(param.data)

value_criterion = nn.MSELoss()
value_optimizer = optim.Adam(value_net.parameters(), lr=params['value_lr'], weight_decay=params['value_decay'])
policy_optimizer = optim.Adam(policy_net.parameters(), lr=params['policy_lr'], weight_decay=params['policy_decay'])
state_repr_optimizer = optim.Adam(state_repr.parameters(), lr=params['state_repr_lr'], weight_decay=params['state_repr_decay'])



train_env = Env(train_matrix, user_num, item_num, params['N'])
test_env = Env(test_matrix, user_num, item_num, params['N'])
users = np.random.permutation(appropriate_users)
ou_noise = OUNoise(params['embedding_dim'], decay_period=10)

hits, dcgs = [], []
hits_all, dcgs_all = [], []
step, best_step = 0, 0
step, best_step, best_step_all = 0, 0, 0

hit_list = []
dcg_list = []
recall_list = []
precison_list = []
ap_list = []
auc_list = []
avg_recall_list = []
cum_reward_list = []
value_loss_list = []
policy_loss_list = []


epoch_num = 20000
for _ in tqdm.tqdm(range(epoch_num)):
	u = np.random.choice(users)
# for u in tqdm.tqdm(users):
	user, memory = train_env.reset(u)
	if params['ou_noise']:
		ou_noise.reset()
	cum_reward = 0
	time = 1
	for t in range(int(train_matrix[u].sum())):
		action_emb = policy_net(state_repr(user, memory))
		if params['ou_noise']:
			action_emb = ou_noise.get_action(action_emb.detach().cpu().numpy()[0], t)

		items = torch.tensor([item for item in train_env.available_items if item not in train_env.viewed_items]).long()
		scores, action = policy_net.get_action(
            user, 
            torch.tensor(train_env.memory[to_np(user).astype(int), :]), 
            state_repr, 
            action_emb,
            items, 
            return_scores=True
        )

		user, memory, reward, done = train_env.step(
			action, 
			action_emb, 
			buffer=replay_buffer)
		cum_reward += reward
		if len(replay_buffer) > params['batch_size']:
			if step % 100 == 0 and step > 0:
				ddpg_update(train_env, step=step)


		if step % 5000 == 0 and step > 0:
			precison, recall, auc, ap = run_evaluation_2(policy_net, state_repr, train_env.memory)
			precison_list.append(precison)
			recall_list.append(recall)
			auc_list.append(auc)
			ap_list.append(ap)
			print('\n step', step)
			print('\n test:  precision@10, AUC, MAP', np.round(precison, decimals=3), np.round(auc, decimals=3), np.round(ap, decimals=3))


		step += 1
		time += 1
	ratio = cum_reward/time
	# print('user, ratio', to_np(user)[0], np.round(ratio, decimals=2))
	cum_reward_list.append(ratio)


np.save('../results/ihandy_precision_list.npy', precison_list)
np.save('../results/ihandy_ap_list.npy', ap_list)
np.save('../results/ihandy_auc_list.npy', auc_list)




plt.figure(figsize=(6,4))
plt.plot(cum_reward_list)
plt.xlabel('training iteration', fontsize=12)
plt.ylabel('cum reward', fontsize=12)
plt.tight_layout()
plt.savefig('../results/run_ihandy_cum_reward'+'.png', dpi=100)
plt.show()



plt.figure(figsize=(6,4))
plt.plot(precison_list)
plt.xlabel('training iteration', fontsize=12)
plt.ylabel('precison', fontsize=12)
plt.tight_layout()
plt.savefig('../results/run_ihandy_precison'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(6,4))
plt.plot(recall_list)
plt.xlabel('training iteration', fontsize=12)
plt.ylabel('recall', fontsize=12)
plt.tight_layout()
plt.savefig('../results/run_ihandy_recall'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(auc_list)
plt.xlabel('training iteration (x5000)', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.tight_layout()
plt.savefig('../results/run_ihandy_auc'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(6,4))
plt.plot(moving_average(ap_list))
plt.xlabel('training iteration (x5000)', fontsize=12)
plt.ylabel('MAP', fontsize=12)
plt.tight_layout()
plt.savefig('../results/run_ihandy_ap'+'.png', dpi=100)
plt.show()

ap_list = np.load('../results/ihandy_ap_list.npy')
auc_list = np.load('../results/ihandy_auc_list.npy')

plt.figure(figsize=(6,4))
plt.plot(moving_average(auc_list, n=5), label='AUC')
plt.plot(moving_average(ap_list, n=10), label='MAP')
plt.title('iHandy', fontsize=12)
plt.xlabel('training iteration (x5000)', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.legend(loc=2, fontsize=12)
plt.tight_layout()
plt.savefig('../results/run_ihandy_performance'+'.png', dpi=100)
plt.show()

