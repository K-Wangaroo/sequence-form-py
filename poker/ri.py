# 1: more rounds of betting
# 2: doing suits (the slow way)
# 3: doing suits (the fast way)

from extensive_form_game import extensive_form_game as efg
from scipy.sparse import lil_matrix


def init_efg(num_ranks=3,
			 prox_infoset_weights=False,
			 prox_scalar=-1,
			 integer=False,
			 all_negative=False,
			 num_raise_sizes=1,
			 max_bets=2):
	assert num_ranks >= 2
	
	num_suits = 4
	deck_size = num_ranks * num_suits # change this later when suits change
	if integer:
		hand_combinations = deck_size - 2
		rollout_combinations = 1
	else:
		hand_combinations = 1.0 / float(deck_size * (deck_size - 1))
		rollout_combinations = 1.0 / float(deck_size * (deck_size - 1) * \
									 (deck_size - 2)) 

	parent = ([], [])
	begin = ([], [])
	end = ([], [])
	payoff = []
	reach = []
	next_s = [1, 1]
	# below only used for outputting a payoff-shifted constant-sum game when
	# all_negative is True
	payoff_shift = 0
	if all_negative:
		payoff_shift = -15
		payoff_p1 = []
	# ignore this ^^
		

	def get_card_id(card):
		return card[0] * 4 + card[1]

	# number = (rank * 4) + suit
	from itertools import product
	def _build_terminal(rnd, board, value, previous_seq):

		for i in product(range(num_ranks), range(4)):
			for j in product(range(num_ranks), range(4)):
				payoff.append((previous_seq[0][get_card_id(i)], previous_seq[1][get_card_id(j)],
							   _p_chance(rnd, board, i, j) *
							   (value(i, j) + payoff_shift)))
				if all_negative:
					payoff_p1.append((
						previous_seq[0][get_card_id(i)], previous_seq[1][get_card_id(j)],
						_p_chance(rnd, board, i, j) * (-value(i, j) + \
													 payoff_shift)))

	# used to calculate chance outcomes
	def _p_chance(rnd, board, i, j):
		# i = (rank, suit)
		# j = (rank, suit)

		# unique cards is a set keeping track of all unique cards dealt
		unique_cards = set()
		unique_cards.add(i)
		unique_cards.add(j)

		if rnd == 0:
			# check if each player has a unique card. If not, return 0
			if len(unique_cards) != 2:
				return 0
			return 1.0 / (deck_size * (deck_size - 1))
		
		# add first board card to unique cards
		unique_cards.add(board[0])
		if rnd == 1:
			# check if each player card and first board card is unique. If not, return 0
			if len(unique_cards) != 3:
				return 0
			return 1.0 / (deck_size * (deck_size - 1) * (deck_size - 2))
		
		unique_cards.add(board[1])
		if rnd == 2:
			# check if each player card and first two board cards is unique. If not, return 0
			if len(unique_cards) != 4:
				return 0
			return 1.0 / (deck_size * (deck_size - 1) * (deck_size - 2) * (deck_size - 3))
		
		print("ERROR: _p_chance() called with invalid rnd value")

	def _build_terminal(rnd, board, value, previous_seq): #takes in a value function and the player ranks
		for i in product(range(num_ranks), range(num_suits)):
			for j in product(range(num_ranks), range(num_suits)):
				payoff.append((previous_seq[0][get_card_id(i)], previous_seq[1][get_card_id(j)],
							   _p_chance(rnd, board, i, j),
							   (value(i, j) + payoff_shift))) # stores p1 index, p2 index, and value * P(outcome)
				if all_negative:
					payoff_p1.append((
						previous_seq[0][i], previous_seq[1][j],
						_p_chance(rnd, board, i, j) * (-value(i, j), \
													 payoff_shift)))

	def _build_fold(rnd, board, who_folded, win_amount, previous_seq):
		if who_folded == 1:
			win_amount = -win_amount

		def _value(i, j):
			return win_amount

		_build_terminal(rnd, board, _value, previous_seq)

	def _build_showdown(rnd, board, win_amount, previous_seq):
        
        # if i wins, return negative win amount
        # if j wins, return positive win amount
		def _value(i, j):
			handi = board + [i]
			handj = board + [j]
			
			ranki, details1 = evaluate_hand(handi)
			rankj, details2 = evaluate_hand(handj)


            # i wins
			if ranki > rankj:
				return - win_amount
            # j wins
			if rankj > ranki:
				return win_amount
			else:
				for card1, card2 in zip(details1, details2):
                    # i wins
					if card1 > card2:
						return - win_amount
                    # j wins
					elif card1 < card2:
						return win_amount
                # tie
					return 0

		_build_terminal(rnd, board, _value, previous_seq)



	def evaluate_hand(hand):
		ranks = sorted([card[0] for card in hand], reverse=True)

        # Check for straight flush
		if (ranks[0] - ranks[1] == ranks[1] - ranks[2] == 1) and all(card[1] == hand[0][1] for card in hand): 
			return 6, [ranks[0]]

        # Check for three of a kind
		if ranks[0] == ranks[1] == ranks[2]:
			return 5, [ranks[0]]

        # Check for straight
		if (ranks[0] - ranks[1] == ranks[1] - ranks[2] == 1): 
			return 4, [ranks[0]]

        # Check for flush
		if all(card[1] == hand[0][1] for card in hand):
			return 3, [ranks]

        # Check for pair
		for i in range(2):
			if ranks[i] == ranks[i + 1]:
				if i == 0:
					other = 2
				else:
					other = 0
				return 2, [ranks[i], ranks[other]]

        # High card
		return 1, ranks

	def _build(rnd, board, actor, num_bets, pot, previous_seq): 
		opponent = 1 - actor
		facing = pot[opponent] - pot[actor] # the bet we are facing
		# e.g. if our opponent has put more than we did in the pot that means we are facing a bet
		pot_actor = pot[actor]
		num_actions = (facing > # call/check, can fold if facing > 0, can raise if allowed (let's just do one size)
					   0) + 1 + (num_bets < max_bets) * num_raise_sizes 
		action = 0 # action is used for indexing
		first_action = actor == 0 and num_bets == 0 # indicator(first action)

		# this is basically indexing the same way that kuhn did:
		# e.g. after first iteration for player 0, parent = [0,0,0], start = [1,3,5], end = [3,5,7]
		# info_set tracks the START of the next set of actions
		info_set = len(begin[actor])
		for i in range(deck_size):
			parent[actor].append(previous_seq[actor][i])
			begin[actor].append(next_s[actor])
			next_s[actor] += num_actions
			end[actor].append(next_s[actor])
			# for j in range(num_ranks):
			#     # we can ignore reach -- this is meant to encode probabilities but we can do that in payoffs directly
			#     reach.append((actor, info_set + i, previous_seq[opponent][j],
			#                   _p_chance(rnd, board, i, j)))

		# this generates the previous_seq array for the next iteration
		# e.g maybe idx 0 is fold, 1 is call/check, etc.
		# and we index through the begin/end array by rank (i) and by action (idx)
		# begin is [infoset0action0_rank0, infoset0action0_rank1, ..., infoset0action1rank0, ...,infoset1action0rank0, etc.]
		def _pn(idx):
			t = [begin[actor][info_set + i] + idx for i in range(deck_size)]
			if actor == 0:
				return (t, previous_seq[1])
			return (previous_seq[0], t)

		# building actions: we recursively generate possible all possible future actions
		if facing > 0:
			_build_fold(rnd, board, actor, pot[actor], _pn(action))
			action += 1

		pot[actor] = pot[opponent]
		if first_action:  #  check
			_build(rnd, board, opponent, 0, pot, _pn(action))
		elif rnd + 1 < 3:  #  call and deal board card
			if rnd == 0:
				for rank in range(num_ranks):
					for suit in range(num_suits):
						_build(rnd + 1, [(rank, suit), -1], 0, 0, pot, _pn(action))		
			else: # round = 1
				for rank in range(num_ranks):
					for suit in range(num_suits):
						_build(rnd + 1, [board[0], (rank,suit)], 0, 0, pot, _pn(action))
		else:  #  call and showdown
			_build_showdown(rnd, board, pot[actor], _pn(action))
		action += 1

		# generate bets
		if num_bets < max_bets:
			if rnd == 0:
				init_raise_size = 2
			else:
				init_raise_size = 4
			for raise_amt in [
					init_raise_size * raise_size
					for raise_size in range(1, num_raise_sizes + 1)
			]:
				pot[actor] = pot[opponent] + raise_amt
				_build(rnd, board, opponent, 1 + num_bets, pot, _pn(action))
				action += 1

		pot[actor] = pot_actor
		print(len(parent[0]))

	previous_seq = ([0] * deck_size, [0] * deck_size)
	# rnd, board, actor, num_bets, pot, previous_seq -- _build is recursive
	# each card is enumerated as an ID
	_build(0, [-1, -1], 0, 0, [1, 1], previous_seq)

	payoff_matrix = [None, None]
	# payoff matrix is negative for player 1 winning, should pass in the same one
	# we construct it so that it is positive for each player for them winning
	# and flip them at the end
	def utility_payoff(change, alpha, initial = 100):
		return ((initial + change)**alpha)/alpha - (initial**alpha)/alpha
	alpha = [1,1]

	if integer:
		payoff_matrix[0] = lil_matrix((next_s[0], next_s[1]), dtype=int)
		payoff_matrix[1] = lil_matrix((next_s[0], next_s[1]), dtype=int)
	else:
		payoff_matrix[0] = lil_matrix((next_s[0], next_s[1]))
		payoff_matrix[1] = lil_matrix((next_s[0], next_s[1]))
	#payoff value is negative for p1 winning, p2 for p2 winning by default
	for i, j, chance, payoff_value in payoff:
		payoff_matrix[0][i, j] += -1 * chance * utility_payoff(-payoff_value,alpha[0])
		payoff_matrix[1][i, j] += chance * utility_payoff(payoff_value, alpha[1])
		diff = payoff_matrix[1][i,j] - payoff_matrix[0][i,j]
		if diff:
			print(diff)
	#sign adjust p0 back
	# payoff_matrix[0] = payoff_matrix[0] * -1
	# #unused
	# reach_matrix = (lil_matrix((len(begin[0]), next_s[1])), lil_matrix(
	#     (len(begin[1]), next_s[0])))
	# for player, infoset, opponent_seq, prob in reach:
	#     reach_matrix[player][infoset, opponent_seq] += prob

	print("passing in")

	if all_negative:
		if integer:
			payoff_p1_matrix = lil_matrix((next_s[0], next_s[1]), dtype=int)
		else:
			payoff_p1_matrix = lil_matrix((next_s[0], next_s[1]))
		for i, j, payoff_value in payoff_p1:
			payoff_p1_matrix[i, j] += payoff_value
		return efg.ExtensiveFormGame(
			'RI-%d' % num_ranks,
			payoff_matrix[0],
			payoff_matrix[1],
			begin,
			end,
			parent,
			prox_infoset_weights=prox_infoset_weights,
			prox_scalar=prox_scalar,
			reach=None, # changed because unused
			B=payoff_p1_matrix,
			offset=2 * payoff_shift * (deck_size * (deck_size - 1) *
									   (deck_size - 2)))
	else:
		return efg.ExtensiveFormGame(
			'RI-%d' % num_ranks,
			payoff_matrix[0],
			payoff_matrix[1],
			begin,
			end,
			parent,
			prox_infoset_weights=prox_infoset_weights,
			prox_scalar=prox_scalar,
			reach=None) # changed because unused
