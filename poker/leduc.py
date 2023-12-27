# step 1: doing suits (the slow way)
# 2: more rounds of betting
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
    deck_size = num_ranks * 2
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

    # used to calculate chance outcomes
    def _p_chance(rnd, board, i, j): 
        if rnd == 0: # pre flop
            if i == j:
                return 2 * hand_combinations
            return 4 * hand_combinations
        if i == board and j == board: # having same card as opponent affects probabilities
            return 0
        elif i == board or j == board or i == j:
            return 4 * rollout_combinations
        return 8 * rollout_combinations

    def _build_terminal(rnd, board, value, previous_seq): #takes in a value function and the player ranks
        for i in range(num_ranks):
            for j in range(num_ranks):
                payoff.append((previous_seq[0][i], previous_seq[1][j],
                               _p_chance(rnd, board, i, j) *
                               (value(i, j) + payoff_shift))) # stores p1 index, p2 index, and value * P(outcome)
                if all_negative:
                    payoff_p1.append((
                        previous_seq[0][i], previous_seq[1][j],
                        _p_chance(rnd, board, i, j) * (-value(i, j) + \
                                                     payoff_shift)))

    def _build_fold(rnd, board, who_folded, win_amount, previous_seq):
        if who_folded == 1:
            win_amount = -win_amount

        def _value(i, j):
            return win_amount

        _build_terminal(rnd, board, _value, previous_seq)

    def _build_showdown(rnd, board, win_amount, previous_seq): # returns int based on who won
        def _value(i, j): # negative when P1 wins
            if i == board:
                return -win_amount
            elif j == board:
                return win_amount
            elif i > j:
                return -win_amount
            elif j > i:
                return win_amount
            return 0

        _build_terminal(rnd, board, _value, previous_seq)

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
        for i in range(num_ranks):
            parent[actor].append(previous_seq[actor][i])
            begin[actor].append(next_s[actor])
            next_s[actor] += num_actions
            end[actor].append(next_s[actor])
            for j in range(num_ranks):
                # we can ignore reach -- this is meant to encode probabilities but we can do that in payoffs directly
                reach.append((actor, info_set + i, previous_seq[opponent][j],
                              _p_chance(rnd, board, i, j)))

        # this generates the previous_seq array for the next iteration
		# e.g maybe idx 0 is fold, 1 is call/check, etc.
		# and we index through the begin/end array by rank (i) and by action (idx)
		# begin is [infoset0action0_rank0, infoset0action0_rank1, ..., infoset0action1rank0, ...,infoset1action0rank0, etc.]
        def _pn(idx):
            t = [begin[actor][info_set + i] + idx for i in range(num_ranks)]
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
        elif rnd + 1 < 2:  #  call and deal board card
            for board in range(num_ranks):
                _build(rnd + 1, board, 0, 0, pot, _pn(action))
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

    previous_seq = ([0] * num_ranks, [0] * num_ranks)
    # rnd, board, actor, num_bets, pot, previous_seq -- _build is recursive
    _build(0, -1, 0, 0, [1, 1], previous_seq)

    if integer:
        payoff_matrix = lil_matrix((next_s[0], next_s[1]), dtype=int)
    else:
        payoff_matrix = lil_matrix((next_s[0], next_s[1]))
    for i, j, payoff_value in payoff:
        payoff_matrix[i, j] += payoff_value
    reach_matrix = (lil_matrix((len(begin[0]), next_s[1])), lil_matrix(
        (len(begin[1]), next_s[0])))
    for player, infoset, opponent_seq, prob in reach:
        reach_matrix[player][infoset, opponent_seq] += prob

    if all_negative:
        if integer:
            payoff_p1_matrix = lil_matrix((next_s[0], next_s[1]), dtype=int)
        else:
            payoff_p1_matrix = lil_matrix((next_s[0], next_s[1]))
        for i, j, payoff_value in payoff_p1:
            payoff_p1_matrix[i, j] += payoff_value
        return efg.ExtensiveFormGame(
            'Leduc-%d' % num_ranks,
            payoff_matrix,
            begin,
            end,
            parent,
            prox_infoset_weights=prox_infoset_weights,
            prox_scalar=prox_scalar,
            reach=reach_matrix,
            B=payoff_p1_matrix,
            offset=2 * payoff_shift * (deck_size * (deck_size - 1) *
                                       (deck_size - 2)))
    else:
        return efg.ExtensiveFormGame(
            'Leduc-%d' % num_ranks,
            payoff_matrix,
            payoff_matrix,
            begin,
            end,
            parent,
            prox_infoset_weights=prox_infoset_weights,
            prox_scalar=prox_scalar,
            reach=reach_matrix)
