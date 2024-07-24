import numpy as np

class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()
    def onehot2index(self,array):
        return np.argmax(array,axis=0)
    def select_w_p(self,p):
        if np.random.random() < p:
            return 1
        else:
            return 0

class PolicyKuhn(Policy):
    def __init__(self,alpha,beta,gamma,eta,xi):
        super(PolicyKuhn, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.xi = xi

    def get_return_complex(self, probs):
        # the average of 6 possible dealt_cards combo
        # the game tree of playe 1 is not simplified
        # xi: player 2 bet J
        # eta: player 2 call Q
        xi = self.xi
        eta = self.eta
        # JQ
        r1 = (1 - probs[0]) * (-1) + probs[0] * (eta * (-2) + (1 - eta) * 1)  # check check;bet call;bet fold
        # JK
        r2 = (1 - probs[0]) * (probs[3] * (-2) + (1 - probs[3]) * (-1)) + probs[0] * (
            -2)  # check bet call;check bet fold;bet call
        # QJ
        r3 = (1 - probs[1]) * (xi * (probs[4] * 2 - (1 - probs[4])) + (1 - xi)) + probs[
            1]  # check bet call;check bet fold;check check;bet fold
        # QK
        r4 = (1 - probs[1]) * (probs[4] * (-2) + (1 - probs[4]) * (-1)) + probs[1] * (
            -2)  # check bet call;check bet fold;bet call
        # KJ
        r5 = (1 - probs[2]) * (xi * (probs[5] * 2 - (1 - probs[5])) + (1 - xi)) + probs[
            2]  # check bet call;check bet fold;check check;bet fold
        # KQ
        r6 = (1 - probs[2]) + probs[2] * (eta * (2) + (1 - eta) * 1)  # check check;bet call;bet fold
        return (r1 + r2 + r3 + r4 + r5 + r6) / 6

    def get_return(self, alpha, beta, gamma):
        # the average of 6 possible dealt_cards combo
        # the game tree of playe 1 is simplified by cutting trivial sub-optimal solutions
        # gamma: player 1 bet J
        # beta: player 1 call Q
        # gamma: player 1 bet K
        # xi: player 2 bet J
        # eta: player 2 call Q
        xi = self.xi
        eta = self.eta
        # JQ
        r1 = -(1 - alpha) + alpha * (1 - eta) - 2 * alpha * eta
        # JK
        r2 = -(1 - alpha) - 2 * alpha
        # QJ
        r3 = (1 - xi) + xi * (beta * 2 - (1 - beta))
        # QK
        r4 = -(1 - beta) - 2 * beta
        # KJ
        r5 = (1 - gamma) * (1 - xi) + 2 * ((1 - gamma) * xi) + gamma
        # KQ
        r6 = (1 - gamma) + gamma * (1 - eta + 2 * eta)
        return (r1 + r2 + r3 + r4 + r5 + r6) / 6

    def action(self,obs):
        player = self.onehot2index(obs[:2])
        private_card = self.onehot2index(obs[2:5])
        cur_round = int(np.sum(obs[5:]))
        betting = []
        for i in range(cur_round):
            betting.append(np.argmax(obs[5+i*2:7+i*2],axis=0))

        if player == 0:
            return self.action_p0(player,private_card,cur_round,betting)
        else:
            return self.action_p1(player,private_card,cur_round,betting)

    def action_p0(self,player,private_card,cur_round,betting):
        # round 0
        if cur_round == 0:
            # jack, w/ alpha
            if private_card == 0:
                return self.select_w_p(self.alpha)
            # queen, w/ 0
            elif private_card == 1:
                return 0
            # king, w/ gamma
            elif private_card == 2:
                return self.select_w_p(self.gamma)
            else:
                raise ValueError('private_card not found')
        elif cur_round == 2:
            is_bet_last = betting[1]
            # jack, w/ 0
            if private_card == 0:
                return 0
            # queen, w/ beta
            elif private_card == 1:
                if is_bet_last == 1:
                    return self.select_w_p(self.beta)
                else:
                    ValueError('round error')
            # king, w/ 1
            elif private_card == 2:
                return 1
            else:
                raise ValueError('private_card not found')
        else:
            raise ValueError('round error')

    def action_p1(self,player,private_card,cur_round,betting):
        assert cur_round == 1
        other_player = 1 - player
        is_bet_last = betting[0]
        # round 1 only
        # jack, w/ xi when last pass
        if private_card == 0:
            if is_bet_last == 0:
                return self.select_w_p(self.xi)
            else:
                return 0 
        # queen, w/ 0 eta when last bet
        elif private_card == 1:
            if is_bet_last == 0:
                return 0 
            else:
                return self.select_w_p(self.eta)
        # king, w/ 1
        elif private_card == 2:
            return 1
        else:
            raise ValueError('private_card not found')


class BestResponseKuhn(Policy):
    def __init__(self,alpha,beta,gamma,eta,xi):
        super(BestResponseKuhn, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.xi = xi
        self.get_response()

    def action(self,obs):
        player = self.onehot2index(obs[:2])
        private_card = self.onehot2index(obs[2:5])
        cur_round = int(np.sum(obs[5:]))
        betting = []
        for i in range(cur_round):
            betting.append(np.argmax(obs[5+i*2:7+i*2],axis=0))

        if player == 0:
            return self.response_p0.action_p0(player,private_card,cur_round,betting)
        else:
            return self.response_p1.action_p1(player,private_card,cur_round,betting)

    def get_response(self):
        self.response_p1 = PolicyKuhn(alpha=0,beta=0,gamma=0,eta=1/3,xi=1/3) # NE

        if self.eta <= 1/3 and self.xi <= 1/3:
            if self.eta > self.xi: 
                self.response_p0 = PolicyKuhn(alpha=1,beta=0,gamma=1,eta=1/3,xi=1/3) # s6
            else:
                self.response_p0 = PolicyKuhn(alpha=1,beta=0,gamma=0,eta=1/3,xi=1/3) # s5
        elif self.eta <= 1/3 and self.xi > 1/3:
            self.response_p0 = PolicyKuhn(alpha=1,beta=1,gamma=0,eta=1/3,xi=1/3) # s7
        elif self.eta > 1/3 and self.xi <= 1/3:
            self.response_p0 = PolicyKuhn(alpha=0,beta=0,gamma=1,eta=1/3,xi=1/3) # s2
        elif self.eta > 1/3 and self.xi > 1/3:
            if self.eta > self.xi: 
                self.response_p0 = PolicyKuhn(alpha=0,beta=1,gamma=1,eta=1/3,xi=1/3) # s4
                # self.response_p0 = PolicyKuhn(alpha=0,beta=1/3,gamma=0,eta=1/3,xi=1/3) # s4
            else:
                self.response_p0 = PolicyKuhn(alpha=0,beta=1,gamma=0,eta=1/3,xi=1/3) # s3
        else:
            raise ValueError('incorrect policy vector')

    def get_br_parameters(self):
        return np.array([self.response_p0.alpha,self.response_p0.beta,self.response_p0.gamma,self.response_p1.eta,self.response_p1.xi])

def get_policy_by_vector(policy_vector,is_best_response):
    if is_best_response:
        return BestResponseKuhn(policy_vector[0],policy_vector[1],policy_vector[2],policy_vector[3],policy_vector[4])
    else:
        return PolicyKuhn(policy_vector[0],policy_vector[1],policy_vector[2],policy_vector[3],policy_vector[4])


if __name__ == '__main__':

    test_policy = Policy0(alpha=1,beta=1,gamma=0,eta=0,xi=1)

    # test p0
    info_state = [1,0,1,0,0,0,0,0,0,0,0]
    action = test_policy.action(info_state)
    print ('info_state',info_state,'action',action, 'shoule be 1')
    info_state = [1,0,0,1,0,0,0,0,0,0,0]
    action = test_policy.action(info_state)
    print ('info_state',info_state,'action',action, 'shoule be 0')
    info_state = [1,0,0,0,1,0,0,0,0,0,0]
    action = test_policy.action(info_state)
    print ('info_state',info_state,'action',action, 'shoule be 0')
    info_state = [1,0,0,1,0,0,0,1,1,0,0]
    action = test_policy.action(info_state)
    print ('info_state',info_state,'action',action, 'shoule be 1')

    # test p1
    info_state = [0,1,1,0,0,1,0,0,0,0,0]
    action = test_policy.action(info_state)
    print ('info_state',info_state,'action',action, 'shoule be 1')
    info_state = [0,1, 0,1,0, 0,1, 0,0, 0,0]
    action = test_policy.action(info_state)
    print ('info_state',info_state,'action',action, 'shoule be 0')
    info_state = [0,1, 0,0,1, 0,1, 0,0, 0,0]
    action = test_policy.action(info_state)
    print ('info_state',info_state,'action',action, 'shoule be 1')
