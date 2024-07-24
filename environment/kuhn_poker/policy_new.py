import numpy as np


class KuhnPokerPolicy:
    # Player 2 policy
    def __init__(self, probs):
        self.xi = probs[0]  # probability of bet when self card is J
        self.eta = probs[1] # probability of call when self card is Q

    def __call__(self, obs):
        stage = np.argmax(obs[:7])
        self_card = np.argmax(obs[7:10])
        if stage == 0:
            # faces check
            if self_card == 0:
                # J
                if np.random.rand() < self.xi:
                    # bet
                    return 1
                else:
                    # check
                    return 0
            elif self_card == 1:
                # Q
                return 0
            elif self_card == 2:
                # K
                return 1
            else:
                raise ValueError("Wrong self card")
        elif stage == 1:
            # faces bet
            if self_card == 0:
                # J
                return 0
            elif self_card == 1:
                # Q
                if np.random.rand() < self.eta:
                    # call
                    return 1
                else:
                    # fold
                    return 0
            elif self_card == 2:
                # K
                return 1
            else:
                raise ValueError("Wrong self card")
        else:
            raise ValueError("Wrong stage")

    def reset(self):
        pass

    def __str__(self):
        return f'KuhnPokerPolicy(probs={self.xi, self.eta})'
    
    def get_return_complex(self,probs):
        # the average of 6 possible dealt_cards combo
        # the game tree of playe 1 is not simplified
        # xi: player 2 bet J
        # eta: player 2 call Q
        xi = self.xi
        eta = self.eta
        # JQ
        r1 = (1-probs[0])*(-1) + probs[0]*(eta*(-2)+(1-eta)*1) #check check;bet call;bet fold
        # JK
        r2 = (1-probs[0])*(probs[3]*(-2)+(1-probs[3])*(-1)) + probs[0]*(-2) #check bet call;check bet fold;bet call
        # QJ
        r3 = (1-probs[1])*(xi*(probs[4]*2-(1-probs[4]))+(1-xi)) + probs[1] #check bet call;check bet fold;check check;bet fold
        # QK
        r4 = (1-probs[1])*(probs[4]*(-2)+(1-probs[4])*(-1)) + probs[1]*(-2) #check bet call;check bet fold;bet call
        # KJ
        r5 = (1-probs[2])*(xi*(probs[5]*2-(1-probs[5]))+(1-xi)) + probs[2] #check bet call;check bet fold;check check;bet fold
        # KQ
        r6 = (1-probs[2]) + probs[2]*(eta*(2)+(1-eta)*1) #check check;bet call;bet fold
        return (r1+r2+r3+r4+r5+r6)/6

    def get_return(self,alpha,beta,gamma):
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
        r1 = -(1-alpha) + alpha*(1-eta) - 2*alpha*eta
        # JK
        r2 = -(1-alpha) - 2*alpha
        # QJ
        r3 = (1-xi) + xi*(beta*2-(1-beta))
        # QK
        r4 = -(1-beta) - 2*beta
        # KJ
        r5 = (1-gamma)*(1-xi) + 2*((1-gamma)*xi) + gamma
        # KQ
        r6 = (1-gamma) + gamma*(1-eta+2*eta)
        return (r1+r2+r3+r4+r5+r6)/6

    def get_best_response(self):
        xi = self.xi
        eta = self.eta
        alpha, beta, gamma = -1, -1, -1
        if eta <= 1 / 3 and xi <= 1 / 3:
            if eta < xi:
                # S5
                alpha, beta, gamma = 1, 0, 0
            else:
                # S6
                alpha, beta, gamma = 1, 0, 1
        elif eta <= 1 / 3 and xi >= 1 / 3:
            # S7
            alpha, beta, gamma = 1, 1, 0
        elif eta >= 1 / 3 and xi <= 1 / 3:
            # S2
            alpha, beta, gamma = 0, 0, 1
        elif eta >= 1 / 3 and xi >= 1 / 3:
            if eta < xi:
                # S3
                alpha, beta, gamma = 0, 1, 0
            else:
                # S4
                alpha, beta, gamma = 0, 1, 1
        return alpha, beta, gamma

    def get_best_response_return(self):
        alpha, beta, gamma = self.get_best_response()

        return self.get_return(alpha,beta,gamma)


def generate_policy_pool(num_policies):
    old_state = np.random.get_state()
    np.random.seed(1)
    pool = [KuhnPokerPolicy(np.random.rand(2)) for _ in range(num_policies)]
    np.random.set_state(old_state)
    return pool


test_probs = [[0.25,0.67],[0.75,0.8],[0.67,0.4],[0.5,0.29],[0.28,0.10],[0.17,0.2],[1/3,1/3]]
test_policies = [KuhnPokerPolicy(probs) for probs in test_probs]


def get_train_eval_pool(args):
    assert args.rule_based_opponents == args.train_pool_size, \
        f'Only rule-based opponents are supported in KuhnPoker, so the number of rule policies ' \
        f'{args.rule_based_opponents} must be equal to total train pool size {args.train_pool_size}'
    all_pool = generate_policy_pool(args.train_pool_size + args.eval_pool_size)
    return all_pool[:args.train_pool_size], all_pool[args.train_pool_size:]
